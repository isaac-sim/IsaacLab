# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import trimesh
import warp as wp

from pxr import Usd, UsdPhysics

from ... import cloner
from ... import sim as sim_utils
from ...scene_data import SceneDataFormat
from ...utils.mesh import PRIMITIVE_MESH_TYPES, create_trimesh_from_geom_mesh, create_trimesh_from_geom_shape
from ...utils.string import resolve_matching_names
from ...utils.warp import ProxyArray, convert_to_warp_mesh
from ...utils.warp import kernels as warp_kernels
from .base_ray_caster import BaseRayCaster
from .kernels import copy_mesh_transforms_to_table_kernel, fill_ray_hits_distance_inf_kernel
from .multi_mesh_ray_caster_data import MultiMeshRayCasterData

if TYPE_CHECKING:
    from .multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg

logger = logging.getLogger(__name__)


def _matrix_from_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    """Return a rotation matrix from an ``(x, y, z, w)`` quaternion."""
    x, y, z, w = quat
    two_s = 2.0 / np.dot(quat, quat)
    return np.array(
        [
            [1.0 - two_s * (y * y + z * z), two_s * (x * y - z * w), two_s * (x * z + y * w)],
            [two_s * (x * y + z * w), 1.0 - two_s * (x * x + z * z), two_s * (y * z - x * w)],
            [two_s * (x * z - y * w), two_s * (y * z + x * w), 1.0 - two_s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class BaseMultiMeshRayCaster(BaseRayCaster):
    """A multi-mesh ray-casting sensor.

    The ray-caster uses a set of rays to detect collisions with meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor can be configured to ray-cast against
    a set of meshes with a given ray pattern.

    The meshes are parsed from the list of primitive paths provided in the configuration. These are then
    converted to warp meshes and stored in the :attr:`meshes` dictionary. The ray-caster then ray-casts
    against these warp meshes using the ray pattern provided in the configuration.

    Compared to the default RayCaster, the MultiMeshRayCaster provides additional functionality and flexibility as
    an extension of the default RayCaster with the following enhancements:

    - Raycasting against multiple target types : Supports primitive shapes (spheres, cubes, etc.) as well as arbitrary
      meshes.
    - Dynamic mesh tracking : Keeps track of specified meshes, enabling raycasting against moving parts
      (e.g., robot links, articulated bodies, or dynamic obstacles).
    - Memory-efficient caching : Avoids redundant memory usage by reusing mesh data across environments.

    .. warning::
        **Known limitation (multi-mesh closest-hit resolution):** When two meshes produce a
        hit at the exact same distance for a given ray, the ``atomic_min`` + equality-check
        pattern in the raycasting kernel is not fully thread-safe. The hit *position* is always
        correct, but auxiliary outputs (normals, face IDs, mesh IDs) may originate from
        different meshes for the affected ray. This requires an exact floating-point tie and is
        rare in practice. See `warp#1058 <https://github.com/NVIDIA/warp/issues/1058>`_ for
        upstream progress on a thread-safe ``atomic_min`` return value.

    Example usage to raycast against the visual meshes of a robot (e.g. ANYmal):

    .. code-block:: python

        ray_caster_cfg = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            mesh_prim_paths=[
                "/World/Ground",
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/LF_[^/]*/visuals"),
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/RF_[^/]*/visuals"),
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/LH_[^/]*/visuals"),
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/RH_[^/]*/visuals"),
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/base/visuals"),
            ],
            ray_alignment="world",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=(2.5, 2.5), direction=(0, 0, -1)),
        )

    """

    cfg: MultiMeshRayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: MultiMeshRayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)

        self._num_meshes_per_env: dict[str, int] = {}

        self._raycast_targets_cfg: list[MultiMeshRayCasterCfg.RaycastTargetCfg] = []
        for target in self.cfg.mesh_prim_paths:
            if isinstance(target, str):
                target_cfg = cfg.RaycastTargetCfg(prim_expr=target, track_mesh_transforms=False)
            else:
                target_cfg = target
            target_cfg.prim_expr = cloner.expand_env_regex_ns(target_cfg.prim_expr)
            self._raycast_targets_cfg.append(target_cfg)

        self._data = MultiMeshRayCasterData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {self._num_envs} x {sum(self._num_meshes_per_env.values())} \n"
            f"\tnumber of sensors    : {self._view_count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view_count}"
        )

    """
    Properties
    """

    @property
    def data(self) -> MultiMeshRayCasterData:
        self._update_outdated_buffers()
        return self._data

    """
    Implementation.
    """

    def _initialize_warp_meshes(self):
        """Initialize mesh buffers from the ClonePlan when env-scoped, else from the stage."""
        plan = self._sim.get_clone_plan()
        target_records = []
        self._track_mesh_transforms = any(target.track_mesh_transforms for target in self._raycast_targets_cfg)
        self._mesh_transforms = SceneDataFormat.Transform()

        # Build one per-env mesh list for each configured raycast target.
        for target_cfg in self._raycast_targets_cfg:
            records_per_env = self._build_mesh_records(target_cfg, plan)
            self._num_meshes_per_env[target_cfg.prim_expr] = max(len(records) for records in records_per_env)
            target_records.append(records_per_env)

        dummy_mesh_id = next((record[0] for target in target_records for records in target for record in records), None)
        if dummy_mesh_id is None:
            raise RuntimeError(
                f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
            )

        total_meshes_per_env = sum(
            self._num_meshes_per_env[target_cfg.prim_expr] for target_cfg in self._raycast_targets_cfg
        )
        mesh_ids = np.full((self._num_envs, total_meshes_per_env), dummy_mesh_id, dtype=np.uint64)
        mesh_positions = np.full((self._num_envs, total_meshes_per_env, 3), 1.0e9, dtype=np.float32)
        mesh_orientations = np.zeros((self._num_envs, total_meshes_per_env, 4), dtype=np.float32)
        mesh_orientations[..., 3] = 1.0
        body_indices = np.full(mesh_ids.shape, -1, dtype=np.int32)
        body_paths = self._scene_data_provider.backend.transform_paths if self._track_mesh_transforms else ()
        body_indices_by_path = {path: index for index, path in enumerate(body_paths)}

        mesh_offset = 0
        for target_cfg, records_per_env in zip(self._raycast_targets_cfg, target_records, strict=True):
            target_width = self._num_meshes_per_env[target_cfg.prim_expr]
            for env_id, records in enumerate(records_per_env):
                if not records:
                    continue
                count = len(records)
                record_mesh_ids, record_positions, record_orientations, record_bodies = zip(*records)
                target_slice = slice(mesh_offset, mesh_offset + count)
                mesh_ids[env_id, target_slice] = np.asarray(record_mesh_ids, dtype=np.uint64)
                mesh_positions[env_id, target_slice] = np.asarray(record_positions, dtype=np.float32)
                mesh_orientations[env_id, target_slice] = np.asarray(record_orientations, dtype=np.float32)
                for mesh_index, path in enumerate(record_bodies, start=mesh_offset):
                    if path is not None:
                        try:
                            body_indices[env_id, mesh_index] = body_indices_by_path[path]
                        except KeyError:
                            raise RuntimeError(
                                f"Cannot track ray-cast target '{target_cfg.prim_expr}': body '{path}'"
                                " has no scene-data transform."
                            ) from None
            mesh_offset += target_width

        self._mesh_ids_wp = wp.array2d(mesh_ids, dtype=wp.uint64, device=self.device)
        self._mesh_positions_w = wp.array2d(mesh_positions, dtype=wp.vec3f, device=self.device)
        self._mesh_orientations_w = wp.array2d(mesh_orientations, dtype=wp.quatf, device=self.device)
        self._mesh_body_indices = wp.array2d(body_indices, dtype=wp.int32, device=self.device)

    def _build_mesh_records(  # noqa: C901
        self, target_cfg: MultiMeshRayCasterCfg.RaycastTargetCfg, plan: cloner.ClonePlan | None
    ):
        """Build mesh records for the target configuration."""
        records_per_env = [[] for _ in range(self._num_envs)]
        target_in_plan = False
        has_rigid_body_api = lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI)  # noqa: E731
        # Prefer ClonePlan data for env-scoped targets; destination USD prims may not exist.
        if plan is not None:
            sources = cloner.path.get_asset_prototype_paths(plan)
            templates, starts, worlds, world_starts = cloner.path.get_world_prototype_asset_templates(
                plan, include_world_indices=True
            )
            matches = []
            # Match populated world prototypes, excluding the shared-world slice.
            for group in np.flatnonzero(np.diff(world_starts[1:])) + 1:
                for index in range(*starts[group : group + 2]):
                    if (matched := cloner.path.match(target_cfg.prim_expr, templates[index])) is not None:
                        matches.append((group, index, matched))
            suffix = min((matched.suffix for _, _, matched in matches), key=len, default=None)
            for group, index, matched in matches:
                if matched.suffix != suffix:
                    continue
                env_ids = worlds[world_starts[group] : world_starts[group + 1]]
                env_ids = env_ids[
                    resolve_matching_names(matched.instance, env_ids.astype(str), raise_when_no_match=False)[0]
                ]
                if not len(env_ids):
                    continue
                source_root, destination_template = sources[plan.topology.world_prototypes[index]], templates[index]
                source_path = source_root + suffix
                target_in_plan = True

                # Load meshes from the authored source entry.
                source_pattern = re.compile(source_path)
                source_prims = sim_utils.get_all_matching_child_prims(
                    source_root, lambda prim: source_pattern.fullmatch(prim.GetPath().pathString) is not None
                )
                if not source_prims:
                    # A static target may exist in only some prototypes; their environments hold no copy.
                    if not target_cfg.track_mesh_transforms:
                        continue
                    raise RuntimeError(f"No ClonePlan source prims matched '{source_path}'.")
                if not target_cfg.track_mesh_transforms:
                    stage = source_prims[0].GetStage()
                    source_env = cloner.path.match(source_root, plan.env_template)
                    source_anchor = (
                        stage.GetPrimAtPath(plan.env_template.format(source_env.instance)) if source_env else None
                    )
                    source_records = [
                        (str(prim.GetPath()), prim, *sim_utils.resolve_prim_pose(prim, source_anchor))
                        for prim in source_prims
                    ]
                    for env_id in env_ids:
                        origin = plan.positions[env_id] if plan.positions is not None else 0.0
                        for source_path, source_prim, local_pos, local_quat in source_records:
                            dest_path = cloner.path.rebase(
                                source_path, source_root, destination_template.format(env_id)
                            )
                            dest_prim = stage.GetPrimAtPath(dest_path)
                            if dest_prim.IsValid() and dest_path != source_path:
                                mesh_id = self._load_target_prim_warp_mesh(dest_prim, target_cfg)
                                pos, quat = sim_utils.resolve_prim_pose(dest_prim)
                            else:
                                # The source USD prim is a prototype, even if its path is a destination.
                                mesh_id = self._load_target_prim_warp_mesh(source_prim, target_cfg)
                                pos, quat = tuple(np.add(local_pos, origin)), local_quat
                            records_per_env[env_id].append((mesh_id, pos, quat, None))
                    continue

                source_records = []
                for source_prim in source_prims:
                    source_prim_path = str(source_prim.GetPath())
                    # Use a bounded rigid-body ancestor when the match is below a body; otherwise
                    # enumerate rigid bodies under the match, including the matched prim itself.
                    owner_prim = None
                    rigid_body_records: list[tuple[Usd.Prim, Usd.Prim]]
                    if not source_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                        owner_prim = sim_utils.get_first_matching_ancestor_prim(source_prim_path, has_rigid_body_api)
                    owner_path = None if owner_prim is None else str(owner_prim.GetPath())
                    if owner_path and (owner_path == source_root or owner_path.startswith(f"{source_root}/")):
                        rigid_body_records = [(source_prim, owner_prim)]
                    else:
                        rigid_body_records = [
                            (p, p) for p in sim_utils.get_all_matching_child_prims(source_prim_path, has_rigid_body_api)
                        ]
                    if not rigid_body_records:
                        raise RuntimeError(
                            f"Cannot track ClonePlan target '{target_cfg.prim_expr}' because source prim "
                            f"'{source_prim.GetPath()}' has no rigid-body ancestor or descendant."
                        )
                    for geometry_prim, owner_prim in rigid_body_records:
                        mesh_id = self._load_target_prim_warp_mesh(geometry_prim, target_cfg, reference_prim=owner_prim)
                        owner_path = str(owner_prim.GetPath())
                        owner_suffix = cloner.path.relative_to(owner_path, source_root)
                        if owner_suffix is None:
                            raise RuntimeError(
                                f"Tracked target owner '{owner_path}' is not under ClonePlan source root "
                                f"'{source_root}'."
                            )
                        source_records.append((mesh_id, owner_suffix))

                # Bind each mesh to its exact native body, including multiple meshes on one body.
                for env_id in env_ids:
                    for mesh_id, owner_suffix in source_records:
                        body_path = destination_template.format(env_id) + owner_suffix
                        records_per_env[env_id].append(
                            (mesh_id, (1.0e9, 1.0e9, 1.0e9), (0.0, 0.0, 0.0, 1.0), body_path)
                        )

            if target_in_plan:
                if not target_cfg.track_mesh_transforms and not any(records_per_env):
                    raise RuntimeError(f"No ClonePlan source prims matched '{target_cfg.prim_expr}'.")
                return records_per_env

        # Fall back to authored USD prims for global targets and scenes without ClonePlan data.
        target_prims = sim_utils.find_matching_prims(target_cfg.prim_expr)
        if not target_prims:
            raise RuntimeError(f"Failed to find a prim at path expression: {target_cfg.prim_expr}")

        records = []
        for target_prim in target_prims:
            reference_prim = target_prim
            if target_cfg.track_mesh_transforms:
                reference_prim = sim_utils.get_first_matching_ancestor_prim(target_prim.GetPath(), has_rigid_body_api)
                if reference_prim is None:
                    raise RuntimeError(
                        f"Cannot track non-physics ray-cast target '{target_cfg.prim_expr}'. "
                        "Set track_mesh_transforms=False for static targets, or apply RigidBodyAPI to dynamic targets."
                    )

            mesh_id = self._load_target_prim_warp_mesh(target_prim, target_cfg, reference_prim=reference_prim)
            body_path = str(reference_prim.GetPath()) if target_cfg.track_mesh_transforms else None
            records.append((mesh_id, *sim_utils.resolve_prim_pose(reference_prim), body_path))

        if len(records) == 1:
            return [list(records) for _ in range(self._num_envs)]

        # Multiple USD matches are expected to be laid out evenly by environment.
        if len(records) % self._num_envs != 0:
            raise RuntimeError(
                f"Target expression '{target_cfg.prim_expr}' matched {len(records)} mesh records, "
                f"which cannot be evenly partitioned across {self._num_envs} environments."
            )
        n_meshes = len(records) // self._num_envs
        records_per_env = [records[env_id * n_meshes : (env_id + 1) * n_meshes] for env_id in range(self._num_envs)]

        return records_per_env

    def _load_target_prim_warp_mesh(self, target_prim, target_cfg, reference_prim=None) -> int:
        reference_prim = target_prim if reference_prim is None else reference_prim
        prim_key = (f"{target_prim.GetPath()}@{reference_prim.GetPath()}", self._device)
        if prim_key in BaseMultiMeshRayCaster.meshes:
            return BaseMultiMeshRayCaster.meshes[prim_key].id

        mesh_prims = sim_utils.get_all_matching_child_prims(
            target_prim.GetPath(), lambda prim: prim.GetTypeName() in PRIMITIVE_MESH_TYPES + ["Mesh"]
        )
        if len(mesh_prims) == 0:
            raise RuntimeError(
                f"No mesh prims found at path: {target_prim.GetPath()} with supported types:"
                f" {PRIMITIVE_MESH_TYPES + ['Mesh']}"
            )

        trimesh_meshes = []
        for mesh_prim in mesh_prims:
            if mesh_prim is None or not mesh_prim.IsValid():
                raise RuntimeError(f"Invalid mesh prim path: {target_prim}")
            if mesh_prim.GetTypeName() == "Mesh":
                mesh = create_trimesh_from_geom_mesh(mesh_prim)
            else:
                mesh = create_trimesh_from_geom_shape(mesh_prim)
            mesh.apply_scale(sim_utils.resolve_prim_scale(mesh_prim))
            relative_pos, relative_quat = sim_utils.resolve_prim_pose(mesh_prim, reference_prim)
            relative_pos = np.asarray(relative_pos, dtype=np.float64)
            relative_quat = np.asarray(relative_quat, dtype=np.float64)
            transform = np.eye(4)
            transform[:3, :3] = _matrix_from_quat_xyzw(relative_quat)
            transform[:3, 3] = relative_pos
            mesh.apply_transform(transform)
            trimesh_meshes.append(mesh)

        if len(trimesh_meshes) == 1:
            trimesh_mesh = trimesh_meshes[0]
        elif target_cfg.merge_prim_meshes:
            trimesh_mesh = trimesh.util.concatenate(trimesh_meshes)
        else:
            raise RuntimeError(
                f"Multiple mesh prims found at path: {target_prim.GetPath()} but merging is disabled. Please"
                " enable `merge_prim_meshes` in the configuration or specify each mesh separately."
            )

        wp_mesh = convert_to_warp_mesh(trimesh_mesh.vertices, trimesh_mesh.faces, device=self._device)
        BaseMultiMeshRayCaster.meshes[prim_key] = wp_mesh
        logger.info(
            f"Read '{len(mesh_prims)}' mesh prims under path '{target_prim.GetPath()}' with"
            f" {len(trimesh_mesh.vertices)} vertices and {len(trimesh_mesh.faces)} faces."
        )
        return wp_mesh.id

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        # Persistent buffer for tracking closest-hit distance across meshes (for atomic_min)
        self._ray_distance_wp = wp.empty((self._view_count, self.num_rays), dtype=wp.float32, device=self._device)
        if self.cfg.update_mesh_ids:
            self._data.ray_mesh_ids = ProxyArray(
                wp.zeros((self._view_count, self.num_rays), dtype=wp.int16, device=self._device)
            )
        else:
            # Dummy 1×1 buffer so the kernel launch always has a valid array to bind
            self._ray_mesh_id_wp = wp.empty((1, 1), dtype=wp.int16, device=self._device)
        # Persistent dummy buffers for unused kernel outputs; allocated once to avoid per-step allocations.
        self._dummy_normal_wp = wp.empty((1, 1), dtype=wp.vec3, device=self._device)
        self._dummy_face_id_wp = wp.empty((1, 1), dtype=wp.int32, device=self._device)

    def _update_mesh_transforms(self) -> None:
        """Gather live body poses into bound mesh slots; static and padded slots remain unchanged."""
        if not self._track_mesh_transforms:
            return
        self._scene_data_provider.get_transforms(self._mesh_transforms)
        wp.launch(
            copy_mesh_transforms_to_table_kernel,
            dim=self._mesh_body_indices.shape,
            inputs=[self._mesh_transforms.transforms, self._mesh_body_indices],
            outputs=[self._mesh_positions_w, self._mesh_orientations_w],
            device=self.device,
        )

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        self._update_ray_infos(env_mask)
        self._update_mesh_transforms()

        # Fill output and distance buffers with inf for masked environments
        wp.launch(
            fill_ray_hits_distance_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, False],
            outputs=[self._data._ray_hits_w, self._ray_distance_wp, self._dummy_normal_wp],
            device=self._device,
        )

        # Ray-cast against all meshes; closest hit wins via atomic_min on ray_distance.
        wp.launch(
            warp_kernels.raycast_dynamic_meshes_kernel,
            dim=(self._mesh_ids_wp.shape[1], self._num_envs, self.num_rays),
            inputs=[
                env_mask,
                self._mesh_ids_wp,
                self._ray_starts_w,
                self._ray_directions_w,
                self._data._ray_hits_w,
                self._ray_distance_wp,
                self._dummy_normal_wp,
                self._dummy_face_id_wp,
                self._data.ray_mesh_ids.warp if self.cfg.update_mesh_ids else self._ray_mesh_id_wp,
                self._mesh_positions_w,
                self._mesh_orientations_w,
                float(self.cfg.max_distance),
                int(False),  # return_normal
                int(False),  # return_face_id
                int(self.cfg.update_mesh_ids),
            ],
            device=self._device,
        )
