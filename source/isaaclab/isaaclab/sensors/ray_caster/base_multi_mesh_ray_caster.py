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

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES, create_trimesh_from_geom_mesh, create_trimesh_from_geom_shape
from isaaclab.utils.warp import ProxyArray, convert_to_warp_mesh
from isaaclab.utils.warp import kernels as warp_kernels

from .base_ray_caster import BaseRayCaster
from .kernels import copy_mesh_poses_to_table_kernel, fill_ray_hits_distance_inf_kernel
from .multi_mesh_ray_caster_data import MultiMeshRayCasterData

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan

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
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/LF_.*/visuals"),
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/RF_.*/visuals"),
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/LH_.*/visuals"),
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/RH_.*/visuals"),
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
            target_cfg.prim_expr = target_cfg.prim_expr.format(ENV_REGEX_NS="/World/envs/env_.*")
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
        sim = SimulationContext.instance()
        plan = sim.get_clone_plan() if sim is not None else None
        target_records_by_expr = {}
        dummy_mesh_id: int | None = None
        self._mesh_views = []

        # Build one per-env mesh list for each configured raycast target.
        for target_cfg in self._raycast_targets_cfg:
            records_per_env, dummy_mesh_id, tracked_target_exprs = self._build_mesh_records(
                target_cfg, plan, dummy_mesh_id
            )
            self._num_meshes_per_env[target_cfg.prim_expr] = max(len(records) for records in records_per_env)
            target_records_by_expr[target_cfg.prim_expr] = records_per_env
            self._mesh_views.append(
                self._create_tracked_target_view(tracked_target_exprs) if target_cfg.track_mesh_transforms else None
            )

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

        mesh_offset = 0
        for target_cfg in self._raycast_targets_cfg:
            records_per_env = target_records_by_expr[target_cfg.prim_expr]
            target_width = self._num_meshes_per_env[target_cfg.prim_expr]
            for env_id, records in enumerate(records_per_env):
                if not records:
                    continue
                count = len(records)
                record_mesh_ids, record_positions, record_orientations = zip(*records)
                target_slice = slice(mesh_offset, mesh_offset + count)
                mesh_ids[env_id, target_slice] = np.asarray(record_mesh_ids, dtype=np.uint64)
                mesh_positions[env_id, target_slice] = np.asarray(record_positions, dtype=np.float32)
                mesh_orientations[env_id, target_slice] = np.asarray(record_orientations, dtype=np.float32)
            mesh_offset += target_width

        self._mesh_ids_wp = wp.array2d(mesh_ids, dtype=wp.uint64, device=self.device)
        self._mesh_positions_w = wp.array2d(mesh_positions, dtype=wp.vec3f, device=self.device)
        self._mesh_orientations_w = wp.array2d(mesh_orientations, dtype=wp.quatf, device=self.device)

    def _build_mesh_records(
        self,
        target_cfg: MultiMeshRayCasterCfg.RaycastTargetCfg,
        plan: ClonePlan | None,
        dummy_mesh_id: int | None,
    ):
        """Build mesh records for the target configuration."""
        records_per_env = [[] for _ in range(self._num_envs)]
        target_in_plan = False
        tracked_target_exprs: list[str] = [target_cfg.prim_expr]

        # Prefer ClonePlan data for env-scoped targets; destination USD prims may not exist.
        if plan is not None and target_cfg.track_mesh_transforms:
            target_path = re.sub(r"env_\.\*", "env_0", target_cfg.prim_expr)
            plan_tracked_target_exprs: list[str] = []
            for row, (source_root, destination_template) in enumerate(zip(plan.sources, plan.destinations)):
                if "{}" not in destination_template:
                    continue

                dest_path = destination_template.format(0)
                suffix = target_path.removeprefix(dest_path)
                if suffix == target_path or (suffix and not suffix.startswith("/")):
                    continue

                target_in_plan = True
                env_ids = plan.clone_mask[row].nonzero(as_tuple=False).squeeze(-1)
                if env_ids.numel() == 0:
                    continue

                # Load meshes from the authored source row.
                source_prims = sim_utils.find_matching_prims(source_root + suffix)
                if not source_prims:
                    raise RuntimeError(f"No ClonePlan source prims matched '{source_root + suffix}'.")

                mesh_ids: list[int] = []
                row_tracked_target_exprs: list[str] = []
                for source_prim in source_prims:
                    owner_prim = source_prim
                    while owner_prim and owner_prim.IsValid() and str(owner_prim.GetPath()) != "/":
                        if owner_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                            break
                        owner_prim = owner_prim.GetParent()
                    if owner_prim is None or not owner_prim.IsValid() or not owner_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        raise RuntimeError(
                            f"Cannot track ClonePlan target '{target_cfg.prim_expr}' because source prim "
                            f"'{source_prim.GetPath()}' has no rigid-body ancestor."
                        )
                    mesh_id = self._load_target_prim_warp_mesh(source_prim, target_cfg, reference_prim=owner_prim)
                    dummy_mesh_id = mesh_id if dummy_mesh_id is None else dummy_mesh_id
                    mesh_ids.append(mesh_id)
                    owner_path = str(owner_prim.GetPath())
                    if owner_path == source_root:
                        owner_suffix = ""
                    elif owner_path.startswith(source_root + "/"):
                        owner_suffix = owner_path[len(source_root) :]
                    else:
                        raise RuntimeError(
                            f"Tracked target owner '{owner_path}' is not under ClonePlan source root '{source_root}'."
                        )
                    row_tracked_target_exprs.append(destination_template.replace("{}", ".*") + owner_suffix)

                if len(row_tracked_target_exprs) > len(plan_tracked_target_exprs):
                    plan_tracked_target_exprs = row_tracked_target_exprs

                # Geometry is selected by ClonePlan; live pose is supplied by backend body/site views.
                for env_id in env_ids.tolist():
                    for mesh_id in mesh_ids:
                        records_per_env[env_id].append((mesh_id, (1.0e9, 1.0e9, 1.0e9), (0.0, 0.0, 0.0, 1.0)))

            if target_in_plan:
                if not plan_tracked_target_exprs:
                    raise RuntimeError(
                        f"No tracked body expressions were resolved for target '{target_cfg.prim_expr}'."
                    )
                return records_per_env, dummy_mesh_id, plan_tracked_target_exprs

        # Fall back to authored USD prims for global targets and scenes without ClonePlan data.
        target_prims = sim_utils.find_matching_prims(target_cfg.prim_expr)
        if not target_prims:
            raise RuntimeError(f"Failed to find a prim at path expression: {target_cfg.prim_expr}")

        records = []
        tracked_target_exprs = []
        for target_prim in target_prims:
            reference_prim = target_prim
            if target_cfg.track_mesh_transforms:
                while reference_prim and reference_prim.IsValid() and str(reference_prim.GetPath()) != "/":
                    if reference_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        break
                    reference_prim = reference_prim.GetParent()
                if (
                    reference_prim is None
                    or not reference_prim.IsValid()
                    or not reference_prim.HasAPI(UsdPhysics.RigidBodyAPI)
                ):
                    raise RuntimeError(
                        f"Cannot track non-physics ray-cast target '{target_cfg.prim_expr}'. "
                        "Set track_mesh_transforms=False for static targets, or apply RigidBodyAPI to dynamic targets."
                    )
                tracked_target_exprs.append(str(reference_prim.GetPath()))

            mesh_id = self._load_target_prim_warp_mesh(target_prim, target_cfg, reference_prim=reference_prim)
            dummy_mesh_id = mesh_id if dummy_mesh_id is None else dummy_mesh_id
            pos, quat = sim_utils.resolve_prim_pose(reference_prim)
            pos = (float(pos[0]), float(pos[1]), float(pos[2]))
            quat = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            records.append((mesh_id, pos, quat))

        if len(records) == 1:
            return [list(records) for _ in range(self._num_envs)], dummy_mesh_id, tracked_target_exprs

        # Multiple USD matches are expected to be laid out evenly by environment.
        if len(records) % self._num_envs != 0:
            raise RuntimeError(
                f"Target expression '{target_cfg.prim_expr}' matched {len(records)} mesh records, "
                f"which cannot be evenly partitioned across {self._num_envs} environments."
            )
        n_meshes = len(records) // self._num_envs
        records_per_env = [records[env_id * n_meshes : (env_id + 1) * n_meshes] for env_id in range(self._num_envs)]

        return records_per_env, dummy_mesh_id, tracked_target_exprs

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

    def _create_tracked_target_view(self, target_prim_paths: str | list[str]):
        raise NotImplementedError("Tracked multi-mesh targets must be implemented by the active physics backend.")

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
        """Update world-frame mesh positions and orientations for dynamically tracked targets.

        Iterates over all tracked views and writes the current world poses into
        the rectangular mesh pose buffers. Static (non-tracked) targets are
        skipped; their initial poses were set during :meth:`_initialize_warp_meshes`.
        """
        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            pos_w, ori_w = view.get_world_poses(None)
            view_count = getattr(view, "count", pos_w.warp.shape[0])
            meshes_per_env = view_count
            if view_count != 1:
                # Backend views return a flat list across envs; the mesh table is indexed per env.
                meshes_per_env = view_count // self._num_envs

            wp.launch(
                copy_mesh_poses_to_table_kernel,
                dim=(self._num_envs, meshes_per_env),
                inputs=[
                    pos_w.warp,
                    ori_w.warp,
                    int(meshes_per_env),
                    int(mesh_idx),
                    bool(view_count == 1),
                    self._mesh_positions_w,
                    self._mesh_orientations_w,
                ],
                device=self._device,
            )
            mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]

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

        n_meshes = self._mesh_ids_wp.shape[1]
        return_normal = False
        return_face_id = False
        write_mesh_ids = self.cfg.update_mesh_ids

        # Ray-cast against all meshes; closest hit wins via atomic_min on ray_distance.
        wp.launch(
            warp_kernels.raycast_dynamic_meshes_kernel,
            dim=(n_meshes, self._num_envs, self.num_rays),
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
                int(return_normal),
                int(return_face_id),
                int(write_mesh_ids),
            ],
            device=self._device,
        )

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)
        # clear mesh views so they are re-created on the next initialization
        self._mesh_views = []
