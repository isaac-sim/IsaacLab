# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
import warp as wp

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.utils.math import matrix_from_quat
from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES, create_trimesh_from_geom_mesh, create_trimesh_from_geom_shape
from isaaclab.utils.warp import convert_to_warp_mesh
from isaaclab.utils.warp import kernels as warp_kernels

from .base_ray_caster import BaseRayCaster
from .kernels import fill_float2d_masked_kernel, fill_vec3_inf_kernel
from .multi_mesh_ray_caster_data import MultiMeshRayCasterData

if TYPE_CHECKING:
    from .multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg

logger = logging.getLogger(__name__)


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
                self._raycast_targets_cfg.append(cfg.RaycastTargetCfg(prim_expr=target, track_mesh_transforms=False))
            else:
                self._raycast_targets_cfg.append(target)

        for cfg in self._raycast_targets_cfg:
            cfg.prim_expr = cfg.prim_expr.format(ENV_REGEX_NS="/World/envs/env_.*")

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
        """Initialize mesh buffers, using ClonePlan rows for env-scoped targets when available."""
        plan = SimulationContext.instance().get_clone_plan() if SimulationContext.instance() is not None else None
        if plan is None or not any(
            self._collect_clone_plan_matches(plan, cfg.prim_expr) for cfg in self._raycast_targets_cfg
        ):
            self._initialize_warp_meshes_from_stage()
            return
        self._initialize_warp_meshes_from_clone_plan(plan)

    def _initialize_warp_meshes_from_clone_plan(self, plan) -> None:
        """Initialize rectangular mesh buffers from ClonePlan source rows.

        The current PR keeps the existing rectangular kernel ABI. Environments
        with fewer meshes than the target maximum are padded with a valid mesh
        placed far outside the ray-cast range. A follow-up PR can replace this
        padded representation with a new kernel.
        """
        target_records_by_expr: dict[
            str, list[list[tuple[int, tuple[float, float, float], tuple[float, float, float, float]]]]
        ] = {}
        dummy_mesh_id: int | None = None
        self._mesh_views = []

        for target_cfg in self._raycast_targets_cfg:
            records_per_env: list[list[tuple[int, tuple[float, float, float], tuple[float, float, float, float]]]] = [
                [] for _ in range(self._num_envs)
            ]
            matches = self._collect_clone_plan_matches(plan, target_cfg.prim_expr)
            if matches:
                for row, source_root, source_expr in matches:
                    target_prims = sim_utils.find_matching_prims(source_expr)
                    if len(target_prims) == 0:
                        raise RuntimeError(
                            f"ClonePlan row source '{source_root}' matched target '{target_cfg.prim_expr}', "
                            f"but no prototype prims matched '{source_expr}'."
                        )
                    prototype_records = []
                    for target_prim in target_prims:
                        mesh_id = self._load_target_prim_warp_mesh(target_prim, target_cfg)
                        dummy_mesh_id = mesh_id if dummy_mesh_id is None else dummy_mesh_id
                        source_root_prim = self.stage.GetPrimAtPath(source_root)
                        local_pos, local_quat = sim_utils.resolve_prim_pose(target_prim, source_root_prim)
                        prototype_records.append((mesh_id, local_pos, local_quat))

                    active_envs = plan.clone_mask[row].nonzero(as_tuple=False).squeeze(-1).tolist()
                    for env_id in active_envs:
                        root_pos_t, root_quat_t = self._clone_plan_destination_pose(
                            source_root, plan.destinations[row], env_id
                        )
                        for mesh_id, local_pos, local_quat in prototype_records:
                            local_pos_t = torch.tensor([local_pos], dtype=torch.float32, device=self.device)
                            local_quat_t = torch.tensor([local_quat], dtype=torch.float32, device=self.device)
                            mesh_pos_t, mesh_quat_t = math_utils.combine_frame_transforms(
                                root_pos_t, root_quat_t, local_pos_t, local_quat_t
                            )
                            records_per_env[env_id].append(
                                (
                                    mesh_id,
                                    tuple(float(v) for v in mesh_pos_t[0].tolist()),
                                    tuple(float(v) for v in mesh_quat_t[0].tolist()),
                                )
                            )
            else:
                target_prims = sim_utils.find_matching_prims(target_cfg.prim_expr)
                if len(target_prims) == 0:
                    raise RuntimeError(f"Failed to find a prim at path expression: {target_cfg.prim_expr}")
                records = []
                for target_prim in target_prims:
                    mesh_id = self._load_target_prim_warp_mesh(target_prim, target_cfg)
                    dummy_mesh_id = mesh_id if dummy_mesh_id is None else dummy_mesh_id
                    pos, quat = sim_utils.resolve_prim_pose(target_prim)
                    records.append((mesh_id, tuple(float(v) for v in pos), tuple(float(v) for v in quat)))
                for env_id in range(self._num_envs):
                    records_per_env[env_id].extend(records)

            self._num_meshes_per_env[target_cfg.prim_expr] = max(len(records) for records in records_per_env)
            target_records_by_expr[target_cfg.prim_expr] = records_per_env
            self._mesh_views.append(
                self._create_tracked_target_view(target_cfg.prim_expr) if target_cfg.track_mesh_transforms else None
            )

        self._install_rectangular_mesh_table(target_records_by_expr, dummy_mesh_id)

    def _collect_clone_plan_matches(self, plan, target_expr: str) -> list[tuple[int, str, str]]:
        target_env0 = _target_expr_for_env(target_expr, 0)
        matches: list[tuple[int, str, str]] = []
        for row, (source_root, destination_template) in enumerate(zip(plan.sources, plan.destinations)):
            if "{}" not in destination_template:
                continue
            destination_env0 = destination_template.format(0)
            if target_env0 == destination_env0:
                relative_expr = ""
            elif target_env0.startswith(destination_env0 + "/"):
                relative_expr = target_env0[len(destination_env0) :]
            else:
                continue
            matches.append((row, source_root, source_root + relative_expr))
        return matches

    def _clone_plan_destination_pose(
        self, source_root: str, destination_template: str, env_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        destination_path = destination_template.format(env_id)
        destination_prim = self.stage.GetPrimAtPath(destination_path)
        if destination_prim is not None and destination_prim.IsValid():
            pos, quat = sim_utils.resolve_prim_pose(destination_prim)
        else:
            # USD-skip cloning may omit the destination object prim entirely.
            # Walk to the nearest authored destination ancestor and preserve the
            # source root's local pose under the matching source ancestor.
            ancestor_path = destination_path
            missing_depth = 0
            ancestor_prim = None
            while ancestor_path and ancestor_path != "/":
                ancestor_path = ancestor_path.rsplit("/", 1)[0] or "/"
                missing_depth += 1
                ancestor_prim = self.stage.GetPrimAtPath(ancestor_path)
                if ancestor_prim is not None and ancestor_prim.IsValid():
                    break

            source_ancestor_path = source_root
            for _ in range(missing_depth):
                source_ancestor_path = source_ancestor_path.rsplit("/", 1)[0] or "/"
            source_root_prim = self.stage.GetPrimAtPath(source_root)
            source_ancestor_prim = self.stage.GetPrimAtPath(source_ancestor_path)
            if (
                ancestor_prim is None
                or not ancestor_prim.IsValid()
                or source_root_prim is None
                or not source_root_prim.IsValid()
                or source_ancestor_prim is None
                or not source_ancestor_prim.IsValid()
            ):
                pos = (0.0, 0.0, 0.0)
                quat = (0.0, 0.0, 0.0, 1.0)
            else:
                ancestor_pos, ancestor_quat = sim_utils.resolve_prim_pose(ancestor_prim)
                local_pos, local_quat = sim_utils.resolve_prim_pose(source_root_prim, source_ancestor_prim)
                ancestor_pos_t = torch.tensor([ancestor_pos], dtype=torch.float32, device=self.device)
                ancestor_quat_t = torch.tensor([ancestor_quat], dtype=torch.float32, device=self.device)
                local_pos_t = torch.tensor([local_pos], dtype=torch.float32, device=self.device)
                local_quat_t = torch.tensor([local_quat], dtype=torch.float32, device=self.device)
                pos_t, quat_t = math_utils.combine_frame_transforms(
                    ancestor_pos_t, ancestor_quat_t, local_pos_t, local_quat_t
                )
                pos = tuple(float(v) for v in pos_t[0].tolist())
                quat = tuple(float(v) for v in quat_t[0].tolist())
        return (
            torch.tensor([pos], dtype=torch.float32, device=self.device),
            torch.tensor([quat], dtype=torch.float32, device=self.device),
        )

    def _load_target_prim_warp_mesh(self, target_prim, target_cfg) -> int:
        prim_key = (str(target_prim.GetPath()), self._device)
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
            mesh = (
                create_trimesh_from_geom_mesh(mesh_prim)
                if mesh_prim.GetTypeName() == "Mesh"
                else create_trimesh_from_geom_shape(mesh_prim)
            )
            mesh.apply_scale(sim_utils.resolve_prim_scale(mesh_prim))
            relative_pos, relative_quat = sim_utils.resolve_prim_pose(mesh_prim, target_prim)
            relative_pos = torch.tensor(relative_pos, dtype=torch.float32)
            relative_quat = torch.tensor(relative_quat, dtype=torch.float32)
            transform = np.eye(4)
            transform[:3, :3] = matrix_from_quat(relative_quat).numpy()
            transform[:3, 3] = relative_pos.numpy()
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

    def _create_tracked_target_view(self, target_prim_path: str):
        raise NotImplementedError("Tracked multi-mesh targets must be implemented by the active physics backend.")

    def _initialize_warp_meshes_from_stage(self):
        """Parse mesh prim expressions from USD and install the rectangular mesh table."""
        target_records_by_expr: dict[
            str, list[list[tuple[int, tuple[float, float, float], tuple[float, float, float, float]]]]
        ] = {}
        dummy_mesh_id: int | None = None
        self._mesh_views = []

        for target_cfg in self._raycast_targets_cfg:
            target_prims = sim_utils.find_matching_prims(target_cfg.prim_expr)
            if len(target_prims) == 0:
                raise RuntimeError(f"Failed to find a prim at path expression: {target_cfg.prim_expr}")

            records = []
            for target_prim in target_prims:
                mesh_id = self._load_target_prim_warp_mesh(target_prim, target_cfg)
                dummy_mesh_id = mesh_id if dummy_mesh_id is None else dummy_mesh_id
                pos, quat = sim_utils.resolve_prim_pose(target_prim)
                records.append((mesh_id, tuple(float(v) for v in pos), tuple(float(v) for v in quat)))

            if len(target_prims) == 1:
                per_env_records = [records for _ in range(self._num_envs)]
            else:
                if len(records) % self._num_envs != 0:
                    raise RuntimeError(
                        f"Target expression '{target_cfg.prim_expr}' matched {len(records)} mesh records, "
                        f"which cannot be evenly partitioned across {self._num_envs} environments."
                    )
                n_meshes = len(records) // self._num_envs
                per_env_records = [records[i * n_meshes : (i + 1) * n_meshes] for i in range(self._num_envs)]

            self._num_meshes_per_env[target_cfg.prim_expr] = max(len(env_records) for env_records in per_env_records)
            target_records_by_expr[target_cfg.prim_expr] = per_env_records
            self._mesh_views.append(
                self._create_tracked_target_view(target_cfg.prim_expr) if target_cfg.track_mesh_transforms else None
            )

        self._install_rectangular_mesh_table(target_records_by_expr, dummy_mesh_id)

    def _install_rectangular_mesh_table(
        self,
        target_records_by_expr: dict[
            str, list[list[tuple[int, tuple[float, float, float], tuple[float, float, float, float]]]]
        ],
        dummy_mesh_id: int | None,
    ) -> None:
        """Pack per-target mesh records into the rectangular table used by the existing kernel."""
        if dummy_mesh_id is None:
            raise RuntimeError(
                f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
            )

        dummy_record = (dummy_mesh_id, (1.0e9, 1.0e9, 1.0e9), (0.0, 0.0, 0.0, 1.0))
        multi_mesh_ids_flattened: list[list[int]] = []
        mesh_positions: list[list[tuple[float, float, float]]] = []
        mesh_orientations: list[list[tuple[float, float, float, float]]] = []

        for env_id in range(self._num_envs):
            meshes_in_env: list[int] = []
            positions_in_env: list[tuple[float, float, float]] = []
            orientations_in_env: list[tuple[float, float, float, float]] = []
            for target_cfg in self._raycast_targets_cfg:
                records = list(target_records_by_expr[target_cfg.prim_expr][env_id])
                records.extend([dummy_record] * (self._num_meshes_per_env[target_cfg.prim_expr] - len(records)))
                for mesh_id, pos, quat in records:
                    meshes_in_env.append(mesh_id)
                    positions_in_env.append(pos)
                    orientations_in_env.append(quat)
            multi_mesh_ids_flattened.append(meshes_in_env)
            mesh_positions.append(positions_in_env)
            mesh_orientations.append(orientations_in_env)

        total_n_meshes_per_env = len(multi_mesh_ids_flattened[0])
        self._mesh_ids_wp = wp.array2d(multi_mesh_ids_flattened, dtype=wp.uint64, device=self.device)
        self._mesh_positions_w = wp.zeros((self._num_envs, total_n_meshes_per_env), dtype=wp.vec3, device=self.device)
        self._mesh_orientations_w = wp.zeros(
            (self._num_envs, total_n_meshes_per_env), dtype=wp.quat, device=self.device
        )
        self._mesh_positions_w_torch = wp.to_torch(self._mesh_positions_w)
        self._mesh_orientations_w_torch = wp.to_torch(self._mesh_orientations_w)
        self._mesh_positions_w_torch[:] = torch.tensor(mesh_positions, dtype=torch.float32, device=self.device)
        self._mesh_orientations_w_torch[:] = torch.tensor(mesh_orientations, dtype=torch.float32, device=self.device)

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        # Persistent buffer for tracking closest-hit distance across meshes (for atomic_min)
        self._ray_distance_w = wp.zeros((self._view_count, self.num_rays), dtype=wp.float32, device=self._device)
        if self.cfg.update_mesh_ids:
            self._ray_mesh_id_w = wp.zeros((self._view_count, self.num_rays), dtype=wp.int16, device=self._device)
            # Zero-copy torch view with the trailing dim expected by consumers of ray_mesh_ids
            self._data.ray_mesh_ids = wp.to_torch(self._ray_mesh_id_w).unsqueeze(-1)
        else:
            # Dummy 1×1 buffer so the kernel launch always has a valid array to bind
            self._ray_mesh_id_w = wp.empty((1, 1), dtype=wp.int16, device=self._device)
        # Persistent dummy buffers for unused kernel outputs; allocated once to avoid per-step allocations.
        self._dummy_normal_w = wp.empty((1, 1), dtype=wp.vec3, device=self._device)
        self._dummy_face_id_w = wp.empty((1, 1), dtype=wp.int32, device=self._device)

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

            # update position of the target meshes
            pos_w, ori_w = view.get_world_poses(None)
            pos_w, ori_w = pos_w.torch, ori_w.torch
            pos_w = pos_w.squeeze(0) if len(pos_w.shape) == 3 else pos_w
            ori_w = ori_w.squeeze(0) if len(ori_w.shape) == 3 else ori_w

            count = getattr(view, "count", pos_w.shape[0])
            if count != 1:
                count = count // self._num_envs
                pos_w = pos_w.view(self._num_envs, count, 3)
                ori_w = ori_w.view(self._num_envs, count, 4)

            self._mesh_positions_w_torch[:, mesh_idx : mesh_idx + count] = pos_w
            self._mesh_orientations_w_torch[:, mesh_idx : mesh_idx + count] = ori_w
            mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        self._update_ray_infos(env_mask)
        self._update_mesh_transforms()

        # Fill output and distance buffers with inf for masked environments
        wp.launch(
            fill_vec3_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, float("inf"), self._data._ray_hits_w],
            device=self._device,
        )
        wp.launch(
            fill_float2d_masked_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, float("inf"), self._ray_distance_w],
            device=self._device,
        )

        n_meshes = self._mesh_ids_wp.shape[1]

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
                self._ray_distance_w,
                self._dummy_normal_w,
                self._dummy_face_id_w,
                self._ray_mesh_id_w,
                self._mesh_positions_w,
                self._mesh_orientations_w,
                float(self.cfg.max_distance),
                int(False),
                int(False),
                int(self.cfg.update_mesh_ids),
            ],
            device=self._device,
        )

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)
        # clear mesh views so they are re-created on the next initialization
        self._mesh_views = []

    def __del__(self):
        super().__del__()
        if BaseRayCaster._instance_count == 0:
            pass


"""
Helper functions
"""


def _registered_points_idx(points: np.ndarray, registered_points: list[np.ndarray | None]) -> int:
    """Check if the points are already registered in the list of registered points.

    Args:
        points: The points to check.
        registered_points: The list of registered points.

    Returns:
        The index of the registered points if found, otherwise -1.
    """
    for idx, reg_points in enumerate(registered_points):
        if reg_points is None:
            continue
        if reg_points.shape == points.shape and (reg_points == points).all():
            return idx
    return -1


def _target_expr_for_env(target_expr: str, env_id: int) -> str:
    """Render the common Isaac Lab env regex for one concrete environment."""
    return re.sub(r"env_\.\*", f"env_{env_id}", target_expr)
