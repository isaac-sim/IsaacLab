# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.sensors.ray_caster.kernels import copy_mesh_transforms_to_table_kernel

from isaaclab_physx.physics import PhysxManager


def _find_physics_ancestor(prim):
    """Return the nearest rigid-body ancestor for a sensor or target prim."""
    ancestor = prim
    while ancestor and ancestor.IsValid() and ancestor.GetPath().pathString != "/":
        if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            return ancestor
        ancestor = ancestor.GetParent()
    return None


def _body_expr_from_sensor_expr(sensor_expr: str, first_sensor_prim, first_body_prim) -> str:
    """Convert a sensor/target expression to the matching rigid-body expression."""
    sensor_path = first_sensor_prim.GetPath().pathString
    body_path = first_body_prim.GetPath().pathString
    if sensor_path == body_path:
        return sensor_expr
    # Example: ``.../Robot/base/sensor`` target -> ``.../Robot/base`` body view.
    suffix = sensor_path[len(body_path) :]
    if suffix and sensor_expr.endswith(suffix):
        return sensor_expr[: -len(suffix)]
    return body_path


def _physx_body_glob(body_expr: str) -> str:
    """Convert internal env regex/template expressions to PhysX glob syntax."""
    return body_expr.replace("{}", "*").replace(".*", "*")


class _PhysXRayCasterMixin:
    """PhysX pose tracking for ray-caster sensors.

    PhysX can provide live rigid-body transforms after physics is ready. Static
    non-physics prims are cached once at initialization; they are intentionally
    not polled through USD during sensor updates.
    """

    @property
    def count(self: Any) -> int:
        """Number of tracked sensor frames."""
        return self._view_count

    def _initialize_pose_tracking(self: Any) -> None:
        """Initialize direct PhysX body tracking or a cached static pose table."""
        prims = sim_utils.find_matching_prims(self.cfg.prim_path)
        if len(prims) == 0:
            raise RuntimeError(f"No sensor prims matched: {self.cfg.prim_path}")

        # The base classes still use ``self._view.count`` in a few generic
        # places. Point it at the sensor instead of constructing an adapter.
        self._view = self
        body = _find_physics_ancestor(prims[0])
        if body is None:
            self._initialize_static_pose_tracking(prims)
            return

        requested_prim_path = getattr(self, "_requested_prim_path", self.cfg.prim_path)
        # When the public prim path pointed at a rigid body, BaseRayCaster
        # spawned a child sensor prim and preserved the original body path.
        body_expr = (
            requested_prim_path
            if self.cfg.prim_path != requested_prim_path
            else _body_expr_from_sensor_expr(self.cfg.prim_path, prims[0], body)
        )
        physics_sim_view = PhysxManager.get_physics_sim_view()
        if physics_sim_view is None:
            raise RuntimeError("PhysX simulation view is not initialized.")
        self._physx_body_view = physics_sim_view.create_rigid_body_view(body_expr.replace(".*", "*"))
        self._view_count = self._physx_body_view.count

        offset_pos = []
        offset_quat = []
        for prim in prims:
            body_prim = _find_physics_ancestor(prim)
            p, q = sim_utils.resolve_prim_pose(prim, body_prim)
            offset_pos.append(p)
            offset_quat.append(q)
        if len(offset_pos) == 1 and self._view_count > 1:
            offset_pos = offset_pos * self._view_count
            offset_quat = offset_quat * self._view_count
        self._offset_pos_wp = wp.array(offset_pos[: self._view_count], dtype=wp.vec3f, device=self._device)
        self._offset_quat_contiguous = torch.tensor(
            offset_quat[: self._view_count], dtype=torch.float32, device=self._device
        )
        self._offset_quat_wp = wp.from_torch(self._offset_quat_contiguous, dtype=wp.quatf)

    def _initialize_static_pose_tracking(self: Any, prims) -> None:
        """Cache authored poses for non-physics sensor frames."""
        poses = []
        for prim in prims:
            pos, quat = sim_utils.resolve_prim_pose(prim)
            poses.append((*pos, *quat))
        self._static_view_transforms_torch = torch.tensor(poses, dtype=torch.float32, device=self._device).contiguous()
        self._static_view_transforms_wp = wp.from_torch(self._static_view_transforms_torch).view(wp.transformf)
        self._physx_body_view = None
        self._view_count = len(prims)
        self._offset_pos_wp = wp.zeros(self._view_count, dtype=wp.vec3f, device=self._device)
        identity_quat = torch.zeros(self._view_count, 4, device=self._device)
        identity_quat[:, 3] = 1.0
        self._offset_quat_contiguous = identity_quat.contiguous()
        self._offset_quat_wp = wp.from_torch(self._offset_quat_contiguous, dtype=wp.quatf)

    def _get_view_transforms_wp(self: Any) -> wp.array:
        """Return tracked sensor-frame transforms as ``wp.transformf``."""
        if self._physx_body_view is None:
            return self._static_view_transforms_wp
        transforms = self._physx_body_view.get_transforms()
        if isinstance(transforms, wp.array):
            return transforms.view(wp.transformf)
        return wp.from_torch(transforms.contiguous()).view(wp.transformf)

    def get_world_poses(self: Any, indices=None):
        """Return world poses for camera helpers that still use pose tuples."""
        transforms = self._get_view_transforms_wp()
        transforms_t = wp.to_torch(transforms).reshape(-1, 7)
        if indices is not None:
            idx = wp.to_torch(indices).to(dtype=torch.long) if isinstance(indices, wp.array) else indices
            transforms_t = transforms_t[idx]
        return SimpleNamespace(torch=transforms_t[:, 0:3]), SimpleNamespace(torch=transforms_t[:, 3:7])

    def _create_tracked_target_view(self: Any, target_prim_paths: str | list[str]):
        """Create a PhysX rigid-body view for dynamic multi-mesh targets."""
        if isinstance(target_prim_paths, str):
            target_prim_paths = [target_prim_paths]
        body_paths = []
        for target_prim_path in target_prim_paths:
            prims = sim_utils.find_matching_prims(target_prim_path)
            if len(prims) == 0:
                # ClonePlan-backed targets may not have destination mesh prims.
                # In that case BaseMultiMeshRayCaster passes the destination owner-body expression.
                body_paths.append(target_prim_path)
                continue
            for prim in prims:
                body = _find_physics_ancestor(prim)
                if body is None:
                    raise RuntimeError(
                        f"Cannot track non-physics ray-cast target '{target_prim_path}' with PhysX. "
                        "Set track_mesh_transforms=False for static targets, or apply RigidBodyAPI to dynamic targets."
                    )
                body_paths.append(body.GetPath().pathString)

        if len(body_paths) == 0:
            raise RuntimeError(f"No tracked target bodies resolved from: {target_prim_paths}")
        physics_sim_view = PhysxManager.get_physics_sim_view()
        if physics_sim_view is None:
            raise RuntimeError("PhysX simulation view is not initialized.")
        return physics_sim_view.create_rigid_body_view([_physx_body_glob(path) for path in body_paths])

    def _update_mesh_transforms(self: Any) -> None:
        """Refresh dynamic multi-mesh targets directly from PhysX views."""
        if not hasattr(self, "_mesh_views"):
            return
        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            transforms = view.get_transforms()
            transforms_wp = (
                transforms.view(wp.transformf)
                if isinstance(transforms, wp.array)
                else wp.from_torch(transforms.contiguous()).view(wp.transformf)
            )

            view_count = view.count
            meshes_per_env = view_count
            if view_count != 1:
                # PhysX views return a flat list across envs; the mesh table is indexed per env.
                meshes_per_env = view_count // self._num_envs

            wp.launch(
                copy_mesh_transforms_to_table_kernel,
                dim=(self._num_envs, meshes_per_env),
                inputs=[
                    transforms_wp,
                    int(meshes_per_env),
                    int(mesh_idx),
                    bool(view_count == 1),
                    self._mesh_positions_w,
                    self._mesh_orientations_w,
                ],
                device=self._device,
            )
            mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]


class RayCaster(_PhysXRayCasterMixin, BaseRayCaster):
    """PhysX ray-caster implementation."""
