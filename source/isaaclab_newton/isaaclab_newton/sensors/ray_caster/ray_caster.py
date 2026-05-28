# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# pyright: reportInvalidTypeForm=none, reportPrivateUsage=none
import re
from typing import Any

import numpy as np
import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.sensors.ray_caster.kernels import (
    ALIGNMENT_BASE,
    copy_mesh_poses_to_table_kernel,
    update_ray_caster_kernel,
)
from isaaclab.utils.warp import ProxyArray

from isaaclab_newton.physics import NewtonManager


@wp.kernel
def _newton_site_world_poses_kernel(
    site_indices: wp.array(dtype=wp.int32),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    out_pose: wp.array(dtype=wp.transformf),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.quatf),
):
    i = wp.tid()
    site_idx = site_indices[i]
    body_idx = shape_body[site_idx]
    site_xform = shape_transform[site_idx]
    if body_idx == -1:
        world_xform = site_xform
    else:
        world_xform = wp.transform_multiply(body_q[body_idx], site_xform)
    out_pose[i] = world_xform
    out_pos[i] = wp.transform_get_translation(world_xform)
    out_quat[i] = wp.transform_get_rotation(world_xform)


@wp.kernel
def _gather_pose_by_index_kernel(
    indices: wp.array(dtype=wp.int32),
    pos_src: wp.array(dtype=wp.vec3f),
    quat_src: wp.array(dtype=wp.quatf),
    pos_dst: wp.array(dtype=wp.vec3f),
    quat_dst: wp.array(dtype=wp.quatf),
):
    i = wp.tid()
    src_idx = indices[i]
    pos_dst[i] = pos_src[src_idx]
    quat_dst[i] = quat_src[src_idx]


def _find_physics_ancestor(prim):
    """Return the nearest rigid-body ancestor for a sensor or target prim."""
    ancestor = prim
    while ancestor and ancestor.IsValid() and ancestor.GetPath().pathString != "/":
        if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            return ancestor
        ancestor = ancestor.GetParent()
    return None


def _newton_body_pattern(body_path: str) -> str:
    """Convert a concrete env index to a regex wildcard for prototype body matching."""
    body_path = body_path.replace("{}", ".*")
    return re.sub(r"^(/World/envs/)env_\d+/", r"\1env_.*/", body_path)


def _identity_offsets(count: int, device: str) -> tuple[wp.array, wp.array]:
    """Create identity sensor offsets for site poses that already include the offset."""
    offset_pos_wp = wp.zeros(count, dtype=wp.vec3f, device=device)
    identity_quat = np.zeros((count, 4), dtype=np.float32)
    identity_quat[:, 3] = 1.0
    return offset_pos_wp, wp.array(identity_quat, dtype=wp.quatf, device=device)


class _NewtonRayCasterMixin:
    """Newton site registration and pose tracking for ray-caster sensors.

    Sites must be registered during construction so Newton can inject them into
    prototype builders before cloning. Once physics is ready, the mixin resolves
    those labels to concrete site indices and updates the sensor-owned buffers
    directly from Newton model/state arrays.
    """

    @property
    def count(self: Any) -> int:
        """Number of resolved Newton sites tracked as sensor frames."""
        return self._view_count

    def __init__(self: Any, cfg):
        """Register sensor and dynamic target sites before cloning occurs."""
        super().__init__(cfg)  # pyright: ignore[reportCallIssue]
        self._sensor_site_labels = self._register_sites_for_expr(self.cfg.prim_path)
        self._tracked_site_labels_by_expr: dict[str | tuple[str, ...], list[str]] = {}
        for target_cfg in getattr(self, "_raycast_targets_cfg", []):
            if target_cfg.track_mesh_transforms:
                owner_exprs = self._resolve_target_owner_exprs(target_cfg.prim_expr)
                labels = self._register_target_sites_for_exprs(owner_exprs)
                self._tracked_site_labels_by_expr[target_cfg.prim_expr] = labels
                self._tracked_site_labels_by_expr[tuple(owner_exprs)] = labels

    def _register_sites_for_expr(self, prim_expr: str) -> list[str]:
        """Register Newton sites for a prim expression and return site labels."""
        prims = sim_utils.find_matching_prims(prim_expr)
        labels: list[str] = []
        if len(prims) == 0:
            identity = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))
            return [NewtonManager.cl_register_site(_newton_body_pattern(prim_expr), identity)]

        for prim in prims:
            body = _find_physics_ancestor(prim)
            if body is None:
                pos, quat = sim_utils.resolve_prim_pose(prim)
                xform = wp.transform(wp.vec3(*[float(v) for v in pos]), wp.quat(*[float(v) for v in quat]))
                labels.append(NewtonManager.cl_register_site(None, xform))
            else:
                pos, quat = sim_utils.resolve_prim_pose(prim, body)
                xform = wp.transform(wp.vec3(*[float(v) for v in pos]), wp.quat(*[float(v) for v in quat]))
                labels.append(NewtonManager.cl_register_site(_newton_body_pattern(str(body.GetPath())), xform))
        # Keep the first copy of each label; cloned envs can report the same prototype site more than once.
        return list(dict.fromkeys(labels))

    def _resolve_target_owner_exprs(self, prim_expr: str) -> list[str]:
        """Resolve mesh target expressions to owning rigid-body expressions."""
        prims = sim_utils.find_matching_prims(prim_expr)
        if len(prims) == 0:
            return [_newton_body_pattern(prim_expr)]

        owner_exprs: list[str] = []
        for prim in prims:
            body = _find_physics_ancestor(prim)
            if body is None:
                raise RuntimeError(
                    f"Cannot track non-physics ray-cast target '{prim_expr}' with Newton. "
                    "Set track_mesh_transforms=False for static targets, or apply RigidBodyAPI to dynamic targets."
                )
            owner_exprs.append(_newton_body_pattern(str(body.GetPath())))
        return list(dict.fromkeys(owner_exprs))

    def _register_target_sites_for_exprs(self, owner_exprs: list[str]) -> list[str]:
        """Register identity-pose Newton sites on target owner bodies."""
        identity = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))
        labels = [NewtonManager.cl_register_site(owner_expr, identity) for owner_expr in owner_exprs]
        return list(dict.fromkeys(labels))

    def _initialize_pose_tracking(self: Any) -> None:
        """Resolve registered site labels and allocate sensor-owned pose buffers."""
        site_indices = self._resolve_site_indices(self._sensor_site_labels, self.cfg.prim_path, self._num_envs)
        # The base classes still use ``self._view.count`` in a few generic
        # places. Point it at the sensor instead of constructing an adapter.
        self._view = self
        self._view_count = len(site_indices)
        self._sensor_site_indices = wp.array(site_indices, dtype=wp.int32, device=self._device)
        self._newton_pose_w = wp.empty(self._view_count, dtype=wp.transformf, device=self._device)
        self._newton_pos_w = ProxyArray(wp.empty(self._view_count, dtype=wp.vec3f, device=self._device))
        self._newton_quat_w = ProxyArray(wp.empty(self._view_count, dtype=wp.quatf, device=self._device))
        self._offset_pos_wp, self._offset_quat_wp = _identity_offsets(self._view_count, self._device)

    def _update_ray_infos(self: Any, env_mask: wp.array):
        """Update Newton site poses and transform local rays in a single ray-caster kernel."""
        self._update_newton_site_transforms(
            self._sensor_site_indices, self._newton_pose_w, self._newton_pos_w.warp, self._newton_quat_w.warp
        )
        pos_w = self._data.pos_w.warp
        quat_w = self._data.quat_w_world.warp if hasattr(self._data, "quat_w_world") else self._data.quat_w.warp
        ray_starts = self.ray_starts.warp if hasattr(self.ray_starts, "warp") else self._ray_starts_local
        ray_directions = (
            self.ray_directions.warp if hasattr(self.ray_directions, "warp") else self._ray_directions_local
        )
        alignment_mode = int(ALIGNMENT_BASE) if hasattr(self._data, "quat_w_world") else self._alignment_mode
        wp.launch(
            update_ray_caster_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                self._newton_pose_w,
                env_mask,
                self._offset_pos_wp,
                self._offset_quat_wp,
                self.drift.warp,
                self.ray_cast_drift.warp,
                ray_starts,
                ray_directions,
                alignment_mode,
            ],
            outputs=[
                pos_w,
                quat_w,
                self._ray_starts_w,
                self._ray_directions_w,
            ],
            device=self._device,
        )

    def get_world_poses(self: Any, indices=None):
        """Return world poses for camera helpers that still use pose tuples."""
        self._update_newton_site_transforms(
            self._sensor_site_indices, self._newton_pose_w, self._newton_pos_w.warp, self._newton_quat_w.warp
        )
        if indices is None:
            return self._newton_pos_w, self._newton_quat_w
        if not isinstance(indices, wp.array):
            indices = wp.array(indices, dtype=wp.int32, device=self._device)
        pos_w = wp.empty(indices.shape[0], dtype=wp.vec3f, device=self._device)
        quat_w = wp.empty(indices.shape[0], dtype=wp.quatf, device=self._device)
        wp.launch(
            _gather_pose_by_index_kernel,
            dim=indices.shape[0],
            inputs=[indices, self._newton_pos_w.warp, self._newton_quat_w.warp],
            outputs=[pos_w, quat_w],
            device=self._device,
        )
        return ProxyArray(pos_w), ProxyArray(quat_w)

    def _create_tracked_target_view(self: Any, target_prim_path: str | list[str]):
        """Resolve dynamic multi-mesh target sites to raw Newton site indices."""
        target_key = tuple(target_prim_path) if isinstance(target_prim_path, list) else target_prim_path
        labels = self._tracked_site_labels_by_expr.get(target_key)
        if labels is None:
            target_exprs = target_prim_path if isinstance(target_prim_path, list) else [target_prim_path]
            labels = self._register_target_sites_for_exprs([_newton_body_pattern(expr) for expr in target_exprs])
            self._tracked_site_labels_by_expr[target_key] = labels
        site_indices = self._resolve_site_indices(labels, str(target_prim_path), self._num_envs)
        return wp.array(site_indices, dtype=wp.int32, device=self._device)

    def _update_mesh_transforms(self: Any) -> None:
        """Refresh dynamic multi-mesh targets directly from Newton sites."""
        if not hasattr(self, "_mesh_views"):
            return
        mesh_idx = 0
        for site_indices, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            site_count = site_indices.shape[0]
            pos_buf = wp.empty(site_count, dtype=wp.vec3f, device=self._device)
            quat_buf = wp.empty(site_count, dtype=wp.quatf, device=self._device)
            pose_buf = wp.empty(site_count, dtype=wp.transformf, device=self._device)
            self._update_newton_site_transforms(site_indices, pose_buf, pos_buf, quat_buf)
            meshes_per_env = site_count
            if site_count != 1:
                # Newton sites arrive as a flat list across envs; the mesh table is indexed per env.
                meshes_per_env = site_count // self._num_envs

            wp.launch(
                copy_mesh_poses_to_table_kernel,
                dim=(self._num_envs, meshes_per_env),
                inputs=[
                    pos_buf,
                    quat_buf,
                    int(meshes_per_env),
                    int(mesh_idx),
                    bool(site_count == 1),
                    self._mesh_positions_w,
                    self._mesh_orientations_w,
                ],
                device=self._device,
            )
            mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]

    def _update_newton_site_transforms(
        self: Any,
        site_indices: wp.array,
        pose_buf: wp.array,
        pos_buf: wp.array,
        quat_buf: wp.array,
    ) -> None:
        """Launch the Newton site pose kernel into caller-provided buffers."""
        model = NewtonManager._model
        state = NewtonManager._state_0
        if model is None or state is None:
            raise RuntimeError("Newton simulation state is not initialized.")
        wp.launch(
            _newton_site_world_poses_kernel,
            dim=site_indices.shape[0],
            inputs=[site_indices, model.shape_body, model.shape_transform, state.body_q],
            outputs=[pose_buf, pos_buf, quat_buf],
            device=self._device,
        )

    @staticmethod
    def _resolve_site_indices(labels: list[str], prim_expr: str, num_envs: int) -> list[int]:
        """Expand registered site labels into per-environment Newton site indices."""
        site_map = NewtonManager._cl_site_index_map
        site_indices: list[int] = []
        for env_idx in range(num_envs):
            for label in labels:
                error_prefix = f"RayCaster target '{prim_expr}' site label '{label}'"
                if label not in site_map:
                    raise ValueError(f"{error_prefix} was not found in NewtonManager._cl_site_index_map.")
                global_idx, per_world = site_map[label]
                env_site_indices = [global_idx] if per_world is None else per_world[env_idx]
                site_indices.extend(env_site_indices)
        return site_indices


class RayCaster(_NewtonRayCasterMixin, BaseRayCaster):
    """Newton ray-caster implementation."""
