# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# pyright: reportInvalidTypeForm=none, reportPrivateUsage=none
import re
from types import SimpleNamespace
from typing import Any

import torch
import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster

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


def _find_physics_ancestor(prim):
    """Return the nearest rigid-body ancestor for a sensor or target prim."""
    ancestor = prim
    while ancestor and ancestor.IsValid() and ancestor.GetPath().pathString != "/":
        if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            return ancestor
        ancestor = ancestor.GetParent()
    return None


def _newton_body_pattern(body_path: str) -> str:
    """Strip a concrete env prefix so Newton can register a cloned body pattern."""
    return re.sub(r"^/World/envs/env_\d+/", "", body_path)


def _xform_from_pose(pos, quat) -> wp.transform:
    """Create a Warp transform from Isaac Lab ``xyzw`` pose values."""
    return wp.transform(wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), wp.quat(*[float(v) for v in quat]))


def _identity_offsets(count: int, device: str) -> tuple[wp.array, wp.array]:
    """Create identity sensor offsets for site poses that already include the offset."""
    offset_pos_wp = wp.zeros(count, dtype=wp.vec3f, device=device)
    identity_quat = torch.zeros(count, 4, device=device)
    identity_quat[:, 3] = 1.0
    return offset_pos_wp, wp.from_torch(identity_quat.contiguous(), dtype=wp.quatf)


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
        self._tracked_site_labels_by_expr: dict[str, list[str]] = {}
        for target_cfg in getattr(self, "_raycast_targets_cfg", []):
            if target_cfg.track_mesh_transforms:
                self._tracked_site_labels_by_expr[target_cfg.prim_expr] = self._register_sites_for_expr(
                    target_cfg.prim_expr
                )

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
                labels.append(NewtonManager.cl_register_site(None, _xform_from_pose(pos, quat)))
            else:
                pos, quat = sim_utils.resolve_prim_pose(prim, body)
                labels.append(
                    NewtonManager.cl_register_site(
                        _newton_body_pattern(str(body.GetPath())), _xform_from_pose(pos, quat)
                    )
                )
        return list(dict.fromkeys(labels))

    def _initialize_pose_tracking(self: Any) -> None:
        """Resolve registered site labels and allocate sensor-owned pose buffers."""
        site_indices = self._resolve_site_indices(self._sensor_site_labels, self.cfg.prim_path, self._num_envs)
        # The base classes still use ``self._view.count`` in a few generic
        # places. Point it at the sensor instead of constructing an adapter.
        self._view = self
        self._view_count = len(site_indices)
        self._sensor_site_indices = wp.array(site_indices, dtype=wp.int32, device=self._device)
        self._newton_pose_w = wp.zeros(self._view_count, dtype=wp.transformf, device=self._device)
        self._newton_pos_w = wp.zeros(self._view_count, dtype=wp.vec3f, device=self._device)
        self._newton_quat_w = wp.zeros(self._view_count, dtype=wp.quatf, device=self._device)
        self._newton_pos_w_torch = wp.to_torch(self._newton_pos_w)
        self._newton_quat_w_torch = wp.to_torch(self._newton_quat_w)
        self._offset_pos_wp, self._offset_quat_wp = _identity_offsets(self._view_count, self._device)

    def _get_view_transforms_wp(self: Any) -> wp.array:
        """Return current Newton site transforms as ``wp.transformf``."""
        self._update_newton_site_transforms(
            self._sensor_site_indices, self._newton_pose_w, self._newton_pos_w, self._newton_quat_w
        )
        return self._newton_pose_w

    def get_world_poses(self: Any, indices=None):
        """Return world poses for camera helpers that still use pose tuples."""
        self._get_view_transforms_wp()
        if indices is None:
            return SimpleNamespace(torch=self._newton_pos_w_torch), SimpleNamespace(torch=self._newton_quat_w_torch)
        idx = wp.to_torch(indices).to(dtype=torch.long) if isinstance(indices, wp.array) else indices
        return SimpleNamespace(torch=self._newton_pos_w_torch[idx]), SimpleNamespace(
            torch=self._newton_quat_w_torch[idx]
        )

    def _create_tracked_target_view(self: Any, target_prim_path: str):
        """Resolve dynamic multi-mesh target sites to raw Newton site indices."""
        labels = self._tracked_site_labels_by_expr.get(target_prim_path)
        if labels is None:
            labels = self._register_sites_for_expr(target_prim_path)
            self._tracked_site_labels_by_expr[target_prim_path] = labels
        site_indices = self._resolve_site_indices(labels, target_prim_path, self._num_envs)
        return wp.array(site_indices, dtype=wp.int32, device=self._device)

    def _update_mesh_transforms(self: Any) -> None:
        """Refresh dynamic multi-mesh target slots directly from Newton sites."""
        if not hasattr(self, "_mesh_views"):
            return
        for site_indices, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                continue

            count = site_indices.shape[0]
            pos_buf = wp.zeros(count, dtype=wp.vec3f, device=self._device)
            quat_buf = wp.zeros(count, dtype=wp.quatf, device=self._device)
            pose_buf = wp.zeros(count, dtype=wp.transformf, device=self._device)
            self._update_newton_site_transforms(site_indices, pose_buf, pos_buf, quat_buf)
            pos_w = wp.to_torch(pos_buf)
            quat_w = wp.to_torch(quat_buf)
            slot_start, slot_end = self._slot_ranges_by_target_expr[target_cfg.prim_expr]
            slot_count = slot_end - slot_start
            if count == 1 and slot_count > 1:
                pos_w = pos_w.repeat(slot_count, 1)
                quat_w = quat_w.repeat(slot_count, 1)
            if pos_w.shape[0] != slot_count:
                raise RuntimeError(
                    f"Tracked target '{target_cfg.prim_expr}' produced {pos_w.shape[0]} poses, "
                    f"but the raycaster has {slot_count} mesh slots for that target."
                )
            self._slot_mesh_positions_w_torch[slot_start:slot_end] = pos_w
            self._slot_mesh_orientations_w_torch[slot_start:slot_end] = quat_w

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
                if label not in site_map:
                    raise ValueError(
                        f"RayCaster target '{prim_expr}' site label '{label}' was not found in "
                        "NewtonManager._cl_site_index_map."
                    )
                global_idx, per_world = site_map[label]
                if per_world is None:
                    if global_idx is None:
                        raise ValueError(
                            f"RayCaster target '{prim_expr}' site label '{label}' has no global Newton site index."
                        )
                    site_indices.append(global_idx)
                else:
                    if len(per_world) != num_envs:
                        raise ValueError(
                            f"RayCaster target '{prim_expr}' site label '{label}' has {len(per_world)} world entries, "
                            f"expected {num_envs}."
                        )
                    if len(per_world[env_idx]) == 0:
                        raise ValueError(
                            f"RayCaster target '{prim_expr}' site label '{label}' matched no bodies in env {env_idx}."
                        )
                    site_indices.extend(per_world[env_idx])
        return site_indices


class RayCaster(_NewtonRayCasterMixin, BaseRayCaster):
    """Newton ray-caster implementation."""
