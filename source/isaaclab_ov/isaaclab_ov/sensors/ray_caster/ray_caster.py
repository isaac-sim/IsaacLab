# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX ray-caster sensors -- mixin + concrete RayCaster class."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.scene_data import SceneDataFormat
from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster


class _OvPhysxRayCasterMixin:
    """Share native body poses for ray-caster frames; cache authored poses for static frames."""

    @property
    def count(self: Any) -> int:
        """Number of tracked sensor frames."""
        return self._view_count

    def _initialize_pose_tracking(self: Any) -> None:
        """Bind sensor bodies in environment order, independently of native binding order."""
        self._view = self
        self._static_view_transforms_wp = None
        prim, sensor_expr = sim_utils.resolve_matching_prims_from_source(self.cfg.prim_path)[0]
        body = sim_utils.get_first_matching_ancestor_prim(
            prim.GetPath(), predicate=lambda ancestor: ancestor.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        if body is None:
            prims = sim_utils.find_matching_prims(sensor_expr)
            self._static_view_transforms_wp = wp.array(
                [(*pos, *quat) for pos, quat in map(sim_utils.resolve_prim_pose, prims)],
                dtype=wp.transformf,
                device=self._device,
            )
            self._view_count = len(prims)
            offset_pos, offset_quat = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        else:
            relative = prim.GetPath().MakeRelativePath(body.GetPath()).pathString
            body_expr = sensor_expr if prim == body else sensor_expr[: -(len(relative) + 1)]
            plan = self._sim.get_clone_plan()
            matched = cloner.path.match(body_expr, plan.env_template) if plan is not None else None
            if matched is not None:
                native_paths = set(self._scene_data_provider.backend.transform_paths)
                body_paths = [
                    path
                    for i in range(self._num_envs)
                    if (path := plan.env_template.format(i) + matched.suffix) in native_paths
                ]
            else:
                body_paths = sim_utils.find_matching_prim_paths(body_expr)
            self._view_count = len(body_paths)
            self._sensor_mapping = self._scene_data_provider.create_mapping(body_paths)
            self._sensor_transforms = SceneDataFormat.Transform()
            offset_pos, offset_quat = sim_utils.resolve_prim_pose(prim, body)
        self._offset_pos_wp = wp.full(self._view_count, wp.vec3f(*offset_pos), device=self._device)
        self._offset_quat_wp = wp.full(self._view_count, wp.quatf(*offset_quat), device=self._device)

    def _get_view_transforms_wp(self: Any) -> wp.array:
        """Return sensor-body poses from the shared publication or the static snapshot."""
        if self._static_view_transforms_wp is not None:
            return self._static_view_transforms_wp
        self._scene_data_provider.get_transforms(self._sensor_transforms, self._sensor_mapping, count=self._view_count)
        return self._sensor_transforms.transforms

    def get_world_poses(self: Any, indices=None):
        """Return world poses as ``(positions, orientations)`` pose tuples.

        Camera-derived base classes inheriting this mixin call this method
        and read ``.torch`` on the returned objects. We mirror PhysX's
        :class:`SimpleNamespace` shape so the contract is identical.
        """
        transforms = self._get_view_transforms_wp()
        transforms_t = wp.to_torch(transforms).reshape(-1, 7)
        if indices is not None:
            idx = wp.to_torch(indices).to(dtype=torch.long) if isinstance(indices, wp.array) else indices
            transforms_t = transforms_t[idx]
        return SimpleNamespace(torch=transforms_t[:, 0:3]), SimpleNamespace(torch=transforms_t[:, 3:7])


class RayCaster(_OvPhysxRayCasterMixin, BaseRayCaster):
    """OVPhysX RayCaster implementation."""
