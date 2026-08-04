# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


class out_of_bound(ManagerTermBase):
    """Termination condition for when the object falls out of bound.

    The world-space bounds are cached and rebuilt per axis only when the corresponding
    ``in_bound_range`` entry changes. This keeps the hot path free of host-to-device
    transfers while still honoring runtime updates (e.g. from a curriculum term).
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self._object: RigidObject = env.scene[asset_cfg.name]

        # Pre-apply env_origins so we can compare directly against world-space positions.
        self._origins = env.scene.env_origins  # (N, 3)
        self._lower = self._origins.clone()  # (N, 3)
        self._upper = self._origins.clone()  # (N, 3)
        self._cached_axis: list[tuple[float, ...] | None] = [None, None, None]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        in_bound_range: dict[str, tuple[float, float]] = {},
    ) -> torch.Tensor:
        # rebuild only the axes whose bounds changed (curriculum typically only moves one)
        for i, key in enumerate(["x", "y", "z"]):
            bounds = tuple(in_bound_range.get(key, (0.0, 0.0)))
            if bounds != self._cached_axis[i]:
                lo, hi = bounds
                self._lower[:, i] = self._origins[:, i] + lo
                self._upper[:, i] = self._origins[:, i] + hi
                self._cached_axis[i] = bounds

        pos_w = self._object.data.root_pos_w.torch
        return ((pos_w < self._lower) | (pos_w > self._upper)).any(dim=1)


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminating environment when violation of velocity limits detects, this usually indicates unstable physics caused
    by very bad, or aggressive action"""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel.torch
    joint_vel_limits = robot.data.joint_vel_limits.torch
    return (joint_vel.abs() > (joint_vel_limits * 2)).any(dim=1)


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return whether the rigid object reached the commanded goal position."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3])
    return torch.linalg.norm(des_pos_w - object.data.root_pos_w.torch[:, :3], dim=1) < threshold


def deformable_com_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Return whether the deformable object's COM is below the minimum height [m]."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    return wp.to_torch(asset.data.root_pos_w)[:, 2] < minimum_height


def deformable_outside_table_bounds(
    env: ManagerBasedRLEnv,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Return whether any deformable node left the table footprint [m]."""
    asset: DeformableObject = env.scene[asset_cfg.name]
    nodal_pos = wp.to_torch(asset.data.nodal_pos_w) - env.scene.env_origins.unsqueeze(1)
    outside_x = (nodal_pos[..., 0] < x_bounds[0]) | (nodal_pos[..., 0] > x_bounds[1])
    outside_y = (nodal_pos[..., 1] < y_bounds[0]) | (nodal_pos[..., 1] > y_bounds[1])
    return torch.any(outside_x | outside_y, dim=1)


def ee_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return whether the end-effector is below the minimum environment-frame height [m]."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_z = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, 2] - env.scene.env_origins[:, 2]
    return ee_z < minimum_height
