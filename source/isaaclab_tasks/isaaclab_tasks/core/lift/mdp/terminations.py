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
    from isaaclab.assets import Articulation, CableObject, DeformableObject, RigidObject
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


def ee_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Return whether the end-effector is below the minimum environment-frame height [m]."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_z = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, 2] - env.scene.env_origins[:, 2]
    return ee_z < minimum_height


def deformable_outside_bounds(
    env: ManagerBasedRLEnv,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("deformable"),
) -> torch.Tensor:
    """Terminate if any deformable nodal point leaves the allowed workspace box.

    Covers both leaving the table footprint (x, y) and being dropped off it (z).

    Args:
        env: The environment instance.
        x_bounds: Allowed x-position range in the environment frame [m].
        y_bounds: Allowed y-position range in the environment frame [m].
        z_bounds: Allowed z-position range in the environment frame [m].
        asset_cfg: The deformable object entity.

    Returns:
        Boolean tensor with shape ``(num_envs,)``.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    nodal_pos = asset.data.nodal_pos_w.torch - env.scene.env_origins.unsqueeze(1)
    lower = torch.tensor([x_bounds[0], y_bounds[0], z_bounds[0]], device=nodal_pos.device)
    upper = torch.tensor([x_bounds[1], y_bounds[1], z_bounds[1]], device=nodal_pos.device)
    return ((nodal_pos < lower) | (nodal_pos > upper)).flatten(1).any(dim=1)


def cable_outside_bounds(
    env: ManagerBasedRLEnv,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cable"),
) -> torch.Tensor:
    """Terminate if any cable segment leaves the allowed workspace box."""
    asset: CableObject = env.scene[asset_cfg.name]
    segment_pos = asset.data.segment_pose_w.torch[..., :3] - env.scene.env_origins.unsqueeze(1)
    lower = torch.tensor([x_bounds[0], y_bounds[0], z_bounds[0]], device=segment_pos.device)
    upper = torch.tensor([x_bounds[1], y_bounds[1], z_bounds[1]], device=segment_pos.device)
    return ((segment_pos < lower) | (segment_pos > upper)).flatten(1).any(dim=1)


def joint_vel_out_of_sim_limit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when joint velocities exceed actuator simulator limits [m/s or rad/s, depending on joint type]."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    limits = torch.full_like(asset.data.joint_vel.torch, torch.inf)
    for actuator in asset.actuators.values():
        limits[:, actuator.joint_indices] = actuator.velocity_limit_sim
    return torch.any(torch.abs(asset.data.joint_vel.torch[:, joint_ids]) > limits[:, joint_ids], dim=1)
