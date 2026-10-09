# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small, measurable manager terms for an already-held teaspoon."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from .coffee_env import CoffeeEnv


class PalmPoseIKAction(DifferentialInverseKinematicsAction):
    """Control palm translation and axis-angle increments in the robot-base frame.

    Policy commands are scaled by the configuration. Absolute targets from the
    demo and teleoperation are converted to the same relative action interface.
    Spoon dynamics are configured independently by the environment.
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        valid = torch.isfinite(actions).all(dim=-1, keepdim=True)
        super().process_actions(torch.where(valid, actions, 0.0))

    def actions_from_pose(self, pose: torch.Tensor) -> torch.Tensor:
        """Encode an absolute robot-base palm pose, shape (N, 7), XYZW."""
        pos, quat = self._compute_frame_pose()
        translation, rotation = math_utils.compute_pose_error(pos, quat, pose[:, :3], pose[:, 3:])
        return torch.cat((translation, rotation), dim=-1) / self._scale

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        selected = slice(None) if env_ids is None else env_ids
        pos, quat = self._compute_frame_pose()
        self._processed_actions[selected] = 0.0
        # Updating every relative command here would advance unreset worlds'
        # targets a second time. Reset only the selected controller targets.
        self._ik_controller.ee_pos_des[selected] = pos[selected]
        self._ik_controller.ee_quat_des[selected] = quat[selected]

    def apply_actions(self) -> None:
        """Keep drive targets inside soft limits before native actuation."""
        super().apply_actions()
        # MJWarp limits are compliant: an unconstrained IK drive can push past
        # even the authored hard stop. Retain the articulation's soft margin.
        targets = self._asset.actuators.target_command.position.torch[:, self._joint_ids]
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        self._asset.actuators.target_command.set_position_index(
            value=targets.clamp(min=limits[..., 0], max=limits[..., 1]), joint_ids=self._joint_ids
        )


class HandPositionAction(JointPositionAction):
    """Use the native affine joint action for policy and absolute hand targets."""

    def actions_from_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode named hand joint positions [rad], shape (N, 22)."""
        return (positions - self._offset) / self._scale


def palm_pose(env: CoffeeEnv) -> torch.Tensor:
    """Measured palm pose in the robot-base frame, shape (N, 7), XYZW."""
    return env.palm_pose()


def spoon_pose(env: CoffeeEnv) -> torch.Tensor:
    """Measured spoon cavity pose in the environment frame, shape (N, 7)."""
    pose = env.scene["spoon"].data.root_link_pose_w.torch.clone()
    pose[:, :3] -= env.scene.env_origins
    return pose


def particle_fractions(env: CoffeeEnv) -> torch.Tensor:
    """Coffee fractions in the source, spoon, cup and below the table, (N, 4)."""
    return env.particle_fractions()


def target_vectors(env: CoffeeEnv) -> torch.Tensor:
    """Vectors from the spoon to the source and cup targets, shape (N, 6)."""
    return (env.tool_targets - spoon_pose(env)[:, None, :3]).flatten(start_dim=1)


def approach(env: CoffeeEnv, std: float) -> torch.Tensor:
    """Approach the source while empty and the cup while carrying coffee.

    The blend depends on measured spoon contents, without a scripted phase or
    a waypoint controller in the learning environment.
    """
    distances = target_vectors(env).reshape(env.num_envs, 2, 3).norm(dim=-1)
    loaded = (env.particle_fractions()[:, 1] / env.cfg.success_fraction).clamp(0.0, 1.0)
    proximity = 1.0 - torch.tanh(distances / std)
    return torch.lerp(proximity[:, 0], proximity[:, 1], loaded)


def delivery(env: CoffeeEnv) -> torch.Tensor:
    """Reward the fraction actually inside the cup, excluding the spoon."""
    return env.particle_fractions()[:, 2]


def spoon_load(env: CoffeeEnv) -> torch.Tensor:
    """Reward retained coffee after the spoon clears the source bowl."""
    height = spoon_pose(env)[:, 2]
    return env.particle_fractions()[:, 1] * (height > 0.055)


def spill(env: CoffeeEnv) -> torch.Tensor:
    """Coffee outside both containers below their support plane."""
    return env.particle_fractions()[:, 3]


def success(env: CoffeeEnv) -> torch.Tensor:
    """Require the configured delivered fraction to remain in the cup."""
    env.particle_fractions()
    return env.delivery_dwell >= env.cfg.success_dwell_s


def invalid_state(env: CoffeeEnv) -> torch.Tensor:
    """Stop episodes containing nonfinite joint or particle state."""
    robot = env.scene["robot"].data
    points = env.scene["coffee"].data.particle_pos_w.torch
    finite = torch.isfinite(robot.joint_pos.torch).all(dim=1) & torch.isfinite(points).all(dim=(1, 2))
    if env.cfg.water_enabled:
        finite &= torch.isfinite(env.scene["water"].data.particle_pos_w.torch).all(dim=(1, 2))
    return ~finite


def reset_task(env: CoffeeEnv, env_ids: torch.Tensor | slice) -> None:
    """Restore robot/tool poses and solver history for selected environments."""
    env.reset_task(env_ids)
