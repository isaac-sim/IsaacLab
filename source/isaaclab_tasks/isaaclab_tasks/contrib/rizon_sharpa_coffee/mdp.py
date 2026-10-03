# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small, measurable manager terms for an already-held teaspoon."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .coffee_env import CoffeeEnv


class PalmPoseIKAction(DifferentialInverseKinematicsAction):
    """Control a bounded absolute palm pose for either tool-grasp mode.

    Absolute commands use robot-base positions [m] and XYZW quaternions. Invalid
    tracking samples retain the preceding target. Spoon dynamics are configured
    independently by the environment.
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        valid = torch.isfinite(actions).all(dim=-1) & (actions[:, 3:7].norm(dim=-1) > 1.0e-6)
        # IsaacTeleop's missing-pose sentinel must never send the hand to the base.
        valid &= actions[:, :3].norm(dim=-1) > 1.0e-6
        safe = torch.where(valid[:, None], actions, self._processed_actions).clone()
        safe[:, 3:7] /= safe[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        super().process_actions(safe)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        selected = slice(None) if env_ids is None else env_ids
        pos, quat = self._compute_frame_pose()
        self._processed_actions[selected] = torch.cat((pos, quat), dim=-1)[selected]
        self._ik_controller.set_command(self._processed_actions, pos, quat)

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


@configclass
class PalmPoseIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Absolute palm inverse kinematics with bounded joint targets."""

    class_type: type = PalmPoseIKAction


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


def delivery(env: CoffeeEnv) -> torch.Tensor:
    """Reward the fraction actually inside the glass, excluding the spoon."""
    return env.particle_fractions()[:, 2]


def spoon_load(env: CoffeeEnv) -> torch.Tensor:
    """Reward retained coffee after the spoon clears the source bowl."""
    height = spoon_pose(env)[:, 2]
    return env.particle_fractions()[:, 1] * (height > 0.055)


def spill(env: CoffeeEnv) -> torch.Tensor:
    """Coffee outside both containers below their support plane."""
    return env.particle_fractions()[:, 3]


def success(env: CoffeeEnv) -> torch.Tensor:
    """Require the configured delivered fraction to remain in the glass."""
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
