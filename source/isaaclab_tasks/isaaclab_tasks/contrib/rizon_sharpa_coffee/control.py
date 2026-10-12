# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Absolute palm IK and scene reset terms reusable by a learning environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

if TYPE_CHECKING:
    from .demo_env import CoffeeDemoEnv


class PalmPoseIKAction(DifferentialInverseKinematicsAction):
    """Use absolute palm poses while keeping IK drives within soft joint limits."""

    def process_actions(self, actions: torch.Tensor) -> None:
        if not torch.isfinite(actions).all():
            raise ValueError("Palm IK targets must be finite.")
        super().process_actions(actions)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        selected = slice(None) if env_ids is None else env_ids
        pos, quat = self._compute_frame_pose()
        self._processed_actions[selected] = torch.cat((pos[selected], quat[selected]), dim=-1)
        # Keep targets in unreset worlds unchanged.
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


def reset_scene(env: CoffeeDemoEnv, env_ids: torch.Tensor | slice) -> None:
    """Restore robot/tool poses and solver history for selected environments."""
    env.reset_scene(env_ids)
