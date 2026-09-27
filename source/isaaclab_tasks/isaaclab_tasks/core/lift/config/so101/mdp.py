# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State observations and lift shaping for the SO-101 cube task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the object position in the robot root frame [m], shape [N, 3]."""
    robot = env.scene["robot"]
    return quat_apply_inverse(
        robot.data.root_quat_w.torch,
        env.scene["object"].data.root_pos_w.torch - robot.data.root_pos_w.torch,
    )


def gripper_to_object_b(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return grasp-to-object displacement in the robot root frame [m], shape [N, 3]."""
    return quat_apply_inverse(
        env.scene["robot"].data.root_quat_w.torch,
        env.scene["object"].data.root_pos_w.torch - env.scene["grasp_frame"].data.target_pos_w.torch[:, 0],
    )


def gripper_orientation_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the gripper's world quaternion in xyzw order, shape [N, 4]."""
    return env.scene["grasp_frame"].data.target_quat_w.torch[:, 0]


def reaching_object(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Reward grasp-frame proximity with a tanh kernel of width ``std`` [m]."""
    distance = torch.linalg.vector_norm(gripper_to_object_b(env), dim=-1)
    return 1.0 - torch.tanh(distance / std)


class LiftReward(ManagerTermBase):
    """Reward lifting near the gripper and independently log sustained success.

    Success requires 5 cm of lift, grasp-frame proximity within 5.5 cm, and object speed
    below 0.2 m/s for 0.5 s continuously. The counter only measures success; it does not
    change the reward or the reset distribution.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._hold_steps = math.ceil(0.5 / env.step_dt)
        self._hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._max_lift = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self, env: ManagerBasedRLEnv, resting_height: float, lift_height: float, speed_std: float
    ) -> torch.Tensor:
        """Reward a raised, stationary cube.

        Args:
            env: The environment.
            resting_height: Resting object center height [m].
            lift_height: Height above rest at which the height reward saturates [m].
            speed_std: Width of the Gaussian penalty on object speed [m/s].
        """
        obj = env.scene["object"]
        height = obj.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2] - resting_height
        near_gripper = torch.linalg.vector_norm(gripper_to_object_b(env), dim=-1) < 0.055
        slow = torch.linalg.vector_norm(obj.data.root_lin_vel_w.torch, dim=-1) < 0.2
        held = (height > 0.05) & near_gripper & slow
        self._hold = torch.where(held, self._hold + 1, 0)
        self._succeeded |= self._hold >= self._hold_steps
        self._max_lift = torch.maximum(self._max_lift, height)
        stillness = torch.exp(-torch.sum((obj.data.root_lin_vel_w.torch / speed_std).square(), dim=-1))
        return (height / lift_height).clamp(0.0, 1.0) * near_gripper * stillness

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Log episode success and peak lift [m], then clear the episode counters."""
        if env_ids is None:
            env_ids = slice(None)
        self._env.extras.setdefault("log", {}).update(
            {
                "Metrics/lift_success": self._succeeded[env_ids].float().mean(),
                "Metrics/max_lift": self._max_lift[env_ids].mean(),
                "Metrics/final_hold": (self._hold[env_ids] >= 2 * self._hold_steps).float().mean(),
            }
        )
        self._hold[env_ids] = 0
        self._succeeded[env_ids] = False
        self._max_lift[env_ids] = 0.0
        return {}
