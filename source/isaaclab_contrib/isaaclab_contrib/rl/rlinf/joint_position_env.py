# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RLinf evaluation of policies that predict absolute joint positions."""

from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction


class AbsoluteJointPositionEnv(gym.Wrapper):
    """Convert policy targets to the joint action term's coordinates and refresh reset camera observations."""

    def __init__(self, env: ManagerBasedRLEnv, *, prefix_pad: int, warmup_steps: int):
        super().__init__(env)
        self.device = env.device
        self._joint = env.action_manager.get_term("joint_pos")
        if (
            not isinstance(self._joint, JointPositionAction)
            or env.action_manager.total_action_dim != self._joint.action_dim
        ):
            raise ValueError("Absolute joint position evaluation requires a single joint_pos position action term.")
        if not 0 <= prefix_pad <= self._joint.action_dim or warmup_steps < 0:
            raise ValueError("Invalid prefix length or warmup step count for absolute joint position evaluation.")
        self._robot = env.scene[self._joint.cfg.asset_name]
        self._prefix_pad = prefix_pad
        if warmup_steps:
            env.reset()
            zeros = torch.zeros((env.num_envs, self._joint.action_dim), device=env.device)
            for _ in range(warmup_steps):
                env.step(zeros)
            env.reset()

    def step(self, action: torch.Tensor):
        action = action.to(self.device)
        if self._prefix_pad:
            action = action.clone()
            defaults = self._robot.data.default_joint_pos.torch[:, self._joint._joint_ids]
            action[:, : self._prefix_pad] = defaults[:, : self._prefix_pad]
        return self.env.step((action - self._joint._offset) / self._joint._scale)

    def reset(self, *, seed=None, options=None, env_ids=None):
        obs, info = self.env.reset(seed=seed, options=options, env_ids=env_ids)
        # A full reset needs one physics step to refresh the tiled cameras. Partial resets must preserve peer episodes.
        count = self.env.unwrapped.num_envs
        if isinstance(env_ids, slice):
            env_ids = range(*env_ids.indices(count))
        if env_ids is None or len(env_ids) == count:
            hold = self._robot.data.joint_pos.torch[:, self._joint._joint_ids]
            obs, _, _, _, info = self.step(hold)
        return obs, info
