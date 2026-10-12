# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based environment behavior shared by the DR Legs tasks."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn


class DrLegsRLEnv(ManagerBasedRLEnv):
    """DR Legs environment that discards invalid rewards from terminated solver states."""

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Step the environment and keep rewards finite for failed copies being reset."""
        observations, rewards, terminated, time_outs, extras = super().step(action)
        invalid_terminal_rewards = terminated & ~torch.isfinite(rewards)
        rewards[invalid_terminal_rewards] = 0.0
        return observations, rewards, terminated, time_outs, extras
