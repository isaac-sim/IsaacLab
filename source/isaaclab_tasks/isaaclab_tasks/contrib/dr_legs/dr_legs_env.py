# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based environment behavior shared by the DR Legs tasks."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn


class DrLegsRLEnv(ManagerBasedRLEnv):
    """DR Legs environment that discards invalid solver rewards."""

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Step the environment and keep rewards finite after solver failures."""
        observations, rewards, terminated, time_outs, extras = super().step(action)
        rewards[~torch.isfinite(rewards)] = 0.0
        return observations, rewards, terminated, time_outs, extras
