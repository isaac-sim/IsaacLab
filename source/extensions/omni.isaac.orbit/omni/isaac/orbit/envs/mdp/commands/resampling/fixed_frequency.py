# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Sequence

from omni.isaac.orbit.managers import ResamplingTerm

from .resampling_cfg import FixedFrequencyCfg


class FixedFrequency(ResamplingTerm):
    """Fixed frequency resampling term.

    The fixed frequency resampling term is used to resample commands at a fixed frequency.
    When an environment is resampled, the time left is sampled from the range specified
    in the FixedFrequencyCfg.
    """

    def __init__(self, cfg: FixedFrequencyCfg, env):
        super().__init__(cfg, env)

        # -- time left before resampling
        self.time_left = torch.zeros(self.num_envs, device=self.device)

    def compute(self, dt: float):
        """Compute the environment ids to be resampled.

        Args:
            dt: The time step.
        """
        # reduce the time left before resampling
        self.time_left -= dt
        # resample expired timers.
        resample_env_ids = (self.time_left <= 0.0).nonzero().flatten()
        return resample_env_ids

    def reset(self, env_ids: Sequence[int]):
        """Reset the resampling term.

        Resamples the time left from the cfg range.

        Args:
            env_ids: The environment ids to be reset.
        """
        self.time_left[env_ids] = self.time_left[env_ids].uniform_(*self.cfg.resampling_time_range)
