# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from omni.isaac.orbit.managers import ResamplingTerm

if TYPE_CHECKING:
    from .resampling_cfg import RandomChanceCfg


class RandomChance(ResamplingTerm):
    """Random chance resampling term.

    Commands are resampled with a fixed probability at every time step.
    """

    def __init__(self, cfg: FixedFrequencyCfg, env):
        super().__init__(cfg, env)


    def __str__(self) -> str:
        msg = f"\t\tResampling probability: {self.cfg.resampling_probability}"
        return msg

    def compute(self, dt: float):
        """Compute the environment ids to be resampled.

        Args:
            dt: The time step.
        """
        # Note: uniform_(0, 1) is inclusive on 0 and exclusive on 1. So we need to use < instead of <=.
        resample_prob_buf = torch.empty(self.num_envs, device=self.device).uniform_(0, 1) < self.cfg.resampling_probability
        resample_env_ids = resample_prob_buf.nonzero(as_tuple=False).flatten()
        return resample_env_ids
