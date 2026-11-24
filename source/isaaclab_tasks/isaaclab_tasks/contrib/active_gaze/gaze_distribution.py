# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gaussian action head whose gaze actions have a fixed standard deviation."""

from __future__ import annotations

import math

import torch
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal

GAZE_ACTIONS = 2
"""The gaze actions (tilt, pan) are the policy's last two actions."""


class GazeGaussianDistribution(GaussianDistribution):
    """Gaussian action head with a learned standard deviation for every action but the gaze, the last two, whose
    standard deviation is fixed at ``gaze_std``.

    The gaze step clamps each gaze action to [-1, 1], so beyond that range a wider gaze distribution changes neither
    the reward nor PPO's surrogate, and the entropy bonus grows a learned standard deviation without bound; the sampled
    gaze then random-walks however well its mean follows the teacher.

    Args:
        output_dim: Number of actions, the two gaze actions included.
        init_std: Initial standard deviation of the other actions.
        std_range: (min, max) clamp of the other actions' standard deviation.
        std_type: Parameterization of the other actions' standard deviation: ``"scalar"`` or ``"log"``.
        learn_std: Whether the other actions' standard deviation is learned.
        gaze_std: Fixed standard deviation of the gaze actions.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        std_type: str = "scalar",
        learn_std: bool = True,
        gaze_std: float = 0.1,
    ) -> None:
        if output_dim <= GAZE_ACTIONS:
            raise ValueError(f"The gaze head needs actions besides the {GAZE_ACTIONS} gaze actions, got {output_dim}.")
        if not (math.isfinite(gaze_std) and gaze_std > 0):
            raise ValueError(f"gaze_std must be positive and finite, got {gaze_std!r}.")
        # The learned parameter covers the other actions only.
        super().__init__(output_dim - GAZE_ACTIONS, init_std, std_range, std_type, learn_std)
        self.output_dim = output_dim
        self.register_buffer("gaze_std", torch.full((GAZE_ACTIONS,), float(gaze_std)))

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the distribution from the MLP output: learned noise for the other actions, fixed for the gaze."""
        if self.std_type == "scalar":
            std = self.std_param.clamp(self.std_range[0], self.std_range[1])
        else:
            std = self.log_std_param.clamp(self.log_std_range[0], self.log_std_range[1]).exp()
        self._distribution = Normal(mlp_output, torch.cat((std, self.gaze_std)))
