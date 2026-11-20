# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with bounded Cartesian means and a vacuum-hold exploration prior."""

import math

import torch
import torch.nn as nn

from isaaclab.utils import configclass

from ..conveyor_franka.agents.rsl_rl_ppo_cfg import (
    ConveyorFrankaPPORunnerCfg,
    ConveyorGaussianBernoulliDistribution,
    ConveyorGaussianBernoulliDistributionCfg,
)


class _SuctionOutput(nn.Module):
    """Bound Cartesian means and retain exploration for either tool command."""

    def __init__(self, logit_bound: float) -> None:
        super().__init__()
        self.logit_bound = logit_bound

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        logit = self.logit_bound * (output[..., -1:] / self.logit_bound).tanh()
        return torch.cat((output[..., :-1].tanh(), logit), dim=-1)


class SuctionGaussianBernoulliDistribution(ConveyorGaussianBernoulliDistribution):
    """Explore Cartesian motion while initially preserving native suction seals."""

    def __init__(self, output_dim: int, tool_probability_floor: float = 0.001, **kwargs) -> None:
        if not 0.0 < tool_probability_floor < 0.02:
            raise ValueError("tool_probability_floor must be between 0 and 0.02 to retain 98% initial vacuum hold.")
        super().__init__(output_dim, **kwargs)
        self._output = _SuctionOutput(math.log((1.0 - tool_probability_floor) / tool_probability_floor))

    def init_mlp_weights(self, mlp: nn.Sequential) -> None:
        """Start near rest with a finite 98% probability of keeping vacuum on."""
        nn.init.orthogonal_(mlp[-1].weight, gain=0.01)
        nn.init.zeros_(mlp[-1].bias)
        with torch.no_grad():
            bound = self._output.logit_bound
            mlp[-1].bias[-1] = bound * math.atanh(math.log(0.02 / 0.98) / bound)

    def update(self, mlp_output: torch.Tensor) -> None:
        super().update(self._output(mlp_output))

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return super().deterministic_output(self._output(mlp_output))

    def as_deterministic_output_module(self) -> nn.Module:
        return nn.Sequential(self._output, super().as_deterministic_output_module())


@configclass
class SuctionGaussianBernoulliDistributionCfg(ConveyorGaussianBernoulliDistributionCfg):
    """Keep both vacuum commands reachable during stochastic training."""

    class_name: str = f"{__name__}:SuctionGaussianBernoulliDistribution"
    tool_probability_floor: float = 0.001
    """Minimum probability for either tool command, strictly between zero and 0.02."""


@configclass
class SuctionSortingPPORunnerCfg(ConveyorFrankaPPORunnerCfg):
    experiment_name = "conveyor_suction_size_sorting"
    num_steps_per_env = 128
    max_iterations = 6000
    save_interval = 100

    def __post_init__(self) -> None:
        self.actor.distribution_cfg = SuctionGaussianBernoulliDistributionCfg(init_std=0.45)
        self.actor.hidden_dims = [256, 256, 128]
        self.critic.hidden_dims = [256, 256, 128]
        # Geometry already uses fixed physical scales. Constant training ray cells
        # must not acquire near-zero variance that amplifies new fixture geometry.
        self.actor.obs_normalization = False
        self.critic.obs_normalization = False
        self.algorithm.num_mini_batches = 8
        self.algorithm.lam = 0.99
