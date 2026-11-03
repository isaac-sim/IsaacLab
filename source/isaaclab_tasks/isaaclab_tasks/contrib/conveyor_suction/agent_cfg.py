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
)

_GRIPPER_LOGIT_BOUND = math.log(99.0)


class _SuctionOutput(nn.Module):
    """Bound Cartesian means and retain at least 1% exploration for either tool command."""

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        logit = _GRIPPER_LOGIT_BOUND * (output[..., -1:] / _GRIPPER_LOGIT_BOUND).tanh()
        return torch.cat((output[..., :-1].tanh(), logit), dim=-1)


class SuctionGaussianBernoulliDistribution(ConveyorGaussianBernoulliDistribution):
    """Explore Cartesian motion while initially preserving native suction seals."""

    def init_mlp_weights(self, mlp: nn.Sequential) -> None:
        """Start near rest with a finite 98% probability of keeping vacuum on."""
        nn.init.orthogonal_(mlp[-1].weight, gain=0.01)
        nn.init.zeros_(mlp[-1].bias)
        with torch.no_grad():
            mlp[-1].bias[-1] = _GRIPPER_LOGIT_BOUND * math.atanh(math.log(0.02 / 0.98) / _GRIPPER_LOGIT_BOUND)

    def update(self, mlp_output: torch.Tensor) -> None:
        super().update(_SuctionOutput()(mlp_output))

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return super().deterministic_output(_SuctionOutput()(mlp_output))

    def as_deterministic_output_module(self) -> nn.Module:
        return nn.Sequential(_SuctionOutput(), super().as_deterministic_output_module())


@configclass
class SuctionSortingPPORunnerCfg(ConveyorFrankaPPORunnerCfg):
    experiment_name = "conveyor_suction_size_sorting"
    max_iterations = 6000
    save_interval = 100

    def __post_init__(self) -> None:
        self.actor.distribution_cfg.class_name = f"{__name__}:SuctionGaussianBernoulliDistribution"
        self.actor.hidden_dims = [256, 256, 128]
        self.critic.hidden_dims = [256, 256, 128]
        self.algorithm.learning_rate = 0.0003
        self.algorithm.num_mini_batches = 8
