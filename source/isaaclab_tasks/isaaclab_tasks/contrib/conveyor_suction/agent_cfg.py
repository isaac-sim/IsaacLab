# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reuse the conveyor's continuous-arm / binary-tool PPO distribution."""

from isaaclab.utils import configclass

from ..conveyor_franka.agents.rsl_rl_ppo_cfg import ConveyorFrankaPPORunnerCfg


@configclass
class SuctionSortingPPORunnerCfg(ConveyorFrankaPPORunnerCfg):
    experiment_name = "conveyor_suction_size_sorting"
    max_iterations = 6000
    save_interval = 100

    def __post_init__(self) -> None:
        self.actor.hidden_dims = [256, 256, 128]
        self.critic.hidden_dims = [256, 256, 128]
        self.algorithm.learning_rate = 0.0003
        self.algorithm.num_mini_batches = 8
