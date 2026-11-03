# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configuration for the MicroDuck stand-up task."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckStandUpPPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck stand-up task."""

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_stand"
    # a third of the velocity budget.
    max_iterations = 15000
