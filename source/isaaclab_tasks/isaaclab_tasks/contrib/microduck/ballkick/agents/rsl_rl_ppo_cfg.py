# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configuration for the MicroDuck ball-kick task."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.ballkick.ballkick_env_cfg import MICRODUCK_KICK_FOOT
from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckBallKickPPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck ball-kick task."""

    experiment_name = f"microduck_ball_kick_{MICRODUCK_KICK_FOOT}"
    # A fifth of the velocity budget.
    max_iterations = 10000
