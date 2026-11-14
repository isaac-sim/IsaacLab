# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for the OneRobotics A1 bimanual reach task."""

from isaaclab.utils.configclass import configclass

from ...unimanual.agents.rsl_rl_ppo_cfg import OneRoboticsA1ReachPPORunnerCfg


@configclass
class OneRoboticsA1BimanualReachPPORunnerCfg(OneRoboticsA1ReachPPORunnerCfg):
    """Reuse the validated unimanual PPO settings under a distinct experiment name."""

    experiment_name = "onerobotics_a1_bimanual_reach"
