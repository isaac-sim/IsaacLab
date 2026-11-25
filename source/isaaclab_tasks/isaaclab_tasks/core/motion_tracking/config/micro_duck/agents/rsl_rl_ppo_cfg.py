# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reuse the G1 motion-tracking PPO configuration for MicroDuck."""

from isaaclab.utils import configclass

from ...g1.agents.rsl_rl_ppo_cfg import G1MotionTrackingPPORunnerCfg


@configclass
class MicroDuckMotionTrackingPPORunnerCfg(G1MotionTrackingPPORunnerCfg):
    experiment_name = "microduck_motion_tracking"
