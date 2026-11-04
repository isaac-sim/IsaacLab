# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in head-position tracking on top of the foot-focused MicroDuck task."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .feet_env_cfg import MicroDuckFeetRewardsCfg, MicroDuckFeetTrackingEnvCfg
from .rewards import motion_head_position_error_exp


@configclass
class MicroDuckFeetHeadRewardsCfg(MicroDuckFeetRewardsCfg):
    """Add head-position tracking without changing the existing foot reward."""

    motion_head_pos = RewTerm(
        func=motion_head_position_error_exp,
        weight=4.0,
        params={"command_name": "motion", "body_name": "jaw_soft", "std": 0.02},
    )


@configclass
class MicroDuckFeetHeadTrackingEnvCfg(MicroDuckFeetTrackingEnvCfg):
    """Preserve foot tracking and failure limits while adding a head reward."""

    rewards: MicroDuckFeetHeadRewardsCfg = MicroDuckFeetHeadRewardsCfg()
