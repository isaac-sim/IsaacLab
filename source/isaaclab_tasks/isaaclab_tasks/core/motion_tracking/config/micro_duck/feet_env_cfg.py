# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Foot-focused tracking with independent foot and head height-error limits."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ... import mdp
from ...motion_tracking_env_cfg import RewardsCfg
from .rewards import motion_feet_position_error_exp
from .tracking_env_cfg import MicroDuckMotionTrackingEnvCfg, MicroDuckTrackingTerminationsCfg


@configclass
class MicroDuckFeetRewardsCfg(RewardsCfg):
    """Keep whole-body tracking and add equally weighted left/right foot tracking."""

    motion_feet_pos = RewTerm(
        func=motion_feet_position_error_exp,
        weight=2.0,
        params={"command_name": "motion", "body_names": ["ankle_left", "ankle_right"], "std": 0.02},
    )


@configclass
class MicroDuckFeetTerminationsCfg(MicroDuckTrackingTerminationsCfg):
    """Control foot tracking tolerance separately from head tracking tolerance."""

    feet_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={"command_name": "motion", "body_names": ["ankle_left", "ankle_right"], "threshold": 0.03},
    )


@configclass
class MicroDuckFeetTrackingEnvCfg(MicroDuckMotionTrackingEnvCfg):
    """Foot-focused variant with unchanged observations, actions and reference loading."""

    rewards: MicroDuckFeetRewardsCfg = MicroDuckFeetRewardsCfg()
    terminations: MicroDuckFeetTerminationsCfg = MicroDuckFeetTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.ee_body_pos.params["body_names"] = ["jaw_soft"]
