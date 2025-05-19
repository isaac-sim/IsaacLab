# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a single drive"""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from rai.eval_sim.ros_manager import (
    JointCommandSubscriberCfg,
    JointStateObsPublisherCfg,
    RosManagerCfg,
)

ENTITY = SceneEntityCfg("robot")


@configclass
class SingleDriveRosManagerCfg(RosManagerCfg):
    # subscribers
    joint_position = JointCommandSubscriberCfg(topic="/joint_command")

    # publishers
    joint_state = JointStateObsPublisherCfg(
        topic="/joint_state",
        asset_cfg=ENTITY,
        obs_group="policy",
        position_obs="joint_pos_rel",
        velocity_obs="joint_vel_rel",
        effort_obs=None,
    )
