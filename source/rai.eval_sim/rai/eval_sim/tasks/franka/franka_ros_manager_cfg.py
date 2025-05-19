# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from rai.eval_sim.ros_manager import (
    JointPositionCommandSubscriberCfg,
    JointStateObsPublisherCfg,
    RosManagerCfg,
)

ENTITY = SceneEntityCfg("robot")


@configclass
class FrankaRosManagerCfg(RosManagerCfg):
    # subscribers
    # action_name is None, because it's a combined joint position and gripper position command
    joint_position = JointPositionCommandSubscriberCfg(topic="/joint_command", action_name=None)

    # publishers
    joint_state = JointStateObsPublisherCfg(
        topic="/joint_state",
        asset_cfg=ENTITY,
        obs_group="policy",
        position_obs="joint_pos",
        velocity_obs="joint_vel",
        effort_obs=None,
    )
