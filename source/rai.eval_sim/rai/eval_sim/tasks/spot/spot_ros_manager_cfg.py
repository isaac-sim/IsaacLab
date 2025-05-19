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
    TwistObsPublisherCfg,
    Vector3StampedObsPublisherCfg,
)

##
# Ros Cfg
##

ENTITY = SceneEntityCfg("robot")


@configclass
class SpotRosManagerCfg(RosManagerCfg):
    # subscribers
    joint_position = JointPositionCommandSubscriberCfg(topic="/joint_command")

    # state publishers
    base_twist = TwistObsPublisherCfg(
        topic="base_twist",
        obs_group="policy",
        frame_id="world",
        lin_vel_obs="base_lin_vel",
        ang_vel_obs="base_ang_vel",
    )
    projected_gravity = Vector3StampedObsPublisherCfg(
        topic="/projected_gravity", obs_group="policy", obs_term_name="projected_gravity"
    )
    joint_state = JointStateObsPublisherCfg(
        topic="/joint_state",
        asset_cfg=ENTITY,
        obs_group="policy",
        position_obs="joint_pos",
        velocity_obs="joint_vel",
        effort_obs=None,
    )
