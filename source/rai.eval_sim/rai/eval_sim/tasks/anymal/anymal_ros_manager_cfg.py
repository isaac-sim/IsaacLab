# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from rai.eval_sim.ros_manager import (
    JointPositionCommandSubscriberCfg,
    JointStateObsPublisherCfg,
    LinkPoseObsPublisherCfg,
    RosManagerCfg,
    TwistObsPublisherCfg,
)

ENTITY = SceneEntityCfg("robot")


@configclass
class AnymalDRosManagerCfg(RosManagerCfg):
    # subscribers
    joint_position = JointPositionCommandSubscriberCfg(topic="/joint_command")

    # publishers
    joint_state = JointStateObsPublisherCfg(
        topic="/joint_state",
        asset_cfg=ENTITY,
        obs_group="policy",
        position_obs="joint_pos",
        velocity_obs="joint_vel",
        effort_obs="joint_effort",
    )
    base_pose = LinkPoseObsPublisherCfg(topic="/base_pose", obs_group="policy", link_pose_obs="base_link_pose")
    base_twist = TwistObsPublisherCfg(
        topic="base_twist",
        obs_group="policy",
        frame_id="world",
        lin_vel_obs="base_lin_vel",
        ang_vel_obs="base_ang_vel",
    )
