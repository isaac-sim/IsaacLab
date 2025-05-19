# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from rai.eval_sim.ros_manager import RosManagerCfg
from rai.eval_sim.ros_manager import publishers_cfg as pub_cfg
from rai.eval_sim.ros_manager import subscribers_cfg as sub_cfg


@configclass
class AnymalDPlusPDGainsRMcfg(RosManagerCfg):
    pd_gains = sub_cfg.PDGainsSubscriberCfg(topic="/joint_command_pd_gains")
    joint_position = sub_cfg.JointPositionCommandSubscriberCfg(topic="/joint_command")

    # publishers
    joint_state = pub_cfg.JointStateObsPublisherCfg(
        topic="/joint_state",
        asset_cfg=SceneEntityCfg("robot"),
        obs_group="policy",
        position_obs="joint_pos",
        velocity_obs="joint_vel",
        effort_obs="joint_effort",
    )


@configclass
class AnymalDBaseLineRosCfg(RosManagerCfg):
    # subscribers
    joint_position = sub_cfg.JointPositionCommandSubscriberCfg(topic="/joint_command")

    # publishers
    joint_state = pub_cfg.JointStateObsPublisherCfg(
        topic="/joint_state",
        asset_cfg=SceneEntityCfg("robot"),
        obs_group="policy",
        position_obs="joint_pos",
        velocity_obs="joint_vel",
        effort_obs="joint_effort",
    )


@configclass
class AnymalDPlusLinkPoseObsCfg(AnymalDBaseLineRosCfg):
    base_link_pose = pub_cfg.LinkPoseObsPublisherCfg(
        topic="/base_pose",
        obs_group="policy",
        link_pose_obs="base_link_pose",
    )


@configclass
class AnymalDPlusTwistObsCfg(AnymalDBaseLineRosCfg):
    base_twist = pub_cfg.TwistObsPublisherCfg(
        topic="base_twist",
        obs_group="policy",
        frame_id="world",
        lin_vel_obs="base_lin_vel",
        ang_vel_obs="base_ang_vel",
    )


@configclass
class AnymalDPlusProjGravObsRMCfg(AnymalDBaseLineRosCfg):
    projected_gravity = pub_cfg.Vector3StampedObsPublisherCfg(
        topic="/projected_gravity",
        obs_group="policy",
        obs_term_name="projected_gravity",
    )


@configclass
class AnymalDPlusImuRMCfg(AnymalDBaseLineRosCfg):
    imu = pub_cfg.ImuObsPublisherCfg(
        topic="/imu",
        obs_group="policy",
        imu_quat_obs="imu_quat",
        imu_ang_vel_obs="imu_ang_vel",
        imu_lin_acc_obs="imu_lin_acc",
    )


@configclass
class AnymalDPlusContactRMCfg(AnymalDBaseLineRosCfg):
    contact_forces = pub_cfg.ContactForcePublisherCfg(
        asset_cfg=SceneEntityCfg("contact_forces", body_names=["RF_FOOT", "LF_FOOT", "RH_FOOT", "LH_FOOT"]),
        topic="/contacts_forces",
    )


@configclass
class AnymalDPlusHeightScanRMCfg(AnymalDBaseLineRosCfg):
    height_scan = pub_cfg.FlattenedObsPublisherCfg(
        topic="/height_scan",
        obs_group="policy",
        obs_term_name="height_scan",
    )


@configclass
class AnymalDPlusGridMapRMCfg(AnymalDBaseLineRosCfg):
    grid_map = pub_cfg.HeightMapPublisherCfg(
        asset_cfg=SceneEntityCfg(name="grid_map"),
        topic="/elevation_map",
        layer="elevation",
    )


@configclass
class AnymalDPlusWrenchRMCfg(AnymalDBaseLineRosCfg):
    wrench = pub_cfg.WrenchStampedObsPublisherCfg(
        topic="/wrench/root", obs_group="policy", obs_term_name="RF_foot_reaction"
    )


@configclass
class AnymalDPlusJointReactionWrenchObsPublisherRMCfg(AnymalDBaseLineRosCfg):
    joint_reactions = pub_cfg.JointReactionWrenchObsPublisherCfg(
        topic="/joint_reactions", obs_group="policy", obs_term_name="joint_reactions"
    )
