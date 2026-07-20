# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import (
    compute_plug_pose,
    compute_socket_root,
)

from .joint_pos_env_cfg import Rizon4sGravDisplayportInsertionEnvCfg

# Deployment socket/plug poses for the physical DisplayPort insertion station.
_DEPLOY_GEOMETRY_POS = (0.476, 0.127, 0.07)
_DEPLOY_SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)
_DEPLOY_PLUG_CLEARANCE_Z = 0.068

_DEPLOY_SOCKET_ROOT = compute_socket_root(_DEPLOY_GEOMETRY_POS, _DEPLOY_SOCKET_ROT)
_DEPLOY_PLUG_ROOT, _DEPLOY_PLUG_ROT = compute_plug_pose(
    _DEPLOY_GEOMETRY_POS,
    _DEPLOY_SOCKET_ROT,
    z_clearance=_DEPLOY_PLUG_CLEARANCE_Z,
)


@configclass
class Rizon4sGravDisplayportInsertionROSInferenceEnvCfg(Rizon4sGravDisplayportInsertionEnvCfg):
    """Configuration for ROS inference with Flexiv Rizon 4s and Grav gripper.

    This configuration:
    - Exposes variables needed for ROS inference
    - Overrides robot and plug/socket initial poses for fixed/deterministic setup
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Variables used by Isaac Manipulator for on robot inference
        # These parameters allow the ROS inference node to validate environment configuration,
        # perform checks during inference, and correctly interpret observations and actions.
        self.obs_order = ["arm_dof_pos", "arm_dof_vel", "socket_pos", "socket_quat"]
        self.policy_action_space = "joint"
        # Use inherited joint names from parent's observation configuration
        self.arm_joint_names = self.observations.policy.joint_pos.params["asset_cfg"].joint_names
        # Use inherited num_arm_joints from parent
        self.action_space = self.num_arm_joints
        # State: 7 joint pos + 7 joint vel + 3 socket pos + 4 socket quat + 3 plug pos + 4 plug quat = 28
        self.state_space = 28
        # Observation: 7 joint pos + 7 joint vel + 3 socket pos + 4 socket quat = 21
        self.observation_space = 21

        # Set joint_action_scale from the existing arm_action.scale
        self.joint_action_scale = self.actions.arm_action.scale

        # Dynamically generate action_scale_joint_space based on action_space
        self.action_scale_joint_space = [self.joint_action_scale] * self.action_space

        # Override robot initial pose for ROS inference (fixed pose, no randomization)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)  # Identity quaternion (x, y, z, w)
        self.scene.robot.init_state.joint_pos = {
            "joint1": math.radians(32.44),
            "joint2": math.radians(-16.71),
            "joint3": math.radians(-5.69),
            "joint4": math.radians(128.38),
            "joint5": math.radians(6.74),
            "joint6": math.radians(55.95),
            "joint7": math.radians(111.54),
        }

        # Override socket/plug initial poses (fixed poses for ROS inference)
        self.scene.dp_socket.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_DEPLOY_SOCKET_ROOT,
            rot=_DEPLOY_SOCKET_ROT,
        )

        self.scene.dp_plug.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_DEPLOY_PLUG_ROOT,
            rot=_DEPLOY_PLUG_ROT,
        )

        self.events.set_robot_to_grasp_pose.params["max_iterations"] = 150

        # Fixed asset parameters for ROS inference - derived from configuration
        # These parameters are used by the ROS inference node to validate the environment setup
        # and apply appropriate noise models for robust real-world deployment.
        self.fixed_asset_init_pos_center = list(_DEPLOY_GEOMETRY_POS)

        pose_range = self.events.randomize_socket_pose.params["pose_range"]
        self.fixed_asset_init_pos_range = [
            pose_range["x"][1],  # max value
            pose_range["y"][1],  # max value
            pose_range["z"][1],  # max value
        ]
        # Orientation in degrees for the vertical table-top Flexiv mount
        self.fixed_asset_init_orn_deg = [0.0, 0.0, 0.0]
        # Derive orientation range from parent's pose_range (radians to degrees)
        self.fixed_asset_init_orn_deg_range = [
            math.degrees(pose_range["roll"][1]),  # convert radians to degrees
            math.degrees(pose_range["pitch"][1]),
            math.degrees(pose_range["yaw"][1]),
        ]
        # Derive observation noise level from parent's socket_pos noise configuration
        socket_pos_noise = self.observations.policy.socket_pos.noise.noise_cfg.n_max
        self.fixed_asset_pos_obs_noise_level = [
            socket_pos_noise,
            socket_pos_noise,
            socket_pos_noise,
        ]


@configclass
class Rizon4sGravDisplayportInsertionNoJointVelROSInferenceEnvCfg(Rizon4sGravDisplayportInsertionROSInferenceEnvCfg):
    """ROS inference configuration without joint velocity in the policy observation."""

    def __post_init__(self):
        super().__post_init__()

        # Remove joint velocity from the actor observation group
        self.observations.policy.joint_vel = None

        # Update Isaac Manipulator metadata for the velocity-free actor
        self.obs_order = ["arm_dof_pos", "socket_pos", "socket_quat"]
        # Observation: 7 joint pos + 3 socket pos + 4 socket quat = 14
        self.observation_space = 14
        # State (critic) is unchanged: 7 jpos + 7 jvel + 3 socket pos + 4 socket quat + 3 plug pos + 4 plug quat = 28
        self.state_space = 28
