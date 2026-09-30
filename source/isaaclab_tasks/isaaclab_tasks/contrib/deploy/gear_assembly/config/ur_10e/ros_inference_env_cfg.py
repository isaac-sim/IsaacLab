# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from .joint_pos_env_cfg import UR10e2F85GearAssemblyEnvCfg, UR10e2F140GearAssemblyEnvCfg


@configclass
class UR10e2F140GearAssemblyROSInferenceEnvCfg(UR10e2F140GearAssemblyEnvCfg):
    """Configuration for ROS inference with UR10e and Robotiq 2F-140 gripper.

    This configuration:
    - Exposes variables needed for ROS inference
    - Overrides robot and gear initial poses for fixed/deterministic setup
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Variables used by Isaac Manipulator for on robot inference
        # These parameters allow the ROS inference node to validate environment configuration,
        # perform checks during inference, and correctly interpret observations and actions.
        self.obs_order = ["arm_dof_pos", "arm_dof_vel", "shaft_pos", "shaft_quat"]
        self.policy_action_space = "joint"
        # Use inherited joint names from parent's observation configuration
        self.arm_joint_names = self.observations.policy.joint_pos.params["asset_cfg"].joint_names
        # Use inherited num_arm_joints from parent
        self.action_space = self.num_arm_joints
        # State space and observation space are set as constants for now
        self.state_space = 42
        self.observation_space = 19

        # Set joint_action_scale from the existing arm_action.scale
        self.joint_action_scale = self.actions.arm_action.scale

        # Dynamically generate action_scale_joint_space based on action_space
        self.action_scale_joint_space = [self.joint_action_scale] * self.action_space

        # Extract initial joint positions from robot configuration
        # Convert joint_pos dict to list in the order specified by arm_joint_names
        self.initial_joint_pos = [
            self.scene.robot.init_state.joint_pos[joint_name] for joint_name in self.arm_joint_names
        ]

        # Override robot initial pose for ROS inference (fixed pose, no randomization)
        # Note: The policy is trained to work with respect to the UR robot's 'base' frame
        # (rotated 180° around Z from base_link), not the base_link frame (USD origin).
        # See: https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_description/doc/robot_frames.html
        # Joint positions and pos are inherited from parent, only override rotation to be deterministic
        self.scene.robot.init_state.rot = (0.0, 0.0, 1.0, 0.0)

        # Override gear base initial pose (fixed pose for ROS inference)
        self.scene.factory_gear_base.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(1.0200, -0.2100, -0.1),
            rot=(0.0, 0.0, 0.70711, -0.70711),
        )

        # Override gear initial poses (fixed poses for ROS inference)
        # Small gear
        self.scene.factory_gear_small.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(1.0200, -0.2100, -0.1),  # z = base_z + 0.1675 (above base)
            rot=(0.0, 0.0, 0.70711, -0.70711),
        )

        # Medium gear
        self.scene.factory_gear_medium.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(1.0200, -0.2100, -0.1),
            rot=(0.0, 0.0, 0.70711, -0.70711),
        )

        # Large gear
        self.scene.factory_gear_large.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(1.0200, -0.2100, -0.1),
            rot=(0.0, 0.0, 0.70711, -0.70711),
        )

        # Fixed asset parameters for ROS inference - derived from configuration
        # These parameters are used by the ROS inference node to validate the environment setup
        # and apply appropriate noise models for robust real-world deployment.
        # Derive position center from gear base init state
        self.fixed_asset_init_pos_center = list(self.scene.factory_gear_base.init_state.pos)
        # Derive position range from parent's randomize_gears_and_base_pose event pose_range
        pose_range = self.events.randomize_gears_and_base_pose.params["pose_range"]
        self.fixed_asset_init_pos_range = [
            pose_range["x"][1],  # max value
            pose_range["y"][1],  # max value
            pose_range["z"][1],  # max value
        ]
        # Orientation in degrees (quaternion (-0.70711, 0.0, 0.0, 0.70711) = -90° around Z)
        self.fixed_asset_init_orn_deg = [0.0, 0.0, -90.0]
        # Derive orientation range from parent's pose_range (radians to degrees)
        self.fixed_asset_init_orn_deg_range = [
            math.degrees(pose_range["roll"][1]),  # convert radians to degrees
            math.degrees(pose_range["pitch"][1]),
            math.degrees(pose_range["yaw"][1]),
        ]
        # Derive observation noise level from parent's gear_shaft_pos noise configuration
        gear_shaft_pos_noise = self.observations.policy.gear_shaft_pos.noise.noise_cfg.n_max
        self.fixed_asset_pos_obs_noise_level = [
            gear_shaft_pos_noise,
            gear_shaft_pos_noise,
            gear_shaft_pos_noise,
        ]


@configclass
class UR10e2F85GearAssemblyROSInferenceEnvCfg(UR10e2F85GearAssemblyEnvCfg):
    """Configuration for ROS inference with UR10e and Robotiq 2F-85 gripper.

    This configuration:
    - Exposes variables needed for ROS inference
    - Overrides robot and gear initial poses for fixed/deterministic setup
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Variables used by Isaac Manipulator for on robot inference
        # These parameters allow the ROS inference node to validate environment configuration,
        # perform checks during inference, and correctly interpret observations and actions.
        self.obs_order = ["arm_dof_pos", "arm_dof_vel", "shaft_pos", "shaft_quat"]
        self.policy_action_space = "joint"
        # Use inherited joint names from parent's observation configuration
        self.arm_joint_names = self.observations.policy.joint_pos.params["asset_cfg"].joint_names
        # Use inherited num_arm_joints from parent
        self.action_space = self.num_arm_joints
        # State space and observation space are set as constants for now
        self.state_space = 38
        self.observation_space = 19

        # Set joint_action_scale from the existing arm_action.scale
        self.joint_action_scale = self.actions.arm_action.scale

        # Dynamically generate action_scale_joint_space based on action_space
        self.action_scale_joint_space = [self.joint_action_scale] * self.action_space

        # Extract initial joint positions from robot configuration
        # Convert joint_pos dict to list in the order specified by arm_joint_names
        self.initial_joint_pos = [
            self.scene.robot.init_state.joint_pos[joint_name] for joint_name in self.arm_joint_names
        ]

        # Override robot initial pose for ROS inference (fixed pose, no randomization)
        # Note: The policy is trained to work with respect to the UR robot's 'base' frame
        # (rotated 180° around Z from base_link), not the base_link frame (USD origin).
        # See: https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_description/doc/robot_frames.html
        # Joint positions and pos are inherited from parent, only override rotation to be deterministic
        self.scene.robot.init_state.rot = (0.0, 0.0, 1.0, 0.0)

        # Override gear base initial pose (fixed pose for ROS inference)
        self.scene.factory_gear_base.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(1.0200, -0.2100, -0.1),
            rot=(0.0, 0.0, 0.70711, -0.70711),
        )

        # Override gear initial poses (fixed poses for ROS inference)
        # Small gear
        self.scene.factory_gear_small.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(1.0200, -0.2100, -0.1),  # z = base_z + 0.1675 (above base)
            rot=(0.0, 0.0, 0.70711, -0.70711),
        )

        # Medium gear
        self.scene.factory_gear_medium.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(1.0200, -0.2100, -0.1),
            rot=(0.0, 0.0, 0.70711, -0.70711),
        )

        # Large gear
        self.scene.factory_gear_large.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(1.0200, -0.2100, -0.1),
            rot=(0.0, 0.0, 0.70711, -0.70711),
        )

        # Fixed asset parameters for ROS inference - derived from configuration
        # These parameters are used by the ROS inference node to validate the environment setup
        # and apply appropriate noise models for robust real-world deployment.
        # Derive position center from gear base init state
        self.fixed_asset_init_pos_center = list(self.scene.factory_gear_base.init_state.pos)
        # Derive position range from parent's randomize_gears_and_base_pose event pose_range
        pose_range = self.events.randomize_gears_and_base_pose.params["pose_range"]
        self.fixed_asset_init_pos_range = [
            pose_range["x"][1],  # max value
            pose_range["y"][1],  # max value
            pose_range["z"][1],  # max value
        ]
        # Orientation in degrees (quaternion (-0.70711, 0.0, 0.0, 0.70711) = -90° around Z)
        self.fixed_asset_init_orn_deg = [0.0, 0.0, -90.0]
        # Derive orientation range from parent's pose_range (radians to degrees)
        self.fixed_asset_init_orn_deg_range = [
            math.degrees(pose_range["roll"][1]),  # convert radians to degrees
            math.degrees(pose_range["pitch"][1]),
            math.degrees(pose_range["yaw"][1]),
        ]
        # Derive observation noise level from parent's gear_shaft_pos noise configuration
        gear_shaft_pos_noise = self.observations.policy.gear_shaft_pos.noise.noise_cfg.n_max
        self.fixed_asset_pos_obs_noise_level = [
            gear_shaft_pos_noise,
            gear_shaft_pos_noise,
            gear_shaft_pos_noise,
        ]
