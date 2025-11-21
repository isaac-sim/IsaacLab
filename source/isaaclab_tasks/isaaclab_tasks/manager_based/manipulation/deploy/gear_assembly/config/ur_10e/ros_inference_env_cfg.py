# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
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
        self.arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.action_space = 6
        self.state_space = 42
        self.observation_space = 19

        # Set joint_action_scale from the existing arm_action.scale
        self.joint_action_scale = self.actions.arm_action.scale

        self.action_scale_joint_space = [
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
        ]

        # Fixed asset parameters for ROS inference
        # These parameters are used by the ROS inference node to validate the environment setup
        # and apply appropriate noise models for robust real-world deployment.
        self.fixed_asset_init_pos_center = [1.02, -0.21, -0.1]
        self.fixed_asset_init_pos_range = [0.1, 0.25, 0.1]
        self.fixed_asset_init_orn_deg = [0.0, 0.0, -90.0]
        self.fixed_asset_init_orn_deg_range = [2.0, 2.0, 30.0]
        self.fixed_asset_pos_obs_noise_level = [0.0025, 0.0025, 0.0025]

        # Override robot initial pose for ROS inference (fixed pose, no randomization)
        # These joint positions are tuned for a good starting configuration.
        # Note: The policy is trained to work with respect to the UR robot's 'base' frame
        # (rotated 180° around Z from base_link), not the base_link frame (USD origin).
        # See: https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_description/doc/robot_frames.html
        self.scene.robot.init_state = ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 2.7228,
                "shoulder_lift_joint": -8.3962e-01,
                "elbow_joint": 1.3684,
                "wrist_1_joint": -2.1048,
                "wrist_2_joint": -1.5691,
                "wrist_3_joint": -1.9896,
                "finger_joint": 0.0,
                ".*_inner_finger_joint": 0.0,
                ".*_inner_finger_pad_joint": 0.0,
                ".*_outer_.*_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        )

        # Override gear base initial pose (fixed pose for ROS inference)
        self.scene.factory_gear_base.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(1.0200, -0.2100, -0.1),
            rot=(-0.70711, 0.0, 0.0, 0.70711),
        )

        # Override gear initial poses (fixed poses for ROS inference)
        # Small gear
        self.scene.factory_gear_small.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(1.0200, -0.2100, -0.1),  # z = base_z + 0.1675 (above base)
            rot=(-0.70711, 0.0, 0.0, 0.70711),
        )

        # Medium gear
        self.scene.factory_gear_medium.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(1.0200, -0.2100, -0.1),
            rot=(-0.70711, 0.0, 0.0, 0.70711),
        )

        # Large gear
        self.scene.factory_gear_large.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(1.0200, -0.2100, -0.1),
            rot=(-0.70711, 0.0, 0.0, 0.70711),
        )


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
        # TODO: @ashwinvk: Remove these from env cfg once the generic inference node has been implemented
        self.obs_order = ["arm_dof_pos", "arm_dof_vel", "shaft_pos", "shaft_quat"]
        self.policy_action_space = "joint"
        self.arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.action_space = 6
        self.state_space = 42
        self.observation_space = 19

        # Set joint_action_scale from the existing arm_action.scale
        self.joint_action_scale = self.actions.arm_action.scale

        self.action_scale_joint_space = [
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
        ]

        # Fixed asset parameters for ROS inference
        # These parameters are used by the ROS inference node to validate the environment setup
        # and apply appropriate noise models for robust real-world deployment.
        self.fixed_asset_init_pos_center = [1.02, -0.21, -0.1]
        self.fixed_asset_init_pos_range = [0.1, 0.25, 0.1]
        self.fixed_asset_init_orn_deg = [0.0, 0.0, -90.0]
        self.fixed_asset_init_orn_deg_range = [2.0, 2.0, 30.0]
        self.fixed_asset_pos_obs_noise_level = [0.0025, 0.0025, 0.0025]

        # Override robot initial pose for ROS inference (fixed pose, no randomization)
        # These joint positions are tuned for a good starting configuration.
        # Note: The policy is trained to work with respect to the UR robot's 'base' frame
        # (rotated 180° around Z from base_link), not the base_link frame (USD origin).
        # See: https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_description/doc/robot_frames.html
        self.scene.robot.init_state = ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 2.7228,
                "shoulder_lift_joint": -8.3962e-01,
                "elbow_joint": 1.3684,
                "wrist_1_joint": -2.1048,
                "wrist_2_joint": -1.5691,
                "wrist_3_joint": -1.9896,
                "finger_joint": 0.0,
                ".*_inner_finger_joint": 0.0,
                ".*_inner_finger_knuckle_joint": 0.0,
                ".*_outer_.*_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        )

        # Override gear base initial pose (fixed pose for ROS inference)
        self.scene.factory_gear_base.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(1.0200, -0.2100, -0.1),
            rot=(-0.70711, 0.0, 0.0, 0.70711),
        )

        # Override gear initial poses (fixed poses for ROS inference)
        # Small gear
        self.scene.factory_gear_small.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(1.0200, -0.2100, -0.1),  # z = base_z + 0.1675 (above base)
            rot=(-0.70711, 0.0, 0.0, 0.70711),
        )

        # Medium gear
        self.scene.factory_gear_medium.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(1.0200, -0.2100, -0.1),
            rot=(-0.70711, 0.0, 0.0, 0.70711),
        )

        # Large gear
        self.scene.factory_gear_large.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(1.0200, -0.2100, -0.1),
            rot=(-0.70711, 0.0, 0.0, 0.70711),
        )
