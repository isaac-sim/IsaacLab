# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    HolonomicBaseActionCfg,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.mobile_manipulation.reach.reach_env_cfg import MobileReachEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaMobileReachEnvCfg(MobileReachEnvCfg):
    """Configuration for Franka arm on mobile base reach environment."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Setup mobile Franka
        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),  # Base starts at ground level
                rot=(1.0, 0.0, 0.0, 0.0),  # Identity rotation
                joint_pos={
                    ".*": 0.0,  # Default pose for all joints
                    # Specific joint overrides for good starting pose
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.785,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356,
                    "panda_joint5": 0.0,
                    "panda_joint6": 1.571,
                    "panda_joint7": 0.785,
                },
            ),
            spawn=sim_utils.UsdComposerCfg(
                components=[
                    # Mobile base
                    sim_utils.HolonomicBaseCfg(
                        mass=50.0,
                        size=(0.6, 0.6, 0.4),  # Size to stably support arm
                        wheel_radius=0.1,
                        wheel_width=0.05,
                        wheel_friction=0.7,
                        wheel_dampening=10.0,
                    ),
                    # Franka arm
                    FRANKA_PANDA_HIGH_PD_CFG.spawn,
                ],
                component_positions=[
                    (0.0, 0.0, 0.2),  # Base center at origin
                    (0.0, 0.0, 0.4),  # Arm mounted on top of base
                ],
                component_orientations=[
                    (1.0, 0.0, 0.0, 0.0),  # Base aligned with world
                    (1.0, 0.0, 0.0, 0.0),  # Arm aligned with base
                ],
            ),
        )

        # Setup actions
        self.actions.base_action = HolonomicBaseActionCfg(
            asset_name="robot",
            joint_names=[".*_wheel_.*"],  # All wheel joints
            scale=1.0,  # Max 1 m/s and 1 rad/s
        )

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],  # Arm joints only
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.5,  # Slower arm movement for coordination
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.107],  # Offset to end effector
            ),
        )

        # Setup end-effector frame tracking
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.107],
                    ),
                ),
            ],
        )

        # Set target body for rewards
        self.rewards.ee_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.ee_position_tracking_fine.params["asset_cfg"].body_names = ["panda_hand"]

        # Adjust command ranges for larger workspace
        self.commands.ee_pose.ranges.pos_x = (-2.0, 2.0)
        self.commands.ee_pose.ranges.pos_y = (-2.0, 2.0)
        self.commands.ee_pose.ranges.pos_z = (0.3, 0.8)

        # Adjust environment settings
        self.scene.env_spacing = 4.0  # More space for mobile base
        self.episode_length_s = 8.0  # Slightly longer episodes
        self.viewer.eye = (6.0, 6.0, 4.0)  # Wider view for mobile workspace


@configclass
class FrankaMobileReachEnvCfg_PLAY(FrankaMobileReachEnvCfg):
    """Play configuration with smaller number of environments."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 4.0
        # Disable randomization for play
        self.observations.policy.enable_corruption = False