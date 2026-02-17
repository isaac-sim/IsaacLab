# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import os

from isaaclab_physx.physics import PhysxCfg
from isaaclab_teleop import IsaacTeleopCfg

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp

from . import stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab.controllers.config.rmp_flow import (  # isort: skip
    GALBOT_LEFT_ARM_RMPFLOW_CFG,
    GALBOT_RIGHT_ARM_RMPFLOW_CFG,
)
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


def _build_se3_rel_gripper_pipeline(hand_side="left"):
    """Build an IsaacTeleop Se3Rel + Gripper pipeline for single-arm manipulator teleoperation.

    Creates a Se3RelRetargeter for end-effector delta pose tracking and
    a GripperRetargeter for pinch-based gripper control from hand tracking data.
    All outputs are flattened into a single 7D action tensor via TensorReorderer.
    """
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3RelRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    hand_key = HandsSource.LEFT if hand_side == "left" else HandsSource.RIGHT

    # SE3 Relative Pose Retargeter
    se3_cfg = Se3RetargeterConfig(
        input_device=hand_key,
        zero_out_xy_rotation=True,
        use_wrist_rotation=False,
        use_wrist_position=True,
        delta_pos_scale_factor=10.0,
        delta_rot_scale_factor=10.0,
    )
    se3 = Se3RelRetargeter(se3_cfg, name="ee_delta")
    connected_se3 = se3.connect({hand_key: transformed_hands.output(hand_key)})

    # Gripper Retargeter (pinch-based)
    gripper_cfg = GripperRetargeterConfig(hand_side=hand_side)
    gripper = GripperRetargeter(gripper_cfg, name="gripper")
    controller_key = ControllersSource.LEFT if hand_side == "left" else ControllersSource.RIGHT
    connected_gripper = gripper.connect(
        {
            f"hand_{hand_side}": hands.output(hand_key),
            f"controller_{hand_side}": controllers.output(controller_key),
        }
    )

    # TensorReorderer: flatten into a 7D action tensor [delta_pose(6), gripper(1)]
    ee_elements = ["dx", "dy", "dz", "drx", "dry", "drz"]
    gripper_elements = ["gripper_cmd"]

    reorderer = TensorReorderer(
        input_config={
            "ee_delta": ee_elements,
            "gripper": gripper_elements,
        },
        output_order=ee_elements + gripper_elements,
        name="action_reorderer",
        input_types={
            "ee_delta": "array",
            "gripper": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "ee_delta": connected_se3.output("ee_delta"),
            "gripper": connected_gripper.output("gripper_command"),
        }
    )

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline


##
# RmpFlow Controller for Galbot Left Arm Cube Stack Task (with Parallel Gripper)
##
@configclass
class RmpFlowGalbotLeftArmCubeStackEnvCfg(stack_joint_pos_env_cfg.GalbotLeftArmCubeStackEnvCfg):
    """Configuration for the Galbot Left Arm Cube Stack Environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # read use_relative_mode from environment variable
        # True for record_demos, and False for replay_demos, annotate_demos and generate_demos
        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "True")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]

        # Set actions for the specific robot type (Galbot)
        self.actions.arm_action = RMPFlowActionCfg(
            asset_name="robot",
            joint_names=["left_arm_joint.*"],
            body_name="left_gripper_tcp_link",
            controller=GALBOT_LEFT_ARM_RMPFLOW_CFG,
            scale=1.0,
            body_offset=RMPFlowActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            articulation_prim_expr="/World/envs/env_.*/Robot",
            use_relative_mode=self.use_relative_mode,
        )

        # Set the simulation parameters
        self.sim.dt = 1 / 60
        self.sim.render_interval = 6

        self.decimation = 3
        self.episode_length_s = 30.0

        # IsaacTeleop-based teleoperation pipeline (left hand)
        pipeline = _build_se3_rel_gripper_pipeline(hand_side="left")
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )


##
# RmpFlow Controller for Galbot Right Arm Cube Stack Task (with Surface Gripper)
##
@configclass
class RmpFlowGalbotRightArmCubeStackEnvCfg(stack_joint_pos_env_cfg.GalbotRightArmCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # read use_relative_mode from environment variable
        # True for record_demos, and False for replay_demos, annotate_demos and generate_demos
        use_relative_mode_env = os.getenv("USE_RELATIVE_MODE", "True")
        self.use_relative_mode = use_relative_mode_env.lower() in ["true", "1", "t"]

        # Set actions for the specific robot type (Galbot)
        self.actions.arm_action = RMPFlowActionCfg(
            asset_name="robot",
            joint_names=["right_arm_joint.*"],
            body_name="right_suction_cup_tcp_link",
            controller=GALBOT_RIGHT_ARM_RMPFLOW_CFG,
            scale=1.0,
            body_offset=RMPFlowActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
            articulation_prim_expr="/World/envs/env_.*/Robot",
            use_relative_mode=self.use_relative_mode,
        )
        # Set the simulation parameters
        self.sim.dt = 1 / 120
        self.sim.render_interval = 6

        self.decimation = 6
        self.episode_length_s = 30.0

        # Enable CCD to avoid tunneling
        self.sim.physics = PhysxCfg(enable_ccd=True)

        # IsaacTeleop-based teleoperation pipeline (right hand)
        pipeline = _build_se3_rel_gripper_pipeline(hand_side="right")
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )


##
# Visuomotor Env for Record, Generate and Replay (in Task Space)
##
@configclass
class RmpFlowGalbotLeftArmCubeStackVisuomotorEnvCfg(RmpFlowGalbotLeftArmCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set left and right wrist cameras for VLA policy training
        self.scene.right_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_arm_camera_sim_view_frame/right_camera",
            update_period=0.0333,
            height=256,
            width=256,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(-0.5, 0.5, -0.5, 0.5), convention="ros"),
        )

        self.scene.left_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_arm_camera_sim_view_frame/left_camera",
            update_period=0.0333,
            height=256,
            width=256,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(-0.5, 0.5, -0.5, 0.5), convention="ros"),
        )

        # Set ego view camera
        self.scene.ego_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_camera_sim_view_frame/head_camera",
            update_period=0.0333,
            height=256,
            width=256,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(-0.5, 0.5, -0.5, 0.5), convention="ros"),
        )

        # Set front view camera
        self.scene.front_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_camera",
            update_period=0.0333,
            height=256,
            width=256,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(1.0, 0.0, 0.6), rot=(0.5963, 0.5963, -0.3799, -0.3799), convention="ros"),
        )

        marker_right_camera_cfg = FRAME_MARKER_CFG.copy()
        marker_right_camera_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_right_camera_cfg.prim_path = "/Visuals/FrameTransformerRightCamera"

        self.scene.right_arm_camera_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_right_camera_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_arm_camera_sim_view_frame",
                    name="right_camera",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                        rot=(-0.5, 0.5, -0.5, 0.5),
                    ),
                ),
            ],
        )

        marker_left_camera_cfg = FRAME_MARKER_CFG.copy()
        marker_left_camera_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_left_camera_cfg.prim_path = "/Visuals/FrameTransformerLeftCamera"

        self.scene.left_arm_camera_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_left_camera_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_arm_camera_sim_view_frame",
                    name="left_camera",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                        rot=(-0.5, 0.5, -0.5, 0.5),
                    ),
                ),
            ],
        )

        # Set settings for camera rendering
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"  # Use DLAA for higher quality rendering

        # List of image observations in policy observations
        self.image_obs_list = ["ego_cam", "left_wrist_cam", "right_wrist_cam"]


##
# Task Env for VLA Policy Close-loop Evaluation (in Joint Space)
##


@configclass
class GalbotLeftArmJointPositionCubeStackVisuomotorEnvCfg_PLAY(RmpFlowGalbotLeftArmCubeStackVisuomotorEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["left_arm_joint.*"], scale=1.0, use_default_offset=False
        )
        # Enable Parallel Gripper with AbsBinaryJointPosition Control
        self.actions.gripper_action = mdp.AbsBinaryJointPositionActionCfg(
            asset_name="robot",
            threshold=0.030,
            joint_names=["left_gripper_.*_joint"],
            open_command_expr={"left_gripper_.*_joint": 0.035},
            close_command_expr={"left_gripper_.*_joint": 0.023},
            # real gripper close data is 0.0235, close to it to meet data distribution,
            # but smaller to ensure robust grasping.
            # during VLA inference, we set the close command to '0.023' since the VLA
            # has never seen the gripper fully closed.
        )


##
# Task Envs for VLA Policy Close-loop Evaluation (in Task Space)
##
@configclass
class GalbotLeftArmRmpFlowCubeStackVisuomotorEnvCfg_PLAY(RmpFlowGalbotLeftArmCubeStackVisuomotorEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Enable Parallel Gripper with AbsBinaryJointPosition Control
        self.actions.gripper_action = mdp.AbsBinaryJointPositionActionCfg(
            asset_name="robot",
            threshold=0.030,
            joint_names=["left_gripper_.*_joint"],
            open_command_expr={"left_gripper_.*_joint": 0.035},
            close_command_expr={"left_gripper_.*_joint": 0.023},
        )
