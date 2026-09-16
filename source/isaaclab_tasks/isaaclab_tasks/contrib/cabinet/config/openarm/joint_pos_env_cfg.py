# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Pre-defined configs
##
from dataclasses import dataclass

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab_tasks.core.cabinet import mdp

from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG

from isaaclab_tasks.contrib.cabinet.config.openarm.cabinet_openarm_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
    CabinetEnvCfg,
)
from isaaclab.utils import replace_config


@dataclass
class OpenArmCabinetEnvCfg(CabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        if parent_post_init := getattr(super(), "__post_init__", None):
            parent_post_init()

        # Set OpenArm as robot
        self.scene.robot = replace_config(OPENARM_UNI_CFG, prim_path="{ENV_REGEX_NS}/Robot")

        # Set Actions for the specific robot type (OpenArm)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_joint.*"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_finger_joint.*"],
            open_command_expr={"openarm_finger_joint.*": 0.044},
            close_command_expr={"openarm_finger_joint.*": 0.0},
        )

        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_link0",
            visualizer_cfg=replace_config(FRAME_MARKER_SMALL_CFG, prim_path="/Visuals/EndEffectorFrameTransformer"),
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_ee_tcp",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, -0.003),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_left_finger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, -0.005, 0.075),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/openarm_right_finger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.005, 0.075),
                    ),
                ),
            ],
        )

        # override rewards
        self.rewards.approach_gripper_handle.params["offset"] = 0.04
        self.rewards.grasp_handle.params["open_joint_pos"] = 0.044
        self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["openarm_finger_joint.*"]

    def play_mode(self):
        # play-mode overrides of parent
        super().play_mode()

        # make a smaller scene for play
        self.scene.env_spacing = 2.5
