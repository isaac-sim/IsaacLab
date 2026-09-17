# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field
from typing import Any

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import replace_config

import isaaclab_tasks.core.cabinet.mdp as mdp
from isaaclab_tasks.core.cabinet.cabinet_env_cfg import FRAME_MARKER_SMALL_CFG, CabinetEnvCfg, CabinetSceneCfg

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


@dataclass
class FrankaCabinetSceneCfg(CabinetSceneCfg):
    """Cabinet scene configured for the Franka robot."""

    robot: Any = field(default_factory=lambda: replace_config(FRANKA_PANDA_CFG, prim_path="{ENV_REGEX_NS}/Robot"))
    ee_frame: Any = field(
        default_factory=lambda: FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=replace_config(FRAME_MARKER_SMALL_CFG, prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
    )


@dataclass
class FrankaCabinetEnvCfg(CabinetEnvCfg):
    scene: FrankaCabinetSceneCfg = field(default_factory=lambda: FrankaCabinetSceneCfg(num_envs=4096, env_spacing=2.0))

    def __post_init__(self):
        if parent_post_init := getattr(super(), "__post_init__", None):
            parent_post_init()

        # actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # override rewards
        self.rewards.approach_gripper_handle.params["offset"] = 0.04
        self.rewards.grasp_handle.params["open_joint_pos"] = 0.04
        self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["panda_finger_.*"]

    def play_mode(self):
        super().play_mode()

        # make a smaller scene for play
        self.scene.env_spacing = 2.5
