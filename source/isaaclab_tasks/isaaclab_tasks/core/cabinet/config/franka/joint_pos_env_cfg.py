# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the manager-based Franka cabinet-opening environment."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.utils import preset

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from ... import mdp
from ...cabinet_env_cfg import FRAME_MARKER_SMALL_CFG, CabinetEnvCfg, CabinetSceneCfg, RewardsCfg


@configclass
class FrankaCabinetSceneCfg(CabinetSceneCfg):
    """Cabinet scene configured for the Franka robot."""

    robot = replace(FRANKA_PANDA_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.variants["Physics"] = preset(default="mujoco", isaacsim_physx="physx", physx="physx", ovphysx="physx")
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(Geometry/)?panda_link0",
        debug_vis=False,
        visualizer_cfg=replace(FRAME_MARKER_SMALL_CFG, prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/(Geometry/.*/)?panda_hand",
                name="ee_tcp",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1034),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/(Geometry/.*/)?panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/(Geometry/.*/)?panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
        ],
    )


@configclass
class FrankaCabinetRewardsCfg(RewardsCfg):
    """Penalize arm motion near position limits and above the requested speed."""

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-100.0,
        params={"soft_ratio": 1.0, "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )


@configclass
class FrankaCabinetEnvCfg(CabinetEnvCfg):
    """Cabinet-opening environment with a Franka Panda arm driven by joint position targets."""

    scene: FrankaCabinetSceneCfg = FrankaCabinetSceneCfg(num_envs=4096, env_spacing=2.0)
    rewards: FrankaCabinetRewardsCfg = FrankaCabinetRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # actions
        self.actions.arm_action = mdp.RateLimitedJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=True,
            max_velocity=0.25,
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
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.joint_vel.weight = -5.0

        # Keep a 5% margin at each end of the joint range.
        self.scene.robot.soft_joint_pos_limit_factor = 0.9
        # Retract the hand so randomized resets clear the lower cabinet doors and knobs.
        self.scene.robot.init_state.joint_pos["panda_joint2"] = -1.1
        self.scene.robot.actuators["panda_arm"].damping = 80.0
        # MJWarp does not enforce this limit; the action term bounds commanded motion instead.
        self.scene.robot.actuators["panda_arm"].joint_velocity_limit = 0.3
        self.episode_length_s = 16.0

    def play_mode(self):
        super().play_mode()
        # make a smaller scene for play
        self.scene.env_spacing = 2.5
