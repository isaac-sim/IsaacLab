# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the H1_2 pick and place environment."""

from isaaclab_assets.robots.unitree import H1_2_MINIMAL_CFG
from isaaclab.devices import DevicesCfg, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters import H1_2RetargeterCfg
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_base_env_cfg import PickPlaceBaseEnvCfg
from isaaclab.utils import configclass


@configclass
class PickPlaceH1_2EnvCfg(PickPlaceBaseEnvCfg):
    """Configuration for the H1_2 environment."""

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 6
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = 2

        # Set the robot to H1_2
        self.scene.robot = H1_2_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Configure teleop devices for H1_2
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        H1_2RetargeterCfg(
                            enable_visualization=True,
                            # OpenXR hand tracking has 26 joints per hand
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            # Define H1_2 specific joint names
                            hand_joint_names=[
                                "left_hand_joint",
                                "right_hand_joint",
                                # Add more specific hand joint names as needed
                            ],
                            arm_joint_names=[
                                "left_shoulder_pitch",
                                "left_shoulder_roll", 
                                "left_shoulder_yaw",
                                "left_elbow",
                                "right_shoulder_pitch",
                                "right_shoulder_roll",
                                "right_shoulder_yaw", 
                                "right_elbow",
                            ],
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )

        # Configure actions for H1_2
        # You may need to adjust this based on H1_2's actual action space
        self.actions.pink_ik_cfg.hand_joint_names = [
            "left_hand_joint",
            "right_hand_joint",
        ]
        
        # Set action space dimensions based on H1_2
        # This should match the output of the H1_2Retargeter
        # 7 (left wrist) + 7 (right wrist) + hand_joints + arm_joints
        self.actions.pink_ik_cfg.action_space = 14 + len(self.actions.pink_ik_cfg.hand_joint_names) + 8  # 8 arm joints
