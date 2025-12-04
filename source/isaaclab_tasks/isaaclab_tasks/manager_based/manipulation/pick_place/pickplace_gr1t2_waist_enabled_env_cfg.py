# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import tempfile

import isaaclab.controllers.utils as ControllerUtils
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.retargeters import DexBiManualRetargeterCfg, DexHandRetargeterCfg, Se3AbsRetargeterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from .pickplace_gr1t2_env_cfg import ActionsCfg, EventCfg, ObjectTableSceneCfg, ObservationsCfg, TerminationsCfg


@configclass
class PickPlaceGR1T2WaistEnabledEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    # OpenXR hand tracking has 26 joints per hand
    NUM_OPENXR_HAND_JOINTS = 26

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 6
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = 2

        # Add waist joint to pink_ik_cfg
        waist_joint_names = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
        for joint_name in waist_joint_names:
            self.actions.upper_body_ik.pink_controlled_joint_names.append(joint_name)

        # Convert USD to URDF and change revolute joints to fixed
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.upper_body_ik.controller.urdf_path = temp_urdf_output_path
        self.actions.upper_body_ik.controller.mesh_path = temp_urdf_meshes_output_path

        # Extract hand joint names for each hand
        original_hand_joints = self.actions.upper_body_ik.hand_joint_names
        left_hand_joints = [j for j in original_hand_joints if "L_" in j]
        right_hand_joints = [j for j in original_hand_joints if "R_" in j]

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3AbsRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                            target_offset_rot=(1.0, 0.0, 0.0, 0.0),
                            zero_out_xy_rotation=False,
                            use_wrist_rotation=True,
                            use_wrist_position=True,
                            sim_device=self.sim.device,
                        ),
                        Se3AbsRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            target_offset_rot=(0.0, 0.0, 0.0, 1.0),
                            zero_out_xy_rotation=False,
                            use_wrist_rotation=True,
                            use_wrist_position=True,
                            sim_device=self.sim.device,
                        ),
                        DexBiManualRetargeterCfg(
                            target_joint_names=original_hand_joints,
                            sim_device=self.sim.device,
                            left_hand_cfg=DexHandRetargeterCfg(
                                target=DeviceBase.TrackingTarget.HAND_LEFT,
                                enable_visualization=True,
                                # number of joints in both hands
                                num_open_xr_hand_joints=self.NUM_OPENXR_HAND_JOINTS,
                                sim_device=self.sim.device,
                                hand_joint_names=left_hand_joints,
                                hand_retargeting_config=os.path.join(
                                    os.path.dirname(__file__), "config/dex_retargeting/fourier_hand_left_dexpilot.yml"
                                ),
                                hand_urdf=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/GR1T2_assets/GR1_T2_left_hand.urdf",
                                handtracking_to_baselink_frame_transform=(0, -1, 0, -1, 0, 0, 0, 0, -1),
                            ),
                            right_hand_cfg=DexHandRetargeterCfg(
                                target=DeviceBase.TrackingTarget.HAND_RIGHT,
                                enable_visualization=True,
                                # number of joints in both hands
                                num_open_xr_hand_joints=self.NUM_OPENXR_HAND_JOINTS,
                                sim_device=self.sim.device,
                                hand_joint_names=right_hand_joints,
                                hand_retargeting_config=os.path.join(
                                    os.path.dirname(__file__), "config/dex_retargeting/fourier_hand_right_dexpilot.yml"
                                ),
                                hand_urdf=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/GR1T2_assets/GR1_T2_right_hand.urdf",
                                handtracking_to_baselink_frame_transform=(0, -1, 0, -1, 0, 0, 0, 0, -1),
                            ),
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
