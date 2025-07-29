# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pink.tasks import FrameTask

import isaaclab.controllers.utils as ControllerUtils
from isaaclab.controllers.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters import GR1T2RetargeterCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place.exhaustpipe_gr1t2_base_env_cfg import (
    ExhaustPipeGR1T2BaseEnvCfg,
)


@configclass
class ExhaustPipeGR1T2PinkIKEnvCfg(ExhaustPipeGR1T2BaseEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.actions.gr1_action = PinkInverseKinematicsActionCfg(
            pink_controlled_joint_names=[
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_pitch_joint",
                "left_wrist_yaw_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_pitch_joint",
                "right_wrist_yaw_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
            ],
            # Joints to be locked in URDF
            ik_urdf_fixed_joint_names=[
                "left_hip_roll_joint",
                "right_hip_roll_joint",
                "left_hip_yaw_joint",
                "right_hip_yaw_joint",
                "left_hip_pitch_joint",
                "right_hip_pitch_joint",
                "left_knee_pitch_joint",
                "right_knee_pitch_joint",
                "left_ankle_pitch_joint",
                "right_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_ankle_roll_joint",
                "L_index_proximal_joint",
                "L_middle_proximal_joint",
                "L_pinky_proximal_joint",
                "L_ring_proximal_joint",
                "L_thumb_proximal_yaw_joint",
                "R_index_proximal_joint",
                "R_middle_proximal_joint",
                "R_pinky_proximal_joint",
                "R_ring_proximal_joint",
                "R_thumb_proximal_yaw_joint",
                "L_index_intermediate_joint",
                "L_middle_intermediate_joint",
                "L_pinky_intermediate_joint",
                "L_ring_intermediate_joint",
                "L_thumb_proximal_pitch_joint",
                "R_index_intermediate_joint",
                "R_middle_intermediate_joint",
                "R_pinky_intermediate_joint",
                "R_ring_intermediate_joint",
                "R_thumb_proximal_pitch_joint",
                "L_thumb_distal_joint",
                "R_thumb_distal_joint",
                "head_roll_joint",
                "head_pitch_joint",
                "head_yaw_joint",
                "waist_yaw_joint",
                "waist_pitch_joint",
                "waist_roll_joint",
            ],
            hand_joint_names=[
                "L_index_proximal_joint",
                "L_middle_proximal_joint",
                "L_pinky_proximal_joint",
                "L_ring_proximal_joint",
                "L_thumb_proximal_yaw_joint",
                "R_index_proximal_joint",
                "R_middle_proximal_joint",
                "R_pinky_proximal_joint",
                "R_ring_proximal_joint",
                "R_thumb_proximal_yaw_joint",
                "L_index_intermediate_joint",
                "L_middle_intermediate_joint",
                "L_pinky_intermediate_joint",
                "L_ring_intermediate_joint",
                "L_thumb_proximal_pitch_joint",
                "R_index_intermediate_joint",
                "R_middle_intermediate_joint",
                "R_pinky_intermediate_joint",
                "R_ring_intermediate_joint",
                "R_thumb_proximal_pitch_joint",
                "L_thumb_distal_joint",
                "R_thumb_distal_joint",
            ],
            # the robot in the sim scene we are controlling
            asset_name="robot",
            # Configuration for the IK controller
            # The frames names are the ones present in the URDF file
            # The urdf has to be generated from the USD that is being used in the scene
            controller=PinkIKControllerCfg(
                articulation_name="robot",
                base_link_name="base_link",
                num_hand_joints=22,
                show_ik_warnings=False,
                variable_input_tasks=[
                    FrameTask(
                        "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                        position_cost=1.0,  # [cost] / [m]
                        orientation_cost=1.0,  # [cost] / [rad]
                        lm_damping=10,  # dampening for solver for step jumps
                        gain=0.1,
                    ),
                    FrameTask(
                        "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                        position_cost=1.0,  # [cost] / [m]
                        orientation_cost=1.0,  # [cost] / [rad]
                        lm_damping=10,  # dampening for solver for step jumps
                        gain=0.1,
                    ),
                ],
                fixed_input_tasks=[
                    # COMMENT OUT IF LOCKING WAIST/HEAD
                    # FrameTask(
                    #     "GR1T2_fourier_hand_6dof_head_yaw_link",
                    #     position_cost=1.0,  # [cost] / [m]
                    #     orientation_cost=0.05,  # [cost] / [rad]
                    # ),
                ],
            ),
        )
        # Convert USD to URDF and change revolute joints to fixed
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )
        ControllerUtils.change_revolute_to_fixed(
            temp_urdf_output_path, self.actions.gr1_action.ik_urdf_fixed_joint_names
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.gr1_action.controller.urdf_path = temp_urdf_output_path
        self.actions.gr1_action.controller.mesh_path = temp_urdf_meshes_output_path

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        GR1T2RetargeterCfg(
                            enable_visualization=True,
                            # OpenXR hand tracking has 26 joints per hand
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.gr1_action.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )
