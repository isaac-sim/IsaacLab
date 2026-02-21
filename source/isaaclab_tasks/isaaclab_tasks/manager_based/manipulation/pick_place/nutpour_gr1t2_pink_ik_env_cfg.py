# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging

from pink.tasks import DampingTask, FrameTask

try:
    from isaaclab_teleop import IsaacTeleopCfg

    _TELEOP_AVAILABLE = True
except ImportError:
    _TELEOP_AVAILABLE = False
    logging.getLogger(__name__).warning("isaaclab_teleop is not installed. XR teleoperation features will be disabled.")

import isaaclab.controllers.utils as ControllerUtils
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place.nutpour_gr1t2_base_env_cfg import NutPourGR1T2BaseEnvCfg
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import (
    _build_gr1t2_pickplace_pipeline,
)


@configclass
class NutPourGR1T2PinkIKEnvCfg(NutPourGR1T2BaseEnvCfg):
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
            target_eef_link_names={
                "left_wrist": "left_hand_pitch_link",
                "right_wrist": "right_hand_pitch_link",
            },
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
                # Determines whether Pink IK solver will fail due to a joint limit violation
                fail_on_joint_limit_violation=False,
                variable_input_tasks=[
                    FrameTask(
                        "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                        position_cost=8.0,  # [cost] / [m]
                        orientation_cost=1.0,  # [cost] / [rad]
                        lm_damping=10,  # dampening for solver for step jumps
                        gain=0.5,
                    ),
                    FrameTask(
                        "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                        position_cost=8.0,  # [cost] / [m]
                        orientation_cost=1.0,  # [cost] / [rad]
                        lm_damping=10,  # dampening for solver for step jumps
                        gain=0.5,
                    ),
                    DampingTask(
                        cost=0.5,  # [cost] * [s] / [rad]
                    ),
                    NullSpacePostureTask(
                        cost=0.2,
                        lm_damping=1,
                        controlled_frames=[
                            "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                            "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                        ],
                        controlled_joints=[
                            "left_shoulder_pitch_joint",
                            "left_shoulder_roll_joint",
                            "left_shoulder_yaw_joint",
                            "left_elbow_pitch_joint",
                            "right_shoulder_pitch_joint",
                            "right_shoulder_roll_joint",
                            "right_shoulder_yaw_joint",
                            "right_elbow_pitch_joint",
                            "waist_yaw_joint",
                            "waist_pitch_joint",
                            "waist_roll_joint",
                        ],
                    ),
                ],
                fixed_input_tasks=[],
            ),
        )
        # Convert USD to URDF and change revolute joints to fixed
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.gr1_action.controller.urdf_path = temp_urdf_output_path
        self.actions.gr1_action.controller.mesh_path = temp_urdf_meshes_output_path

        # IsaacTeleop-based teleoperation pipeline
        if _TELEOP_AVAILABLE:
            pipeline = _build_gr1t2_pickplace_pipeline()
            self.isaac_teleop = IsaacTeleopCfg(
                pipeline_builder=lambda: pipeline,
                sim_device=self.sim.device,
                xr_cfg=self.xr,
            )
