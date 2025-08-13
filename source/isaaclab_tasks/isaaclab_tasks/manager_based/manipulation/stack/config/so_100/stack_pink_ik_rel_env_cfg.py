# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import tempfile

from pink.tasks import FrameTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
from isaaclab.controllers.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from . import stack_joint_pos_env_cfg


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=base_mdp.last_action)
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class SO100CubeStackPinkIKRelEnvCfg(stack_joint_pos_env_cfg.SO100CubeStackJointPosEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Temporary directory for URDF files
        self.temp_urdf_dir = tempfile.gettempdir()
        # Convert USD to URDF and change revolute joints to fixed
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )

        # Set actions to use pink IK controller
        self.actions.arm_action = PinkInverseKinematicsActionCfg(
            pink_controlled_joint_names=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
            # Joints to be locked in URDF
            ik_urdf_fixed_joint_names=["joints_gripper"],
            hand_joint_names=[],
            # the robot in the sim scene we are controlling
            asset_name="robot",
            # Configuration for the IK controller
            # The frames names are the ones present in the URDF file
            # The urdf has to be generated from the USD that is being used in the scene
            controller=PinkIKControllerCfg(
                articulation_name="robot",
                base_link_name="base",
                num_hand_joints=0,
                use_relative_mode=True,
                show_ik_warnings=True,
                variable_input_tasks=[
                    FrameTask(
                        "joints_gripper",
                        position_cost=1.0,  # [cost] / [m]
                        orientation_cost=1.0,  # [cost] / [rad]
                        lm_damping=10,  # dampening for solver for step jumps
                        gain=0.1,
                    )
                ],
                fixed_input_tasks=[],
            ),
        )
        ControllerUtils.change_revolute_to_fixed(
            temp_urdf_output_path, self.actions.arm_action.ik_urdf_fixed_joint_names
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.arm_action.controller.urdf_path = temp_urdf_output_path
        self.actions.arm_action.controller.mesh_path = temp_urdf_meshes_output_path
