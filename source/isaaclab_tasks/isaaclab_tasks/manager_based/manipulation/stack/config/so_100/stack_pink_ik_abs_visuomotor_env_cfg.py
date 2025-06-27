# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile

from pink.tasks import FrameTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.manager_based_env_cfg import LeRobotDatasetCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as base_mdp
from ... import mdp
from . import stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.so_100 import SO_100_CFG  # isort: skip


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
        table_cam = ObsTerm(
            func=base_mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False}
        )
        # wrist_cam = ObsTerm(
        #     func=base_mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False}
        # )

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
class SO100CubeStackPinkIKAbsVisuomotorEnvCfg(stack_joint_pos_env_cfg.SO100CubeStackJointPosEnvCfg):
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

        # Set SO100 as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = SO_100_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0, 0, 0.0),
                rot=(0.7071, 0, 0, 0.7071),
                joint_pos={
                    # SO100 joints
                    "shoulder_pan_joint": 0.0,
                    "shoulder_lift_joint": 1.5708,
                    "elbow_flex_joint": -1.5708,
                    "wrist_flex_joint": 1.2,
                    "wrist_roll_joint": 0.0,
                    "gripper_joint": 0.0,
                },
                joint_vel={".*": 0.0},
            ),
        )

        # Set actions for the specific robot type (SO100)
        self.actions.arm_action = PinkInverseKinematicsActionCfg(
            pink_controlled_joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_flex_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
            ],
            # Joints to be locked in URDF
            ik_urdf_fixed_joint_names=["gripper_joint"],
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
                show_ik_warnings=True,
                variable_input_tasks=[
                    FrameTask(
                        "gripper",
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

        # # Set wrist camera
        # self.scene.wrist_cam = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_cam",
        #     update_period=0.0,
        #     height=512,
        #     width=512,
        #     data_types=["rgb", "distance_to_image_plane"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
        #     ),
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(0.08, 0.0, -0.1), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
        #     ),
        # )

        # Set table view camera
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=512,
            width=512,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.4), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"
            ),
        )

        # Set settings for camera rendering
        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"  # disable dlss

        # List of image observations in policy observations
        self.image_obs_list = ["table_cam"]
        # self.image_obs_list = ["table_cam", "wrist_cam"]
        
        # Configure LeRobot dataset recording
        self.lerobot_dataset = LeRobotDatasetCfg(
            # Record specific observation keys that are useful for training
            observation_keys_to_record=[
                ["policy", "table_cam"]
            ],
            # State observations that should be combined into "observation.state"
            state_observation_keys=[
                ["policy", "joint_pos"] 
            ],
            # Task description for the dataset
            task_description="Stack the red cube on top of the blue cube",
        )
        