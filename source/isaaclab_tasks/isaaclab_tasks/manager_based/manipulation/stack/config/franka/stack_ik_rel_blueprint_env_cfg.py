# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from ... import mdp
from . import stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
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
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        table_cam_normals = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "normals",
                "normalize": True,
                "save_image_to_file": True,
                "image_path": "_cosmos_inputs/table_cam",
            },
        )
        table_cam_segmentation = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_cam"),
                "data_type": "semantic_segmentation",
                "normalize": False,
                "save_image_to_file": True,
                "image_path": "_cosmos_inputs/table_cam",
            },
        )
        table_high_cam_normals = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_high_cam"),
                "data_type": "normals",
                "normalize": True,
                "save_image_to_file": True,
                "image_path": "_cosmos_inputs/table_high_cam",
            },
        )
        table_high_cam_segmentation = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("table_high_cam"),
                "data_type": "semantic_segmentation",
                "normalize": False,
                "save_image_to_file": True,
                "image_path": "_cosmos_inputs/table_high_cam",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()

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
        stack_1 = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()

@configclass
class FrankaCubeStackBlueprintEnvCfg(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )

        MAPPING = {
            "class:cube_1": (255, 36, 66, 255),
            "class:cube_2": (255, 184, 48, 255),
            "class:cube_3": (55, 255, 139, 255),
            "class:table": (255, 237, 218, 255),
            "class:ground": (100, 100, 100, 255),
            "class:robot": (125, 125, 125, 255),
            "class:UNLABELLED": (125, 125, 125, 255),
            "class:BACKGROUND": (10, 10, 10, 255),
        }

        # Set table view camera
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0333,
            height=704,
            width=1280,
            data_types=["rgb", "semantic_segmentation", "normals"],
            colorize_semantic_segmentation=True,
            semantic_segmentation_mapping=MAPPING,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(1.0, 0.0, 0.33), rot=(-0.3799, 0.5963, 0.5963, -0.3799), convention="ros"),
        )

        # Set table view camera
        self.scene.table_high_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_high_cam",
            update_period=0.0333,
            height=704,
            width=1280,
            data_types=["rgb", "semantic_segmentation", "normals"],
            colorize_semantic_segmentation=True,
            semantic_segmentation_mapping=MAPPING,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1.5, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(1.4, 1.8, 1.2), rot=(-0.1393, 0.2025, 0.8185, -0.5192), convention="ros"),
        )
