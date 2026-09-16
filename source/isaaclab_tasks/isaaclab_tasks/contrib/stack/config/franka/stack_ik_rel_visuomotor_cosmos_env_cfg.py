# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import config_field

from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.utils.presets import set_isaac_rtx_global_settings

from . import stack_ik_rel_visuomotor_env_cfg


@dataclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions: Any = config_field(ObsTerm(func=mdp.last_action))
        joint_pos: Any = config_field(ObsTerm(func=mdp.joint_pos_rel))
        joint_vel: Any = config_field(ObsTerm(func=mdp.joint_vel_rel))
        object: Any = config_field(ObsTerm(func=mdp.object_obs))
        cube_positions: Any = config_field(ObsTerm(func=mdp.cube_positions_in_world_frame))
        cube_orientations: Any = config_field(ObsTerm(func=mdp.cube_orientations_in_world_frame))
        eef_pos: Any = config_field(ObsTerm(func=mdp.ee_frame_pos))
        eef_quat: Any = config_field(ObsTerm(func=mdp.ee_frame_quat))
        gripper_pos: Any = config_field(ObsTerm(func=mdp.gripper_pos))
        table_cam: Any = config_field(
            ObsTerm(
                func=mdp.image,
                params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
            )
        )
        wrist_cam: Any = config_field(
            ObsTerm(
                func=mdp.image,
                params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
            )
        )
        table_cam_segmentation: Any = config_field(
            ObsTerm(
                func=mdp.image,
                params={
                    "sensor_cfg": SceneEntityCfg("table_cam"),
                    "data_type": "semantic_segmentation",
                    "normalize": True,
                },
            )
        )
        table_cam_normals: Any = config_field(
            ObsTerm(
                func=mdp.image,
                params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "normals", "normalize": True},
            )
        )
        table_cam_depth: Any = config_field(
            ObsTerm(
                func=mdp.image,
                params={
                    "sensor_cfg": SceneEntityCfg("table_cam"),
                    "data_type": "distance_to_image_plane",
                    "normalize": True,
                },
            )
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @dataclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1: Any = config_field(
            ObsTerm(
                func=mdp.object_grasped,
                params={
                    "robot_cfg": SceneEntityCfg("robot"),
                    "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                    "object_cfg": SceneEntityCfg("cube_2"),
                },
            )
        )
        stack_1: Any = config_field(
            ObsTerm(
                func=mdp.object_stacked,
                params={
                    "robot_cfg": SceneEntityCfg("robot"),
                    "upper_object_cfg": SceneEntityCfg("cube_2"),
                    "lower_object_cfg": SceneEntityCfg("cube_1"),
                },
            )
        )
        grasp_2: Any = config_field(
            ObsTerm(
                func=mdp.object_grasped,
                params={
                    "robot_cfg": SceneEntityCfg("robot"),
                    "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                    "object_cfg": SceneEntityCfg("cube_3"),
                },
            )
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = config_field(PolicyCfg())
    subtask_terms: SubtaskCfg = config_field(SubtaskCfg())


@dataclass
class FrankaCubeStackVisuomotorCosmosEnvCfg(stack_ik_rel_visuomotor_env_cfg.FrankaCubeStackVisuomotorEnvCfg):
    observations: ObservationsCfg = config_field(ObservationsCfg())

    def __post_init__(self):
        # post init of parent
        if parent_post_init := getattr(super(), "__post_init__", None):
            parent_post_init()

        SEMANTIC_MAPPING = {
            "class:cube_1": (120, 230, 255, 255),
            "class:cube_2": (255, 36, 66, 255),
            "class:cube_3": (55, 255, 139, 255),
            "class:table": (255, 237, 218, 255),
            "class:ground": (100, 100, 100, 255),
            "class:robot": (204, 110, 248, 255),
            "class:UNLABELLED": (150, 150, 150, 255),
            "class:BACKGROUND": (200, 200, 200, 255),
        }

        # Set cameras
        # Set wrist camera
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15), rot=(0.03701, 0.03701, -0.70614, -0.70614), convention="ros"
            ),
        )

        # Set table view camera
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "semantic_segmentation", "normals", "distance_to_image_plane"],
            colorize_semantic_segmentation=True,
            semantic_segmentation_mapping=SEMANTIC_MAPPING,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.4), rot=(-0.61237, -0.61237, 0.35355, 0.35355), convention="ros"
            ),
        )

        # Set settings for camera rendering
        self.num_rerenders_on_reset = 1
        for camera_cfg in (self.scene.table_cam, self.scene.wrist_cam):
            set_isaac_rtx_global_settings(
                camera_cfg.renderer_cfg,
                dome_light_upper_lower_strategy=4,
                antialiasing_mode="Off",
            )

        # List of image observations in policy observations
        self.image_obs_list = ["table_cam", "wrist_cam"]
