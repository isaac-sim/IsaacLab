# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import replace_config

from ... import mdp
from . import stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from typing import Any


@dataclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        actions: Any = field(default_factory=lambda: ObsTerm(func=mdp.last_action))
        joint_pos: Any = field(default_factory=lambda: ObsTerm(func=mdp.joint_pos_rel))
        joint_vel: Any = field(default_factory=lambda: ObsTerm(func=mdp.joint_vel_rel))
        object: Any = field(default_factory=lambda: ObsTerm(func=mdp.object_obs))
        cube_positions: Any = field(default_factory=lambda: ObsTerm(func=mdp.cube_positions_in_world_frame))
        cube_orientations: Any = field(default_factory=lambda: ObsTerm(func=mdp.cube_orientations_in_world_frame))
        eef_pos: Any = field(default_factory=lambda: ObsTerm(func=mdp.ee_frame_pos))
        eef_quat: Any = field(default_factory=lambda: ObsTerm(func=mdp.ee_frame_quat))
        gripper_pos: Any = field(default_factory=lambda: ObsTerm(func=mdp.gripper_pos))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @dataclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @dataclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1: Any = field(
            default_factory=lambda: ObsTerm(
                func=mdp.object_grasped,
                params={
                    "robot_cfg": SceneEntityCfg("robot"),
                    "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                    "object_cfg": SceneEntityCfg("cube_2"),
                },
            )
        )
        stack_1: Any = field(
            default_factory=lambda: ObsTerm(
                func=mdp.object_stacked,
                params={
                    "robot_cfg": SceneEntityCfg("robot"),
                    "upper_object_cfg": SceneEntityCfg("cube_2"),
                    "lower_object_cfg": SceneEntityCfg("cube_1"),
                },
            )
        )
        grasp_2: Any = field(
            default_factory=lambda: ObsTerm(
                func=mdp.object_grasped,
                params={
                    "robot_cfg": SceneEntityCfg("robot"),
                    "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                    "object_cfg": SceneEntityCfg("cube_3"),
                },
            )
        )
        stack_2: Any = field(
            default_factory=lambda: ObsTerm(
                func=mdp.object_stacked,
                params={
                    "robot_cfg": SceneEntityCfg("robot"),
                    "upper_object_cfg": SceneEntityCfg("cube_3"),
                    "lower_object_cfg": SceneEntityCfg("cube_2"),
                },
            )
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    rgb_camera: RGBCameraPolicyCfg = field(default_factory=RGBCameraPolicyCfg)
    subtask_terms: SubtaskCfg = field(default_factory=SubtaskCfg)


@dataclass
class FrankaCubeStackSkillgenEnvCfg(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        if parent_post_init := getattr(super(), "__post_init__", None):
            parent_post_init()

        # Override observations with SkillGen-specific config
        self.observations = ObservationsCfg()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = replace_config(FRANKA_PANDA_HIGH_PD_CFG, prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )

        # Apply skillgen-specific cube position randomization
        self.events.randomize_cube_positions.params["pose_range"] = {
            "x": (0.45, 0.6),
            "y": (-0.23, 0.23),
            "z": (0.0203, 0.0203),
            "yaw": (-1.0, 1, 0),
        }

        # Set the offset for the end effector to be 0.0
        for f in self.scene.ee_frame.target_frames:
            if f.name == "end_effector":
                f.offset.pos = [0.0, 0.0, 0.0]
                break
