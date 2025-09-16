# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
import torch

from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg


import isaaclab_tasks.manager_based.manipulation.deploy.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.deploy.gear_assembly.gear_assembly_env_cfg import GearAssemblyEnvCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

import isaaclab_tasks.manager_based.manipulation.deploy.mdp.events as gear_assembly_events


##
# Pre-defined configs
##
from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_GRIPPER_CFG  # isort: skip


##
# Environment configuration
##

@configclass
class EventCfg:
    """Configuration for events."""

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        # min_step_count_between_reset=720,
        # min_step_count_between_reset=200,
        mode="reset",
        params={
            # "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"]),  # only the arm joints are randomized
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            # "stiffness_distribution_params": (1.0, 1.0),
            # "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        # min_step_count_between_reset=720,
        # min_step_count_between_reset=200,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"]),
            "friction_distribution_params": (0.3, 0.7),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    medium_gear_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_gear_medium", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )
    gear_base_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_gear_base", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    randomize_gear_type = EventTerm(
        func=gear_assembly_events.randomize_gear_type,
        mode="reset",
        params={
            # "gear_types": ["gear_small", "gear_medium", "gear_large"]
            "gear_types": ["gear_medium"]
        },
    )


    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


    randomize_gears_and_base_pose = EventTerm(
        func=gear_assembly_events.randomize_gears_and_base_pose,
        mode="reset",
        params={
            "pose_range": {
                # "x": [0.0, 0.0],
                # "y": [0.0, 0.0],
                # "z": [0.0, 0.0],
                # "roll": [-0.0, 0.0], # 0 degree
                # "pitch": [-0.0, 0.0], # 0 degree
                # "yaw": [-0.0, 0.0], # 0 degree
                "x": [-0.1, 0.1],
                "y": [-0.25, 0.25],
                "z": [-0.1,  0.1],
                "roll": [-math.pi/90, math.pi/90], # 2 degree
                "pitch": [-math.pi/90, math.pi/90], # 2 degree
                "yaw": [-math.pi/6, math.pi/6], # 2 degree
            },
            "gear_pos_range": {
                "x": [-0.02, 0.02],
                "y": [-0.02, 0.02],
                "z": [0.0575, 0.0775], # 0.045 + 0.0225
                # "x": [-0.0, 0.0],
                # "y": [-0.0, 0.0],
                # "z": [0.0675, 0.0675],
            },
            "rot_randomization_range": {
                "roll": [-math.pi/36, math.pi/36], # 5 degree
                "pitch": [-math.pi/36, math.pi/36], # 5 degree
                "yaw": [-math.pi/36, math.pi/36], # 5 degree
            },
            "velocity_range": {},
        },
    )

    set_robot_to_grasp_pose = EventTerm(
        func=gear_assembly_events.set_robot_to_grasp_pose,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            # "pos_offset": [-0.030375, 0.0, -0.255],  # Offset from wrist_3_link to gripper position
            # "pos_offset": [0.0, 0.030375, -0.26],
            "rot_offset": [0.0, math.sqrt(2)/2, math.sqrt(2)/2, 0.0],
            "pos_randomization_range": {
                "x": [-0.0, 0.0],
                "y": [-0.005, 0.005],
                "z": [-0.003, 0.003]
            },
            # "pos_randomization_range": None
        },
    )




@configclass
class UR10eGearAssemblyEnvCfg(GearAssemblyEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur10 with local asset path
        self.scene.robot = UR10e_ROBOTIQ_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UR10e_ROBOTIQ_GRIPPER_CFG.spawn.replace(
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=192, solver_velocity_iteration_count=1
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "shoulder_pan_joint": 2.7228e+00,
                    "shoulder_lift_joint": -8.3962e-01,
                    "elbow_joint": 1.3684e+00,
                    "wrist_1_joint": -2.1048e+00,
                    "wrist_2_joint": -1.5691e+00,
                    "wrist_3_joint": -1.9896e+00,
                    "finger_joint": 0.0,
                },
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )


        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

        # override events
        self.events = EventCfg()

        # overriderride command generator body
        self.joint_action_scale = 0.025
        self.action_scale_joint_space = [self.joint_action_scale, self.joint_action_scale, self.joint_action_scale, self.joint_action_scale, self.joint_action_scale, self.joint_action_scale]
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], scale=self.joint_action_scale, use_zero_offset=True
        )

        self.obs_order = ["arm_dof_pos", "arm_dof_vel", "shaft_pos", "shaft_quat"]
        self.policy_action_space = "joint"
        self.arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.policy_action_space = "joint"
        self.action_space = 6
        self.state_space = 0
        self.observation_space = 19

        # Set joint_action_scale from the existing arm_action.scale
        self.joint_action_scale = self.actions.arm_action.scale

        self.action_scale_joint_space = [
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
            self.joint_action_scale,
        ]
        self.fixed_asset_init_pos_center = (1.0200, -0.2100, -0.1)
        self.fixed_asset_init_pos_range = (0.1, 0.25, 0.1)
        self.fixed_asset_init_orn_deg = (180.0, 0.0, 90.0)
        self.fixed_asset_init_orn_deg_range = (5.0, 5.0, 30.0)


@configclass
class UR10eGearAssemblyEnvCfg_PLAY(UR10eGearAssemblyEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
