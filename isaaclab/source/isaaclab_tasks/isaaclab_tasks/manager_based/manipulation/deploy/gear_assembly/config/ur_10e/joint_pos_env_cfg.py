# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.deploy.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.deploy.mdp.events as gear_assembly_events
from isaaclab_tasks.manager_based.manipulation.deploy.gear_assembly.gear_assembly_env_cfg import GearAssemblyEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_GRIPPER_CFG, UR10e_ROBOTIQ_2F_85_CFG  # isort: skip


##
# Gripper-specific helper functions
##


def set_finger_joint_pos_robotiq_2f140(
    joint_pos: torch.Tensor,
    reset_ind_joint_pos: list[int],
    finger_joints: list[int],
    finger_joint_position: float,
):
    """Set finger joint positions for Robotiq 2F-140 gripper.

    Args:
        joint_pos: Joint positions tensor
        reset_ind_joint_pos: Row indices into the sliced joint_pos tensor
        finger_joints: List of finger joint indices
        finger_joint_position: Target position for finger joints
    """
    for idx in reset_ind_joint_pos:
        # For 2F-140 gripper (8 joints expected)
        # Joint structure: [finger_joint, finger_joint, outer_joints x2, inner_finger_joints x2, pad_joints x2]
        if len(finger_joints) < 8:
            raise ValueError(f"2F-140 gripper requires at least 8 finger joints, got {len(finger_joints)}")

        joint_pos[idx, finger_joints[0]] = finger_joint_position
        joint_pos[idx, finger_joints[1]] = finger_joint_position

        # outer finger joints set to 0
        joint_pos[idx, finger_joints[2]] = 0
        joint_pos[idx, finger_joints[3]] = 0

        # inner finger joints: multiply by -1
        joint_pos[idx, finger_joints[4]] = -finger_joint_position
        joint_pos[idx, finger_joints[5]] = -finger_joint_position

        joint_pos[idx, finger_joints[6]] = finger_joint_position
        joint_pos[idx, finger_joints[7]] = finger_joint_position


def set_finger_joint_pos_robotiq_2f85(
    joint_pos: torch.Tensor,
    reset_ind_joint_pos: list[int],
    finger_joints: list[int],
    finger_joint_position: float,
):
    """Set finger joint positions for Robotiq 2F-85 gripper.

    Args:
        joint_pos: Joint positions tensor
        reset_ind_joint_pos: Row indices into the sliced joint_pos tensor
        finger_joints: List of finger joint indices
        finger_joint_position: Target position for finger joints
    """
    for idx in reset_ind_joint_pos:
        # For 2F-85 gripper (6 joints expected)
        # Joint structure: [finger_joint, finger_joint, inner_finger_joints x2, inner_finger_knuckle_joints x2]
        if len(finger_joints) < 6:
            raise ValueError(f"2F-85 gripper requires at least 6 finger joints, got {len(finger_joints)}")

        # Multiply specific indices by -1: [2, 4, 5]
        # These correspond to:
        # ['left_inner_finger_joint', 'right_inner_finger_knuckle_joint', 'left_inner_finger_knuckle_joint']
        joint_pos[idx, finger_joints[0]] = finger_joint_position
        joint_pos[idx, finger_joints[1]] = finger_joint_position
        joint_pos[idx, finger_joints[2]] = -finger_joint_position
        joint_pos[idx, finger_joints[3]] = finger_joint_position
        joint_pos[idx, finger_joints[4]] = -finger_joint_position
        joint_pos[idx, finger_joints[5]] = -finger_joint_position


##
# Environment configuration
##


@configclass
class EventCfg:
    """Configuration for events."""

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"]
            ),  # only the arm joints are randomized
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"]),
            "friction_distribution_params": (0.3, 0.7),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    small_gear_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_gear_small", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
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

    large_gear_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("factory_gear_large", body_names=".*"),
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
            "asset_cfg": SceneEntityCfg("robot", body_names=".*finger"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    randomize_gear_type = EventTerm(
        func=gear_assembly_events.randomize_gear_type,
        mode="reset",
        params={"gear_types": ["gear_small", "gear_medium", "gear_large"]},
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    randomize_gears_and_base_pose = EventTerm(
        func=gear_assembly_events.randomize_gears_and_base_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.1, 0.1],
                "y": [-0.25, 0.25],
                "z": [-0.1, 0.1],
                "roll": [-math.pi / 90, math.pi / 90],  # 2 degree
                "pitch": [-math.pi / 90, math.pi / 90],  # 2 degree
                "yaw": [-math.pi / 6, math.pi / 6],  # 2 degree
            },
            "gear_pos_range": {
                "x": [-0.02, 0.02],
                "y": [-0.02, 0.02],
                "z": [0.0575, 0.0775],  # 0.045 + 0.0225
            },
            "velocity_range": {},
        },
    )

    set_robot_to_grasp_pose = EventTerm(
        func=gear_assembly_events.set_robot_to_grasp_pose,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "pos_randomization_range": {"x": [-0.0, 0.0], "y": [-0.005, 0.005], "z": [-0.003, 0.003]},
        },
    )


@configclass
class UR10eGearAssemblyEnvCfg(GearAssemblyEnvCfg):
    """Base configuration for UR10e Gear Assembly Environment.

    This class contains common setup shared across different gripper configurations.
    Subclasses should configure gripper-specific parameters.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Robot-specific parameters (can be overridden for other robots)
        self.end_effector_body_name = "wrist_3_link"  # End effector body name for IK and termination checks
        self.num_arm_joints = 6  # Number of arm joints (excluding gripper)
        self.grasp_rot_offset = [
            0.0,
            math.sqrt(2) / 2,
            math.sqrt(2) / 2,
            0.0,
        ]  # Rotation offset for grasp pose (quaternion [w, x, y, z])
        self.gripper_joint_setter_func = None  # Gripper-specific joint setter function (set in subclass)

        # Gear orientation termination thresholds (in degrees)
        self.gear_orientation_roll_threshold_deg = 15.0  # Maximum allowed roll deviation
        self.gear_orientation_pitch_threshold_deg = 15.0  # Maximum allowed pitch deviation
        self.gear_orientation_yaw_threshold_deg = 180.0  # Maximum allowed yaw deviation

        # Common observation configuration
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # override events
        self.events = EventCfg()

        # Update termination thresholds from config
        self.terminations.gear_orientation_exceeded.params["roll_threshold_deg"] = (
            self.gear_orientation_roll_threshold_deg
        )
        self.terminations.gear_orientation_exceeded.params["pitch_threshold_deg"] = (
            self.gear_orientation_pitch_threshold_deg
        )
        self.terminations.gear_orientation_exceeded.params["yaw_threshold_deg"] = (
            self.gear_orientation_yaw_threshold_deg
        )

        # override command generator body
        self.joint_action_scale = 0.025
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            scale=self.joint_action_scale,
            use_zero_offset=True,
        )


@configclass
class UR10e2F140GearAssemblyEnvCfg(UR10eGearAssemblyEnvCfg):
    """Configuration for UR10e with Robotiq 2F-140 gripper."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur10e with 2F-140 gripper
        self.scene.robot = UR10e_ROBOTIQ_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UR10e_ROBOTIQ_GRIPPER_CFG.spawn.replace(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            # Joint positions based on IK from center of distribution for randomized gear positions
            # This is done so that the start for the differential IK search after randomizing
            # is close to the optimal grasp pose
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "shoulder_pan_joint": 2.7228,
                    "shoulder_lift_joint": -8.3962e-01,
                    "elbow_joint": 1.3684,
                    "wrist_1_joint": -2.1048,
                    "wrist_2_joint": -1.5691,
                    "wrist_3_joint": -1.9896,
                },
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # 2F-140 gripper actuator configuration
        self.scene.robot.actuators["gripper_finger"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=10.0,
            damping=0.05,
            friction=0.0,
            armature=0.0,
        )

        # Set gripper-specific joint setter function
        self.gripper_joint_setter_func = set_finger_joint_pos_robotiq_2f140

        # gear offsets and grasp positions for the 2F-140 gripper
        self.gear_offsets_grasp = {
            "gear_small": [0.0, self.gear_offsets["gear_small"][0], -0.26],
            "gear_medium": [0.0, self.gear_offsets["gear_medium"][0], -0.26],
            "gear_large": [0.0, self.gear_offsets["gear_large"][0], -0.26],
        }

        # Grasp widths for 2F-140 gripper
        self.hand_grasp_width = {"gear_small": 0.64, "gear_medium": 0.54, "gear_large": 0.51}

        # Close widths for 2F-140 gripper
        self.hand_close_width = {"gear_small": 0.69, "gear_medium": 0.59, "gear_large": 0.56}

        # Populate event term parameters
        self.events.set_robot_to_grasp_pose.params["gear_offsets_grasp"] = self.gear_offsets_grasp
        self.events.set_robot_to_grasp_pose.params["end_effector_body_name"] = self.end_effector_body_name
        self.events.set_robot_to_grasp_pose.params["num_arm_joints"] = self.num_arm_joints
        self.events.set_robot_to_grasp_pose.params["grasp_rot_offset"] = self.grasp_rot_offset
        self.events.set_robot_to_grasp_pose.params["gripper_joint_setter_func"] = self.gripper_joint_setter_func

        # Populate termination term parameters
        self.terminations.gear_dropped.params["gear_offsets_grasp"] = self.gear_offsets_grasp
        self.terminations.gear_dropped.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.gear_dropped.params["grasp_rot_offset"] = self.grasp_rot_offset

        self.terminations.gear_orientation_exceeded.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.gear_orientation_exceeded.params["grasp_rot_offset"] = self.grasp_rot_offset


@configclass
class UR10e2F85GearAssemblyEnvCfg(UR10eGearAssemblyEnvCfg):
    """Configuration for UR10e with Robotiq 2F-85 gripper."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur10e with 2F-85 gripper
        self.scene.robot = UR10e_ROBOTIQ_2F_85_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UR10e_ROBOTIQ_2F_85_CFG.spawn.replace(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            # Joint positions based on IK from center of distribution for randomized gear positions
            # This is done so that the start for the differential IK search after randomizing
            # is close to the optimal grasp pose
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "shoulder_pan_joint": 2.7228,
                    "shoulder_lift_joint": -8.3962e-01,
                    "elbow_joint": 1.3684,
                    "wrist_1_joint": -2.1048,
                    "wrist_2_joint": -1.5691,
                    "wrist_3_joint": -1.9896,
                },
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # 2F-85 gripper actuator configuration (higher effort limits than 2F-140)
        self.scene.robot.actuators["gripper_finger"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=10.0,
            damping=0.05,
            friction=0.0,
            armature=0.0,
        )
        self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
            armature=0.0,
        )

        # Set gripper-specific joint setter function
        self.gripper_joint_setter_func = set_finger_joint_pos_robotiq_2f85

        # gear offsets and grasp positions for the 2F-85 gripper
        self.gear_offsets_grasp = {
            "gear_small": [0.0, self.gear_offsets["gear_small"][0], -0.19],
            "gear_medium": [0.0, self.gear_offsets["gear_medium"][0], -0.19],
            "gear_large": [0.0, self.gear_offsets["gear_large"][0], -0.19],
        }

        # Grasp widths for 2F-85 gripper
        self.hand_grasp_width = {"gear_small": 0.64, "gear_medium": 0.46, "gear_large": 0.4}

        # Close widths for 2F-85 gripper
        self.hand_close_width = {"gear_small": 0.69, "gear_medium": 0.51, "gear_large": 0.45}

        # Populate event term parameters
        self.events.set_robot_to_grasp_pose.params["gear_offsets_grasp"] = self.gear_offsets_grasp
        self.events.set_robot_to_grasp_pose.params["end_effector_body_name"] = self.end_effector_body_name
        self.events.set_robot_to_grasp_pose.params["num_arm_joints"] = self.num_arm_joints
        self.events.set_robot_to_grasp_pose.params["grasp_rot_offset"] = self.grasp_rot_offset
        self.events.set_robot_to_grasp_pose.params["gripper_joint_setter_func"] = self.gripper_joint_setter_func

        # Populate termination term parameters
        self.terminations.gear_dropped.params["gear_offsets_grasp"] = self.gear_offsets_grasp
        self.terminations.gear_dropped.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.gear_dropped.params["grasp_rot_offset"] = self.grasp_rot_offset

        self.terminations.gear_orientation_exceeded.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.gear_orientation_exceeded.params["grasp_rot_offset"] = self.grasp_rot_offset


@configclass
class UR10e2F140GearAssemblyEnvCfg_PLAY(UR10e2F140GearAssemblyEnvCfg):
    """Play configuration for UR10e with Robotiq 2F-140 gripper."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class UR10e2F85GearAssemblyEnvCfg_PLAY(UR10e2F85GearAssemblyEnvCfg):
    """Play configuration for UR10e with Robotiq 2F-85 gripper."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
