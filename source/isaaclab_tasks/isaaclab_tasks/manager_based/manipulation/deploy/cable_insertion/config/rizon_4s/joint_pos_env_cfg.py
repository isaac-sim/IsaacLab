# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.deploy.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.deploy.mdp.terminations as cable_terminations
from isaaclab_tasks.manager_based.manipulation.deploy.cable_insertion.cable_insertion_env_cfg import (
    CableInsertionEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets import FLEXIV_RIZON4S_GRAV_GRIPPER_CFG  # isort: skip


##
# Gripper-specific helper functions
##


def set_finger_joint_pos_grav(
    joint_pos: torch.Tensor,
    reset_ind_joint_pos: list[int],
    finger_joints: list[int],
    finger_joint_position: float,
):
    """Set finger joint positions for the Grav gripper.

    Args:
        joint_pos: Joint positions tensor for the reset envs slice.
        reset_ind_joint_pos: Row indices into the sliced ``joint_pos`` tensor.
        finger_joints: List of all gripper joint indices (6 joints total).
        finger_joint_position: Target position for the main finger joint [rad].

    Note:
        Grav gripper joint structure (indices from ``finger_joints`` list):
            ``[0]`` ``finger_joint`` - main controllable joint
            ``[1]`` ``left_inner_knuckle_joint`` - mimic with -1 gearing
            ``[2]`` ``right_inner_knuckle_joint`` - mimic with -1 gearing
            ``[3]`` ``right_outer_knuckle_joint`` - mimic with -1 gearing
            ``[4]`` ``left_outer_finger_joint`` - mimic with +1 gearing
            ``[5]`` ``right_outer_finger_joint`` - mimic with +1 gearing
    """
    for idx in reset_ind_joint_pos:
        if len(finger_joints) < 6:
            raise ValueError(f"Grav gripper requires at least 6 finger joints, got {len(finger_joints)}")

        joint_pos[idx, finger_joints[0]] = finger_joint_position

        joint_pos[idx, finger_joints[1]] = finger_joint_position
        joint_pos[idx, finger_joints[2]] = finger_joint_position
        joint_pos[idx, finger_joints[3]] = finger_joint_position

        joint_pos[idx, finger_joints[4]] = -finger_joint_position
        joint_pos[idx, finger_joints[5]] = -finger_joint_position


##
# Environment configuration
##


@configclass
class EventCfg:
    """Configuration for events."""

    plug_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("gb300_plug", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    socket_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("gb300_socket", body_names=".*"),
            "static_friction_range": (0.0, 0.0),
            "dynamic_friction_range": (0.0, 0.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*finger.*"),
            "static_friction_range": (3.0, 3.0),
            "dynamic_friction_range": (3.0, 3.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    randomize_socket_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [0.0, 0.0],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("gb300_socket"),
        },
    )

    set_robot_to_grasp_pose = EventTerm(
        func=mdp.set_robot_to_object_grasp_pose,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "pos_randomization_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "target_object_name": "gb300_plug",
            "grasp_offset": [0.0, 0.0, 0.0],
        },
    )


@configclass
class TerminationsCfg:
    """Configuration for termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    plug_dropped = DoneTerm(
        func=cable_terminations.reset_when_plug_dropped,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "plug_asset_cfg": SceneEntityCfg("gb300_plug"),
            "distance_threshold": 0.15,
            "end_effector_body_name": "link7",
            "grasp_offset": [0.0, 0.0, -0.35],
            "grasp_rot_offset": [-0.70711, 0.70711, 0.0, 0.0],
        },
    )

    plug_orientation_exceeded = DoneTerm(
        func=cable_terminations.reset_when_plug_orientation_exceeded,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "plug_asset_cfg": SceneEntityCfg("gb300_plug"),
            "roll_threshold_deg": 15.0,
            "pitch_threshold_deg": 15.0,
            "yaw_threshold_deg": 180.0,
            "end_effector_body_name": "link7",
            "grasp_rot_offset": [-0.70711, 0.70711, 0.0, 0.0],
        },
    )


@configclass
class Rizon4sGravCableInsertionEnvCfg(CableInsertionEnvCfg):
    """Configuration for Flexiv Rizon 4s with Grav Gripper Cable Insertion Environment."""

    def __post_init__(self):
        super().__post_init__()

        self.end_effector_body_name = "link7"
        self.num_arm_joints = 7
        self.grasp_offset = [0.0, 0.0, -0.35]
        self.grasp_rot_offset = [-0.70711, 0.70711, 0.0, 0.0]
        self.gripper_joint_setter_func = set_finger_joint_pos_grav

        self.plug_orientation_roll_threshold_deg = 15.0
        self.plug_orientation_pitch_threshold_deg = 15.0
        self.plug_orientation_yaw_threshold_deg = 180.0

        # Observation configuration for Rizon 4s arm joints only
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        self.events = EventCfg()

        self.terminations = TerminationsCfg()

        self.terminations.plug_orientation_exceeded.params["roll_threshold_deg"] = (
            self.plug_orientation_roll_threshold_deg
        )
        self.terminations.plug_orientation_exceeded.params["pitch_threshold_deg"] = (
            self.plug_orientation_pitch_threshold_deg
        )
        self.terminations.plug_orientation_exceeded.params["yaw_threshold_deg"] = (
            self.plug_orientation_yaw_threshold_deg
        )

        self.joint_action_scale = 0.025
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
            scale=self.joint_action_scale,
            use_zero_offset=True,
        )

        self.scene.robot = FLEXIV_RIZON4S_GRAV_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=FLEXIV_RIZON4S_GRAV_GRIPPER_CFG.spawn.replace(
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
                    enabled_self_collisions=False,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=1,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "joint1": 0.0,
                    "joint2": -0.698,
                    "joint3": 0.0,
                    "joint4": 1.571,
                    "joint5": 0.0,
                    "joint6": 0.698,
                    "joint7": 0.0,
                },
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )

        # Grav gripper actuator configuration (implicit - PhysX built-in joint drives)
        self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=2e3,
            damping=1e1,
        )

        self.scene.robot.actuators["gripper_passive"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_knuckle_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
        )

        # Override plug/socket positions to match the gear assembly Flexiv workspace layout.
        # Socket = gear_base (fixed target), Plug = gear (held object, offset above socket).
        self.scene.gb300_socket.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(0.481, -0.073, 0.071),
            rot=(0.0, 0.0, 0.70711, -0.70711),
        )
        self.scene.gb300_plug.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(0.481, -0.073, 0.139),
            rot=(0.0, 0.0, 0.70711, -0.70711),
        )

        # Grasp widths for the Grav gripper holding the plug. Positive values open
        # the gripper; ``hand_grasp_width`` creates a gap for the plug, while
        # ``hand_close_width`` squeezes the fingers shut to hold it.
        self.hand_grasp_width = 0.05
        self.hand_close_width = 0.0

        self.events.set_robot_to_grasp_pose.params["end_effector_body_name"] = self.end_effector_body_name
        self.events.set_robot_to_grasp_pose.params["num_arm_joints"] = self.num_arm_joints
        self.events.set_robot_to_grasp_pose.params["grasp_rot_offset"] = self.grasp_rot_offset
        self.events.set_robot_to_grasp_pose.params["grasp_offset"] = self.grasp_offset
        self.events.set_robot_to_grasp_pose.params["gripper_joint_setter_func"] = self.gripper_joint_setter_func

        self.terminations.plug_dropped.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.plug_dropped.params["grasp_offset"] = self.grasp_offset
        self.terminations.plug_dropped.params["grasp_rot_offset"] = self.grasp_rot_offset

        self.terminations.plug_orientation_exceeded.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.plug_orientation_exceeded.params["grasp_rot_offset"] = self.grasp_rot_offset


@configclass
class Rizon4sGravCableInsertionEnvCfg_PLAY(Rizon4sGravCableInsertionEnvCfg):
    """Play configuration for Flexiv Rizon 4s cable insertion."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
