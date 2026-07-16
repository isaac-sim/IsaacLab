# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-space DisplayPort insertion environment for Flexiv Rizon 4S + Grav gripper."""

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.deploy.mdp as mdp
import isaaclab_tasks.contrib.deploy.mdp.terminations as cable_terminations
from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import (
    PLUG_GOAL_ROT,
    PLUG_INSERTION_OFFSET,
    SOCKET_INSERTION_OFFSET,
    DisplayportInsertionEnvCfg,
    compute_plug_pose,
    compute_socket_root,
)

# DisplayPort insertion station layout in the Flexiv workspace.
# _GEOMETRY_POS is the desired insertion (mate) point; _SOCKET_ROT orients the socket opening up.
_GEOMETRY_POS = (0.475, 0.125, 0.06)
_SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)  # opening faces +Z (top-down insertion)
_PLUG_CLEARANCE_Z = 0.068  # vertical clearance between plug and socket at reset

_SOCKET_ROOT = compute_socket_root(_GEOMETRY_POS, _SOCKET_ROT)
_PLUG_ROOT, _PLUG_ROT = compute_plug_pose(
    _GEOMETRY_POS,
    _SOCKET_ROT,
    z_clearance=_PLUG_CLEARANCE_Z,
)

# Blade engagement along the insertion axis used by the at-goal curriculum [m].
_INSERTION_LENGTH = 0.011

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
    """Set finger joint positions for Grav gripper.

    Args:
        joint_pos: Joint positions tensor
        reset_ind_joint_pos: Row indices into the sliced joint_pos tensor
        finger_joints: List of all gripper joint indices (6 joints total)
        finger_joint_position: Target position for main finger joint (in radians)

    Note:
        Grav gripper joint structure (indices from finger_joints list):
        [0] finger_joint - main controllable joint
        [1] left_inner_knuckle_joint - mimic with -1 gearing
        [2] right_inner_knuckle_joint - mimic with -1 gearing
        [3] right_outer_knuckle_joint - mimic with -1 gearing
        [4] left_outer_finger_joint - mimic with +1 gearing
        [5] right_outer_finger_joint - mimic with +1 gearing
    """
    for idx in reset_ind_joint_pos:
        if len(finger_joints) < 6:
            raise ValueError(f"Grav gripper requires at least 6 finger joints, got {len(finger_joints)}")

        # Main controllable joint
        joint_pos[idx, finger_joints[0]] = finger_joint_position

        # Mimic joints with -1 gearing
        joint_pos[idx, finger_joints[1]] = finger_joint_position  # left_inner_knuckle_joint
        joint_pos[idx, finger_joints[2]] = finger_joint_position  # right_inner_knuckle_joint
        joint_pos[idx, finger_joints[3]] = finger_joint_position  # right_outer_knuckle_joint

        # Mimic joints with +1 gearing
        joint_pos[idx, finger_joints[4]] = -finger_joint_position  # left_outer_finger_joint
        joint_pos[idx, finger_joints[5]] = -finger_joint_position  # right_outer_finger_joint


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
            "asset_cfg": SceneEntityCfg("dp_plug", body_names=".*"),
            "static_friction_range": (0.001, 0.001),
            "dynamic_friction_range": (0.001, 0.001),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    socket_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("dp_socket", body_names=".*"),
            "static_friction_range": (0.001, 0.001),
            "dynamic_friction_range": (0.001, 0.001),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*finger.*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
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
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
                "z": [-0.02, 0.02],
                "roll": [-math.radians(2.0), math.radians(2.0)],  # 2 degree
                "pitch": [-math.radians(2.0), math.radians(2.0)],  # 2 degree
                "yaw": [-math.radians(2.0), math.radians(2.0)],  # 2 degree
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("dp_socket"),
        },
    )

    reset_plug_curriculum = EventTerm(
        func=mdp.reset_plug_at_goal_curriculum,
        mode="reset",
        params={
            "plug_cfg": SceneEntityCfg("dp_plug"),
            "socket_cfg": SceneEntityCfg("dp_socket"),
            "at_goal_prob": 0.8,
            "at_goal_prob_final": 0.0,
            "anneal_start_iter": 0.0,
            "anneal_end_iter": 500.0,
            "num_steps_per_env": 512,
            "insertion_axis": [1.0, 0.0, 0.0],
            "insertion_length": _INSERTION_LENGTH,
            "at_goal_depth_range": [0.0, 0.015],
            "approach_depth_range": [0.02, 0.06],
            "socket_insertion_offset": SOCKET_INSERTION_OFFSET,
            "plug_insertion_offset": PLUG_INSERTION_OFFSET,
            "goal_rot": list(PLUG_GOAL_ROT),
            "normal_pose_range": {
                "x": [-0.02, 0.02],
                "y": [-0.02, 0.02],
                "z": [0.0, 0.0],
            },
        },
    )

    set_robot_to_grasp_pose = EventTerm(
        func=mdp.set_robot_to_object_grasp_pose,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "pos_randomization_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "target_object_name": "dp_plug",
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
            "plug_asset_cfg": SceneEntityCfg("dp_plug"),
            "distance_threshold": 0.15,
            "end_effector_body_name": "link7",
            "grasp_offset": [0.0, 0.0, 0.0],
            "grasp_rot_offset": [0.0, 0.0, 0.0, 1.0],
        },
    )

    plug_orientation_exceeded = DoneTerm(
        func=cable_terminations.reset_when_plug_orientation_exceeded,
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "plug_asset_cfg": SceneEntityCfg("dp_plug"),
            "roll_threshold_deg": 15.0,
            "pitch_threshold_deg": 15.0,
            "yaw_threshold_deg": 180.0,
            "end_effector_body_name": "link7",
            "grasp_rot_offset": [0.0, 0.0, 0.0, 1.0],
        },
    )


@configclass
class Rizon4sGravDisplayportInsertionEnvCfg(DisplayportInsertionEnvCfg):
    """Configuration for Flexiv Rizon 4s with Grav Gripper DisplayPort insertion.

    The Flexiv Rizon 4s is a 7-DOF collaborative robot arm equipped with the
    Flexiv Grav parallel gripper for DisplayPort plug insertion tasks.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Match exponential keypoint reward weight to the linear term (1:1 weighting)
        self.rewards.plug_socket_keypoint_tracking_exp.weight = abs(self.rewards.plug_socket_keypoint_tracking.weight)

        # Robot-specific parameters for Flexiv Rizon 4s with Grav gripper
        self.end_effector_body_name = "flange"  # End effector body name for IK
        self.num_arm_joints = 7  # Number of arm joints (Rizon 4s has 7 DOF)
        # Grasp offset in the DisplayPort plug's local frame [m]
        self.grasp_offset = [0.0025, 0.0, -0.1875]
        # Rotation offset for grasp pose (quaternion [x, y, z, w])
        self.grasp_rot_offset = [0.0, 0.0, 0.0, 1.0]
        self.gripper_joint_setter_func = set_finger_joint_pos_grav  # Grav gripper joint setter function

        # Plug orientation termination thresholds (in degrees)
        self.plug_orientation_roll_threshold_deg = 15.0  # Maximum allowed roll deviation
        self.plug_orientation_pitch_threshold_deg = 15.0  # Maximum allowed pitch deviation
        self.plug_orientation_yaw_threshold_deg = 180.0  # Maximum allowed yaw deviation

        # Common observation configuration for Rizon 4s joints (arm only, not gripper)
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

        # override events
        self.events = EventCfg()

        self.terminations = TerminationsCfg()

        # Update termination thresholds from config
        self.terminations.plug_orientation_exceeded.params["roll_threshold_deg"] = (
            self.plug_orientation_roll_threshold_deg
        )
        self.terminations.plug_orientation_exceeded.params["pitch_threshold_deg"] = (
            self.plug_orientation_pitch_threshold_deg
        )
        self.terminations.plug_orientation_exceeded.params["yaw_threshold_deg"] = (
            self.plug_orientation_yaw_threshold_deg
        )

        # Action configuration for Rizon 4s arm
        self.joint_action_scale = 0.025
        _arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=_arm_joint_names,
            scale=self.joint_action_scale,
            use_zero_offset=True,
        )

        # Switch robot to Flexiv Rizon 4s with Grav gripper
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
            # Joint positions for the DisplayPort insertion station home pose
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "joint1": math.radians(32.44),
                    "joint2": math.radians(-16.71),
                    "joint3": math.radians(-5.69),
                    "joint4": math.radians(128.38),
                    "joint5": math.radians(6.74),
                    "joint6": math.radians(55.95),
                    "joint7": math.radians(111.54),
                },
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )

        # Grav gripper actuator configuration
        self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=2.0,
            velocity_limit_sim=1.0,
            stiffness=2e3,
            damping=1e1,
        )

        # Passive/mimic joints in the gripper - set to zero stiffness/damping
        self.scene.robot.actuators["gripper_passive"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_knuckle_joint"],
            effort_limit_sim=1.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
        )

        # Override socket/plug initial states for the DisplayPort insertion station
        self.scene.dp_socket.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_SOCKET_ROOT,
            rot=_SOCKET_ROT,
        )
        self.scene.dp_plug.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_PLUG_ROOT,
            rot=_PLUG_ROT,
        )

        # Grasp widths for Grav gripper (raw radian values for finger_joint)
        self.hand_grasp_width = 0.3
        self.hand_hold_width = -0.05
        self.hand_close_width = -0.155

        # Populate event term parameters
        self.events.set_robot_to_grasp_pose.params["end_effector_body_name"] = self.end_effector_body_name
        self.events.set_robot_to_grasp_pose.params["num_arm_joints"] = self.num_arm_joints
        self.events.set_robot_to_grasp_pose.params["grasp_rot_offset"] = self.grasp_rot_offset
        self.events.set_robot_to_grasp_pose.params["grasp_offset"] = self.grasp_offset
        self.events.set_robot_to_grasp_pose.params["gripper_joint_setter_func"] = self.gripper_joint_setter_func
        self.events.set_robot_to_grasp_pose.params["max_iterations"] = 150

        # Populate termination term parameters
        self.terminations.plug_dropped.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.plug_dropped.params["grasp_offset"] = self.grasp_offset
        self.terminations.plug_dropped.params["grasp_rot_offset"] = self.grasp_rot_offset

        self.terminations.plug_orientation_exceeded.params["end_effector_body_name"] = self.end_effector_body_name
        self.terminations.plug_orientation_exceeded.params["grasp_rot_offset"] = self.grasp_rot_offset


@configclass
class Rizon4sGravDisplayportInsertionEnvCfg_PLAY(Rizon4sGravDisplayportInsertionEnvCfg):
    """Play configuration for Flexiv Rizon 4s DisplayPort insertion."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class Rizon4sGravDisplayportInsertionNoJointVelEnvCfg(Rizon4sGravDisplayportInsertionEnvCfg):
    """DisplayPort insertion without joint velocity in the policy observation.

    The critic retains joint velocity as privileged information for the value function.
    """

    def __post_init__(self):
        super().__post_init__()
        # Remove joint velocity from the actor observation group
        self.observations.policy.joint_vel = None


@configclass
class Rizon4sGravDisplayportInsertionNoJointVelEnvCfg_PLAY(Rizon4sGravDisplayportInsertionNoJointVelEnvCfg):
    """Play configuration for the no-joint-velocity joint-space variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
