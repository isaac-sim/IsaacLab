# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-space DisplayPort insertion environment for Flexiv Rizon 4S + Grav gripper.

Mirrors the GB300 ``config/rizon_4s/joint_pos_env_cfg.py`` but builds on the
DisplayPort base env. Relative joint-position control of the 7-DoF arm; the
plug is grasped at reset and the goal is the verified seated mate.
"""

import math

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
from isaaclab_tasks.manager_based.manipulation.deploy.cable_insertion.displayport_insertion_env_cfg import (
    PLUG_GOAL_ROT,
    PLUG_INSERTION_OFFSET,
    SOCKET_INSERTION_OFFSET,
    DisplayportInsertionEnvCfg,
    compute_plug_pose,
    compute_socket_root,
)

# ---------------------------------------------------------------------------
# Flexiv workspace layout (DisplayPort insertion station)
# ---------------------------------------------------------------------------
# Insertion(mate) point in the robot workspace, socket orientation (opening up,
# matching the verified drop-test seated pose), and the vertical clearance the
# plug starts above the socket.
#
# CALIBRATE: _GEOMETRY_POS is seeded from the GB300 Hubble cable-insertion
# station and must be re-measured for the real DisplayPort fixture (reachable by
# the Flexiv, socket opening facing +Z). Verify the reset poses against the live
# sim (see scripts/dp_probe_pose_geometry.py) before training.
_GEOMETRY_POS = (0.475, 0.125, 0.06)
_SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)  # opening faces +Z (top-down insertion)
_PLUG_CLEARANCE_Z = 0.068

_SOCKET_ROOT = compute_socket_root(_GEOMETRY_POS, _SOCKET_ROT)
_PLUG_ROOT, _PLUG_ROT = compute_plug_pose(
    _GEOMETRY_POS, _SOCKET_ROT, z_clearance=_PLUG_CLEARANCE_Z,
)

# DisplayPort plug insertion length (blade engagement along the insertion axis)
# used by the at-goal curriculum. ~11 mm of blade nests in the socket cavity at
# the verified seated pose. Kept identical to the task-space env.
_INSERTION_LENGTH = 0.011

# ---------------------------------------------------------------------------
# Sim-to-real action model toggle (ported from IsaacLab_ashwin gear-assembly
# ``sysid_physx_env_cfg.py``)
# ---------------------------------------------------------------------------
# Master switch. Flip to ``True`` to enable the sim-to-real action pipeline
# (mirrors IsaacLab_ashwin ``Rizon4sGearAssemblyROSInferenceSysIDPhysXShapedEnvCfg``):
#   1. Sim rate       -> 200 Hz PhysX physics with decimation 4 => 50 Hz control,
#      matching the Flexiv deployment command loop.
#   2. Action latency -> ``ShapedDelayedRelativeJointPositionActionCfg`` delays
#      the applied joint target by ``USE_SIM2REAL_ACTION_LATENCY_S`` seconds,
#      approximating the real robot's command loop lag.
#   3. Command shaping -> per-step velocity / acceleration clamping of the arm
#      joint targets (matches the collection-time command limits).
#   4. SysID PD gains  -> replaces the stock arm PD (1320/600/216) with the
#      PhysX SysID-tuned per-joint stiffness/damping for the Flexiv Rizon 4S.
#
# Leave ``False`` to keep the current behavior exactly (plain
# ``RelativeJointPositionActionCfg`` + stock actuator PD gains + 240 Hz / dec 8).
# This single flag is the only thing you need to change to revert.
USE_SIM2REAL_ACTION_MODEL = True

# Sim rate for the sim-to-real deployment loop. 200 Hz PhysX physics with
# decimation 4 gives a 50 Hz effective control rate (== ashwin gear assembly).
# Uses PhysX implicit actuators (not Newton). At 50 Hz, a 20 ms latency maps to
# exactly one control step.
USE_SIM2REAL_PHYSICS_FREQ_HZ = 200.0
USE_SIM2REAL_DECIMATION = 4

# Command latency in seconds. 20 ms = one 50 Hz control sample in ashwin.
USE_SIM2REAL_ACTION_LATENCY_S = 0.02
# Command-target slew limits applied to the arm (rad/s and rad/s^2). Zero
# disables the respective limit. Deployment values from ashwin gear assembly.
USE_SIM2REAL_COMMAND_VELOCITY_LIMIT = 2.0
USE_SIM2REAL_COMMAND_ACCELERATION_LIMIT = 3.0

# PhysX SysID-tuned per-joint PD gains for the Flexiv Rizon 4S arm. Values from
# IsaacLab_ashwin ``input/actuator_models/flexiv/manual/
# flexiv_pd_only_gravityoff_high_cmdlimits_tuned_physx.yaml``.
_SIM2REAL_ARM_STIFFNESS = {
    "joint1": 6051.500977,
    "joint2": 3004.328857,
    "joint3": 7032.64209,
    "joint4": 4346.786133,
    "joint5": 6829.847656,
    "joint6": 3769.23291,
    "joint7": 6181.429199,
}
_SIM2REAL_ARM_DAMPING = {
    "joint1": 121.620995,
    "joint2": 220.929214,
    "joint3": 162.582214,
    "joint4": 186.334442,
    "joint5": 199.962311,
    "joint6": 118.462029,
    "joint7": 152.861771,
}

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
                # "x": [-0.00, 0.00],
                # "y": [-0.00, 0.00],
                # "z": [-0.00, 0.00],
                # "roll": [-math.radians(2.0), math.radians(2.0)],
                # "pitch": [-math.radians(2.0), math.radians(2.0)],
                # "yaw": [-math.radians(2.0), math.radians(2.0)],
                "roll": [0., 0.],
                "pitch": [0., 0.],
                "yaw": [0., 0.],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("dp_socket"),
        },
    )

    # Disabled: plain uniform plug randomization. Superseded by the at-goal
    # curriculum below (re-enable this and comment out reset_plug_curriculum to
    # revert to simple uniform plug randomization).
    # randomize_plug_pose = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": [-0.02, 0.02],
    #             "y": [-0.02, 0.02],
    #             "z": [-0.01, 0.01],
    #             "roll": [0.0, 0.0],
    #             "pitch": [0.0, 0.0],
    #             "yaw": [0.0, 0.0],
    #         },
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("dp_plug"),
    #     },
    # )

    # 80% at-goal curriculum (mirrors IsaacLab_UR gb300, osmo at_goal_prob=0.8).
    # Identical to the task-space env: IsaacLab_UR applies this curriculum
    # regardless of control space (it lives in the object reset, not the
    # controller). A fraction `at_goal_prob` of envs spawn the plug already
    # inserted at a random depth (0 -> insertion_length) with goal orientation;
    # the rest get uniform approach-pose randomization via `normal_pose_range`.
    #
    # at_goal_prob is annealed linearly from `at_goal_prob` (start) down to
    # `at_goal_prob_final` between iterations `anneal_start_iter` and
    # `anneal_end_iter`. Iterations are derived from the env step counter via
    # `num_steps_per_env` (must match the agent cfg). Set `at_goal_prob_final=None`
    # to disable annealing (constant at_goal_prob).
    reset_plug_curriculum = EventTerm(
        func=mdp.reset_plug_at_goal_curriculum,
        mode="reset",
        params={
            "plug_cfg": SceneEntityCfg("dp_plug"),
            "socket_cfg": SceneEntityCfg("dp_socket"),
            "at_goal_prob": 0.8,
            "at_goal_prob_final": 0.2,
            "anneal_start_iter": 0.0,
            "anneal_end_iter": 1000.0,
            "num_steps_per_env": 512,
            "insertion_axis": [1.0, 0.0, 0.0],
            "insertion_length": _INSERTION_LENGTH,
            "socket_insertion_offset": SOCKET_INSERTION_OFFSET,
            "plug_insertion_offset": PLUG_INSERTION_OFFSET,
            "goal_rot": list(PLUG_GOAL_ROT),
            # Non-at-goal approach randomization. Matches gb300's
            # held_asset_init_pos_range = [0.02, 0.02, 0.01] (x, y, z) [m].
            "normal_pose_range": {
                "x": [-0.02, 0.02],
                "y": [-0.02, 0.02],
                "z": [-0.01, 0.01],
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
    """Flexiv Rizon 4s + Grav gripper DisplayPort insertion (joint-space)."""

    def __post_init__(self):
        super().__post_init__()

        if USE_SIM2REAL_ACTION_MODEL:
            # Deployment sim rate: 200 Hz PhysX physics, decimation 4 => 50 Hz
            # control loop (overrides the base 240 Hz / decimation 8). At 50 Hz a
            # 20 ms latency maps to exactly one control step.
            self.decimation = USE_SIM2REAL_DECIMATION
            self.sim.dt = 1.0 / USE_SIM2REAL_PHYSICS_FREQ_HZ
            self.sim.render_interval = self.decimation

        self.end_effector_body_name = "flange"
        self.num_arm_joints = 7
        # CALIBRATE: flange position in the DisplayPort plug's local frame.
        # Seeded from the GB300 grasp offset; the right-angle DP plug/overmold
        # geometry differs, so verify the grasp lands on the plug body in sim
        # before training (the grasp event solves IK toward plug + this offset).
        self.grasp_offset = [0.0025, 0.0, -0.1875]
        # Identity: the target EEF orientation equals the plug orientation.
        self.grasp_rot_offset = [0.0, 0.0, 0.0, 1.0]
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
        _arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        if USE_SIM2REAL_ACTION_MODEL:
            # Delayed + velocity/acceleration-shaped relative joint action.
            self.actions.arm_action = mdp.ShapedDelayedRelativeJointPositionActionCfg(
                asset_name="robot",
                joint_names=_arm_joint_names,
                scale=self.joint_action_scale,
                use_zero_offset=True,
                latency_s=USE_SIM2REAL_ACTION_LATENCY_S,
                command_velocity_limit=USE_SIM2REAL_COMMAND_VELOCITY_LIMIT,
                command_acceleration_limit=USE_SIM2REAL_COMMAND_ACCELERATION_LIMIT,
            )
        else:
            self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
                asset_name="robot",
                joint_names=_arm_joint_names,
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

        # Grav gripper actuator configuration (implicit - PhysX built-in joint drives).
        self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=2.0,
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

        if USE_SIM2REAL_ACTION_MODEL:
            # Replace the stock arm PD (shoulder 1320 / elbow 600 / wrist 216)
            # with the PhysX SysID-tuned per-joint gains. Values are keyed by
            # joint name so each implicit actuator group picks up its own gains.
            for _group, _joints in (
                ("shoulder", ("joint1", "joint2")),
                ("elbow", ("joint3", "joint4")),
                ("wrist", ("joint5", "joint6", "joint7")),
            ):
                self.scene.robot.actuators[_group].stiffness = {
                    _j: _SIM2REAL_ARM_STIFFNESS[_j] for _j in _joints
                }
                self.scene.robot.actuators[_group].damping = {
                    _j: _SIM2REAL_ARM_DAMPING[_j] for _j in _joints
                }

        self.scene.dp_socket.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_SOCKET_ROOT,
            rot=_SOCKET_ROT,
        )
        self.scene.dp_plug.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_PLUG_ROOT,
            rot=_PLUG_ROT,
        )

        # Grav gripper widths (joint angles).
        # grasp: fingers wide open for approach.
        # hold: fingers touching plug surface (no mesh overlap).
        # close: fully-closed target that the actuator drives toward.
        # CALIBRATE: tune for the DisplayPort plug width.
        self.hand_grasp_width = 0.3
        self.hand_hold_width = -0.05
        self.hand_close_width = -0.155

        self.events.set_robot_to_grasp_pose.params["end_effector_body_name"] = self.end_effector_body_name
        self.events.set_robot_to_grasp_pose.params["num_arm_joints"] = self.num_arm_joints
        self.events.set_robot_to_grasp_pose.params["grasp_rot_offset"] = self.grasp_rot_offset
        self.events.set_robot_to_grasp_pose.params["grasp_offset"] = self.grasp_offset
        self.events.set_robot_to_grasp_pose.params["gripper_joint_setter_func"] = self.gripper_joint_setter_func
        self.events.set_robot_to_grasp_pose.params["max_iterations"] = 150

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
    """Joint-space variant that hides joint velocity from the actor (policy).

    Identical to :class:`Rizon4sGravDisplayportInsertionEnvCfg` except the actor
    observation drops joint velocity (actor sees joint positions + socket pose
    only). The critic keeps joint velocity as privileged information, so the
    value function is unaffected. Useful for testing velocity-free deployment
    (e.g. when reliable joint-velocity estimates are not available on the real
    robot).
    """

    def __post_init__(self):
        super().__post_init__()
        # Remove joint velocity from the actor group; setting a term to None
        # disables it in the manager-based ObservationManager. The critic group
        # still includes joint_vel.
        self.observations.policy.joint_vel = None


@configclass
class Rizon4sGravDisplayportInsertionNoJointVelEnvCfg_PLAY(Rizon4sGravDisplayportInsertionNoJointVelEnvCfg):
    """Play configuration for the no-joint-velocity joint-space variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
