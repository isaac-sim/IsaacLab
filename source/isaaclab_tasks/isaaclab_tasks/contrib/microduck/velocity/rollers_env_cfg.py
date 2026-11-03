# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity tracking on passive roller skates.

The four wheel hinges remain unactuated. Roller-specific reset heights keep the tires above
the floor, and contact observations group the two wheels on each foot.
"""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.contrib.microduck.mdp as mdp
from isaaclab_tasks.contrib.microduck.assets import MICRODUCK_ROLLERS_CFG
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    MICRODUCK_HEAD_BODY_NAMES,
    MICRODUCK_HEAD_JOINT_NAMES,
    MICRODUCK_JOINT_NAMES,
    MICRODUCK_LEG_JOINT_NAMES,
    MICRODUCK_STEPS_PER_ITERATION,
    MICRODUCK_TRUNK_BODY_NAME,
)
from isaaclab_tasks.utils import PresetCfg

##
# Roller-model constants
##

MICRODUCK_WHEEL_JOINT_NAMES = ["passive_LF_wheel", "passive_LR_wheel", "passive_RF_wheel", "passive_RR_wheel"]
"""The four undriven wheel hinges, in the model's left-front-to-right-rear order."""

MICRODUCK_ROLLERS_JOINT_NAMES = MICRODUCK_JOINT_NAMES + MICRODUCK_WHEEL_JOINT_NAMES
"""All 18 hinges of the roller model: the 14 servos in deploy order, then the four wheels."""

MICRODUCK_TIRE_BODY_NAMES = ["tire", "tire_2", "tire_3", "tire_4"]
"""The four tire bodies, ordered ``[left-front, left-rear, right-front, right-rear]``.

This order is a contract, not a preference. It is the order the contact sensor reports slots in, and
every foot term folds consecutive pairs of slots into one foot, so a permutation would silently swap
the left and right feet -- which
:class:`~isaaclab_tasks.contrib.microduck.mdp.rewards.gait_symmetry_penalty` reads directly.
"""

MICRODUCK_FOOT_BODY_NAMES = ["ankle_l_v1", "ankle_r_v1"]
"""The two ankle bodies, in ``[left, right]`` order.

The roller model renames the walking model's ``ankle_left`` / ``ankle_right`` and moves the sole
collider onto the tires hanging off them, so these bodies carry no collider at all -- they are the
*frame* the blade-flatness reward measures, not a contact body.
"""

MICRODUCK_TIRES_PER_FOOT = 2
"""Contact bodies making up one foot on this model."""

MICRODUCK_FOOT_NORMAL_AXIS = (0.0, 1.0, 0.0)
"""Sole normal of a skate blade, in the ankle body frame.

Confirmed in simulation rather than only on paper: with the robot upright at the stand pose the
gravity direction in either ankle body frame is ``(0.000, -0.996, 0.087)``, i.e. within 5 degrees of
this axis on both feet, which no other body axis is anywhere near.
"""

MICRODUCK_ROLLERS_STANDING_HEIGHT = 0.14070
"""Height [m] of ``trunk_base`` with the joints at the stand pose and the tires just touching the ground.

Recorded here because two of this task's numbers are sized against a *different* model and only make
sense next to it; see :data:`MICRODUCK_ROLLERS_SPAWN_HEIGHT` and :attr:`RewardsCfg.com_height_target`.
"""

MICRODUCK_ROLLERS_SPAWN_HEIGHT = 0.1385
"""Spawn height [m] of ``trunk_base``, the midpoint of reset band ``(0.1335, 0.1435)``."""

##
# Scene entity selections
##

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_ALL_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_ROLLERS_JOINT_NAMES, preserve_order=True)
_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_LEG_JOINT_NAMES, preserve_order=True)
_HEAD_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)
_HIP_ROLL_JOINT_CFG = SceneEntityCfg("robot", joint_names=["left_hip_roll", "right_hip_roll"], preserve_order=True)
_WHEEL_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_WHEEL_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_HEAD_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_HEAD_BODY_NAMES)
_FOOT_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_TIRE_BODY_NAMES, preserve_order=True)
_SELF_COLLISION_SENSOR_CFG = SceneEntityCfg("self_collision")

_IMU_MISALIGNMENT_DEG = 6.0
"""Upper bound [deg] on the IMU mounting-misalignment angle, matched to the velocity task."""

_IMU_DELAY_UPDATE_PERIOD = 64
"""Control steps between two draws of the IMU latency."""


def _iterations(count: int) -> int:
    """Convert a PPO iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


##
# Physics preset
##


@configclass
class MicroDuckRollersPhysicsCfg(PresetCfg):
    """Backend preset for the MicroDuck roller-skating environment.

    MJWarp is also the only backend that can run this task as configured: the environment sets
    ``sim.use_newton_actuators = True`` and the solver-hosted BAM model is rejected on the PhysX
    family's host adapter. See
    :class:`~isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg.MicroDuckPhysicsCfg`.
    """

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            # Reserve contact and constraint headroom for simultaneous body-ground contact.
            # Constraint limits are per environment; contact storage is shared across environments.
            njmax=160,
            nconmax=32,
            iterations=10,
            ls_iterations=20,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=1,
        default_shape_cfg=NewtonShapeCfg(margin=0.0),
    )
    default = newton_mjwarp


##
# Scene definition
##


@configclass
class MicroDuckRollersSceneCfg(InteractiveSceneCfg):
    """Scene with the roller-skating MicroDuck on a ground plane, and two contact sensors."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialBaseCfg(
            # neutral, so the friction the tires roll on is the one the MJCF authors on them.
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # The roller model, spawned at a height its own wheels support. See
    # :data:`MICRODUCK_ROLLERS_SPAWN_HEIGHT` for why the asset's own 0.125 m is overridden here rather
    # than corrected in the asset.
    robot = MICRODUCK_ROLLERS_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=MICRODUCK_ROLLERS_CFG.init_state.replace(pos=(0.0, 0.0, MICRODUCK_ROLLERS_SPAWN_HEIGHT)),
    )

    # Trunk and the four tires. Unlike the walking task's, this sensor cannot watch the ankle bodies: on this model
    # they carry no collider and would report zero force forever. The four tires are what touches the ground,
    # and the foot terms fold them back into two feet.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(tire|tire_2|tire_3|tire_4))?",
        history_length=3,
        track_air_time=True,
    )

    # Isaac Lab resolves a per-partner force matrix only for a ``prim_path`` matching a single prim per environment;
    # widening it needs the Newton backend's shape-level ``sensor_shape_prim_expr`` / ``filter_shape_prim_expr``
    # and is tracked separately.
    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/(jaw_soft|hip_l|hip_l_2|leg|leg_2|tire|tire_2|tire_3|tire_4)"
        ],
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP.

    One command, and its three slots do not mean what the walking task's mean:
    """

    base_velocity = mdp.RelativeHeadingVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        # Unlike on the stand-up and roll tasks, where the forced surge is read by nothing, here it means a fifth of
        # the resamples get a commanded throttle of at least 0.3 -- which six reward terms do read. It skews
        # the command distribution and is reproduced rather than cleaned up.
        rel_forward_envs=0.2,
        rel_turn_in_place_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.RelativeHeadingVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.6),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Identical to every other MicroDuck task's: the same 14 servos in the same deploy order, at the
    same unit scale, closing the same encoder-bias loop. The four wheel hinges are **not** actions --
    the servo group's ``^(?!passive_).*`` selector leaves them out, which is the property that keeps
    the action space 14 wide on a model with 18 joints.
    """

    joint_pos = mdp.ArticulationEncoderBiasActionCfg(
        asset_name="robot",
        joint_names=MICRODUCK_JOINT_NAMES,
        preserve_order=True,
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP.

    The actor group is the family's 61-wide deploy contract, and on this model the passive-joint
    exclusion in the two joint blocks is **load-bearing** rather than defensive: without it they
    would be 18 wide and the actor would present 69 values to a runtime expecting 61.
    The fourteen servo names exclude the passive wheel joints explicitly.

    This task has neither a head-pose nor a body-pose command, so both slots are zero padding -- the
    deployed runtime sends zeros for them too.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group: the 61-wide deploy contract."""

        base_ang_vel = ObsTerm(
            func=mdp.delayed_observation,
            params={
                "term_func": mdp.base_ang_vel_imu_misaligned,
                "term_params": {"max_angle_deg": _IMU_MISALIGNMENT_DEG},
                "min_lag": 0,
                "max_lag": 1,
                "update_period": _IMU_DELAY_UPDATE_PERIOD,
            },
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        projected_gravity = ObsTerm(
            func=mdp.delayed_observation,
            params={
                "term_func": mdp.projected_gravity_imu_misaligned,
                "term_params": {"max_angle_deg": _IMU_MISALIGNMENT_DEG},
                "min_lag": 0,
                "max_lag": 1,
                "update_period": _IMU_DELAY_UPDATE_PERIOD,
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_articulation_biased,
            params={"asset_cfg": _SERVO_JOINT_CFG, "biased": True},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        joint_vel = ObsTerm(
            func=mdp.delayed_observation,
            params={
                "term_func": mdp.joint_vel_rel,
                "term_params": {"asset_cfg": _SERVO_JOINT_CFG},
                "min_lag": 1,
                "max_lag": 1,
                "update_period": 0,
            },
            noise=Unoise(n_min=-0.25, n_max=0.25),
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        head_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 4})
        body_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 6})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for the value function.

        The actor's terms with every corruption removed, plus the base linear velocity, the three
        foot terms and -- unique to this task -- the four **wheel speeds**, which no sensor on the
        robot reports and which are the most direct measure of whether a stroke worked.
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_articulation_biased, params={"asset_cfg": _SERVO_JOINT_CFG, "biased": False}
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _SERVO_JOINT_CFG})
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        foot_air_time = ObsTerm(
            func=mdp.foot_group_air_time_safe,
            params={"sensor_cfg": _FOOT_SENSOR_CFG, "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT},
        )
        foot_contact = ObsTerm(
            func=mdp.foot_group_contact,
            params={"sensor_cfg": _FOOT_SENSOR_CFG, "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT},
        )
        foot_contact_forces = ObsTerm(
            func=mdp.foot_group_contact_forces_safe,
            params={"sensor_cfg": _FOOT_SENSOR_CFG, "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT},
        )
        wheel_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _WHEEL_JOINT_CFG})
        head_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 4})
        body_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 6})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events.

    The velocity task's randomization suite with three roller-specific changes:

    The push is gentler than the walking task's, ``+/-0.2 m/s`` against ``+/-0.3``: a robot on wheels
    keeps the velocity it is given.
    """

    ##
    # Startup: properties of the individual robot, fixed for its whole life.
    ##

    encoder_bias = EventTerm(
        func=mdp.randomize_articulation_encoder_bias,
        mode="startup",
        params={"bias_range": (-0.015, 0.015)},
    )

    mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    ##
    # Reset: redrawn every episode, in this order.
    ##

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.005, 0.005), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )

    # Wheel-bearing drag, shipped at zero and ramped by :attr:`CurriculumCfg.wheel_friction`.
    randomize_wheel_friction = EventTerm(
        func=mdp.randomize_joint_dry_friction,
        mode="reset",
        params={"asset_cfg": _WHEEL_JOINT_CFG, "friction_range": (0.0, 0.0)},
    )

    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )

    randomize_head_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": _HEAD_BODY_CFG,
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_bam_friction,
        mode="reset",
        params={"scale_range": (0.9, 1.1)},
    )

    # Servos only.
    randomize_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": _SERVO_JOINT_CFG,
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    ##
    # Interval.
    ##

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Read it in three groups:
    """

    upright = RewTerm(
        func=mdp.upright_selected_body,
        weight=2.0,
        params={"std": math.sqrt(0.2), "asset_cfg": _TRUNK_BODY_CFG},
    )

    pose = RewTerm(
        func=mdp.pose_mode_switch,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std_standing": {
                ".*hip_yaw.*": 0.05,
                ".*hip_roll.*": 0.05,
                ".*hip_pitch.*": 0.05,
                ".*knee.*": 0.05,
                ".*ankle.*": 0.05,
                ".*neck.*": 0.05,
                ".*head.*": 0.05,
                ".*passive_.*": 999.0,
            },
            "std_walking": {
                ".*hip_yaw.*": 0.3,
                ".*hip_roll.*": 0.6,
                ".*hip_pitch.*": 0.4,
                ".*knee.*": 0.4,
                ".*ankle.*": 0.25,
                ".*neck.*": 0.05,
                ".*head.*": 0.05,
                ".*passive_.*": 999.0,
            },
            "std_running": {
                ".*hip_yaw.*": 0.5,
                ".*hip_roll.*": 0.8,
                ".*hip_pitch.*": 0.8,
                ".*knee.*": 0.8,
                ".*ankle.*": 0.5,
                ".*neck.*": 0.05,
                ".*head.*": 0.05,
                ".*passive_.*": 999.0,
            },
            "walking_threshold": 0.01,
            "running_threshold": 0.5,
            "asset_cfg": _ALL_JOINT_CFG,
        },
    )

    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2_raw, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2_raw, weight=-0.02)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0)

    ##
    # Posture and height.
    ##

    # The COM target band is 1.7--4.7 cm below the 0.1407 m standing height, favoring a crouched posture.
    com_height_target = RewTerm(
        func=mdp.com_height_target,
        weight=2.0,
        params={"target_height_min": 0.0935, "target_height_max": 0.1235},
    )

    self_collisions = RewTerm(
        func=mdp.self_collision_cost_saturable,
        weight=-1.0,
        params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG},
    )

    # Gated per foot by that foot's own contact, so the swing blade is free to tilt.
    feet_flat = RewTerm(
        func=mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": _FOOT_BODY_CFG,
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "normal_axis": MICRODUCK_FOOT_NORMAL_AXIS,
            "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT,
        },
    )

    ##
    # Head and effort regularizers.
    ##

    neck_action_rate_l2 = RewTerm(
        func=mdp.joint_action_rate_l2,
        weight=-0.5,
        params={"action_name": "joint_pos", "asset_cfg": _HEAD_JOINT_CFG},
    )
    neck_joint_pos_l2 = RewTerm(func=mdp.joint_pose_l2, weight=-0.5, params={"asset_cfg": _HEAD_JOINT_CFG})
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-3, params={"asset_cfg": _SERVO_JOINT_CFG})
    action_over_limit = RewTerm(
        func=mdp.action_over_limit_penalty,
        weight=-0.5,
        params={"action_name": "joint_pos", "overshoot": 0.3},
    )
    hip_roll_neutral = RewTerm(func=mdp.joint_deviation_l1, weight=-2.0, params={"asset_cfg": _HIP_ROLL_JOINT_CFG})

    ##
    # The task: spin the wheels, and stop when asked.
    ##

    wheel_speed = RewTerm(
        func=mdp.wheel_speed_reward,
        weight=10.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": _WHEEL_JOINT_CFG,
            # the ground speed the tanh saturates near.
            "vel_scale": 0.3,
            "wheel_radius": 0.0175,
        },
    )
    braking = RewTerm(
        func=mdp.braking_reward,
        weight=1.0,
        params={"command_name": "base_velocity", "vel_std": 0.3},
    )

    ##
    # The stroke.
    ##

    skating_air_time = RewTerm(
        func=mdp.skating_air_time_reward,
        weight=1.5,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "threshold_min": 0.15,
            "threshold_max": 0.45,
            "vel_gate_ref": 0.2,
            "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT,
        },
    )

    glide = RewTerm(
        func=mdp.glide_reward,
        weight=4.0,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "asset_cfg": _LEG_JOINT_CFG,
            "vel_ref": 0.2,
            "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT,
        },
    )
    single_support = RewTerm(
        func=mdp.single_support_reward,
        weight=3.0,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "vel_gate_ref": 0.2,
            "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT,
        },
    )
    gait_symmetry = RewTerm(
        func=mdp.gait_symmetry_penalty,
        weight=-1.0,
        params={"sensor_cfg": _FOOT_SENSOR_CFG, "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT},
    )
    # ``target_pitch`` is the sine of the lean, so 0.262 asks for 15.2 degrees nose-down.
    forward_lean = RewTerm(
        func=mdp.forward_lean_reward,
        weight=1.5,
        params={"command_name": "base_velocity", "target_pitch": 0.262, "std": 0.1},
    )
    # The only thing keeping a straight-line skater straight, since the heading command is clamped to zero.
    heading_hold = RewTerm(func=mdp.heading_hold_reward, weight=1.0, params={"std": 0.4})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})

    # Catches a *broken* robot rather than a fallen one. The guard only changes behaviour in states that are already
    # broken, and the matching half of it is the pair of NaN-safe critic terms in :class:`ObservationsCfg`.
    nan_state = DoneTerm(
        func=mdp.robot_state_is_nan_optional_sensors,
        time_out=False,
        params={"sensor_names": ("contact_forces",)},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": _iterations(0), "weight": -1.0},
                {"step": _iterations(250), "weight": -1.5},
                {"step": _iterations(500), "weight": -2.0},
            ],
        },
    )

    # Bearing drag, held at zero until skating is robust.
    wheel_friction = CurrTerm(
        func=mdp.event_param_stages,
        params={
            "event_name": "randomize_wheel_friction",
            "inclusive": False,
            "param_stages": [
                {"step": _iterations(0), "params": {"friction_range": (0.0, 0.0)}},
                {"step": _iterations(2000), "params": {"friction_range": (5e-4, 5e-4)}},
                {"step": _iterations(3500), "params": {"friction_range": (1e-3, 1e-3)}},
                {"step": _iterations(5000), "params": {"friction_range": (1.5e-3, 1.5e-3)}},
            ],
        },
    )

    # Capped at +/-10 mm rather than the walking task's +/-15 mm: skating is less forgiving of a
    # displaced centre of mass than walking is.
    com_range = CurrTerm(
        func=mdp.event_range_stages,
        params={
            "event_name": "randomize_com",
            "range_stages": [
                {"step": _iterations(0), "range": 0.003},
                {"step": _iterations(500), "range": 0.005},
                {"step": _iterations(1000), "range": 0.01},
            ],
        },
    )

    head_com_range = CurrTerm(
        func=mdp.event_range_stages,
        params={
            "event_name": "randomize_head_com",
            "range_stages": [
                {"step": _iterations(0), "range": 0.003},
                {"step": _iterations(500), "range": 0.005},
                {"step": _iterations(1000), "range": 0.01},
            ],
        },
    )


##
# Environment configuration
##


@configclass
class MicroDuckVelocityRollersFlatEnvCfg(ManagerBasedRLEnvCfg):
    """MicroDuck roller-skating velocity environment on flat ground."""

    sim: SimulationCfg = SimulationCfg(physics=MicroDuckRollersPhysicsCfg())

    scene: MicroDuckRollersSceneCfg = MicroDuckRollersSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""

        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        # The decimation above is even, which is what lets the stateful servo delay line be CUDA-graph-captured.
        self.sim.use_newton_actuators = True
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        for sensor in (self.scene.contact_forces, self.scene.self_collision):
            if sensor is not None:
                sensor.update_period = self.sim.dt
        # MicroDuck stands 0.14 m tall on skates, so the stock viewer distance frames empty ground.
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.8, 0.8, 0.4), lookat=(0.0, 0.0, 0.1))
