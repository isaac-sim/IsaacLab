# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity tracking with MicroDuck on generated terrain.

The policy uses 14 HOME-relative joint actions, an IMU and velocity/head/body pose commands.
Native BAM servos run at the physics rate; the control rate is 50 Hz.
"""

import math

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
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
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.contrib.microduck.mdp as mdp
import isaaclab_tasks.core.velocity.mdp as velocity_mdp
from isaaclab_tasks.contrib.microduck.assets import MICRODUCK_CFG
from isaaclab_tasks.utils import PresetCfg

##
# Shared constants
##

MICRODUCK_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
"""The 14 servo joints in MJCF order: left leg, then neck/head, then right leg."""

MICRODUCK_LEG_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
"""The 10 leg joints, which are the ones the posture reward holds at the stand pose."""

MICRODUCK_HEAD_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
"""The 4 head servos, in the order the head-pose command indexes its columns.

This order is a contract between the command term and the two head rewards, not a preference, so
every selection of these joints pins it with ``preserve_order=True``.
"""

MICRODUCK_FOOT_BODY_NAMES = ["ankle_left", "ankle_right"]
"""Foot bodies in ``[left, right]`` site order."""

MICRODUCK_TRUNK_BODY_NAME = "trunk_base"
"""Body measures the base pose, upright reward and self-collisions on."""

MICRODUCK_HEAD_BODY_NAMES = ["neck", "neck_pitch(_[0-9]+)?", "yaw_roll_motion", "jaw_soft"]
"""Body-name patterns the head centre-of-mass randomization perturbs."""

MICRODUCK_ALLCOLLISIONS_COLLIDER_XFORMS = [
    "left_foot_collision",
    "right_foot_collision",
    "leg_1",
    "hip_l_1",
    "top_head_shell_1",
    "jaw_1",
    "bottom_head_shell_1",
    "np_f970_1",
]
"""Xforms holding the ten enabled colliders of the all-collisions model, by disambiguated name."""

MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/(.*/)?(" + "|".join(MICRODUCK_ALLCOLLISIONS_COLLIDER_XFORMS) + ")/[^/]*"
)
"""Shape-level prim expression selecting those ten colliders, for the Newton contact sensor.

Shape expressions full-match against shape paths rather than body paths, so the trailing ``/[^/]*``
is what reaches the collider prim below its Xform, and the optional ``(.*/)?`` spans the bodies
between the trunk and each collider -- including none at all, for the battery pack bolted to the
trunk itself.
"""

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_LEG_JOINT_NAMES, preserve_order=True)
_HEAD_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_HEAD_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_HEAD_BODY_NAMES)
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
_FOOT_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
# no ``preserve_order``: the material randomization resolves body IDs into backend shape ranges and
# documents that callers must not pre-swizzle them
_FOOT_MATERIAL_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES)
_SELF_COLLISION_SENSOR_CFG = SceneEntityCfg("self_collision")

_IMU_MISALIGNMENT_DEG = 6.0
"""Upper bound [deg] on the IMU mounting-misalignment angle.

The angle is drawn once per robot and never resampled, so this bounds a fixed mounting error rather
than a disturbance. Both misaligned terms read the *same* rotation, so they must be configured with
this same bound -- the shared accessor raises rather than serve two different ones.
"""

_IMU_DELAY_UPDATE_PERIOD = 64
"""Control steps between two draws of the IMU latency.

The lag itself is 0 or 1 step. Re-drawing it every step would give a flicker the policy can filter
out; holding it for tens of steps makes it a latency regime the policy has to tolerate instead.
"""

##
# Foot-height targets
##

MICRODUCK_SOLE_TO_ANKLE_OFFSET = 0.0225
"""Height [m] of the ankle body frame above the sole, with the foot level."""

MICRODUCK_FOOT_TARGET_HEIGHT = 0.02 + MICRODUCK_SOLE_TO_ANKLE_OFFSET
"""Swing-height target [m] in the ankle-body frame, 0.0425 m."""

_FOOT_SWING_HEIGHT_WEIGHT = -0.25 * (MICRODUCK_FOOT_TARGET_HEIGHT / 0.02) ** 2
"""Swing-height penalty weight, -1.12890625."""

# # Curriculum stage tables.

MICRODUCK_STEPS_PER_ITERATION = 24
"""Environment steps per PPO iteration, matching ``MicroDuckPPORunnerCfg.num_steps_per_env``.

Imported by ``MicroDuckPPORunnerCfg`` so the two cannot drift apart; a runner change here would
otherwise silently desynchronize every curriculum boundary from "iteration N".
"""


def _iterations(count: int) -> int:
    """Convert a iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


##
# Terrain
##

MICRODUCK_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    # 1 mm height quantization, so a 1.7 deg slope is a smooth ramp rather than a staircase of 5 mm ledges the robot
    # cannot step over.
    vertical_scale=0.001,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        # MicroDuck lifts its feet 1-2 cm, so steps are capped at 1.5 cm rather than the 23 cm of
        # the stock ROUGH_TERRAINS_CFG
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.0, 0.015),
            step_width=0.15,
            platform_width=2.0,
            border_width=1.0,
        ),
        # uneven cobblestone-like ground.
        "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.30,
            grid_width=0.45,
            grid_height_range=(0.0, 0.010),
            platform_width=1.5,
        ),
        # gentle slopes, rise over run: 0.03 to 0.10 is 1.7 to 5.7 degrees by difficulty. The stock
        # ROUGH_TERRAINS_CFG rings its slopes with a 0.25 m flat border, but a pyramid slope is already
        # level with its neighbours at the patch edge -- the height field is ``height_max * xx * yy`` with
        # ``xx = yy = 0`` there -- so a border would only shorten the slope for no continuity gain.
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.03, 0.10),
            platform_width=2.0,
        ),
    },
)
"""Rough terrains MicroDuck trains on."""

##
# Physics presets
##


@configclass
class MicroDuckPhysicsCfg(PresetCfg):
    """Backend presets for the MicroDuck velocity environments.

    The environment requires Newton-native BAM drives and MJWarp's friction-constraint bridge.
    Other physics backends are not offered by this task.
    """

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1500,
            nconmax=200,
            iterations=30,
            ls_iterations=50,
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
class MicroDuckSceneCfg(InteractiveSceneCfg):
    """Scene with the MicroDuck robot on a ground plane."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MICRODUCK_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialBaseCfg(
            # neutral, so the friction the robot walks on is the 1.0 the MJCF authors on the foot
            # soles and the foot-friction randomization scales
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot = MICRODUCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # The pattern is deliberately unlike the flat ``Robot/[^/]*`` other locomotion tasks use. The MJCF import nests
    # every link under the trunk -- the ankle links the soles hang off sit four levels below it -- so a
    # single-level pattern matches nothing here. Newton compiles this as one plain regular expression and
    # full-matches it against body labels, which are whole prim paths, so ``.*`` crosses ``/`` and reaches them.
    # That is a Newton-only property: the PhysX backend matches prim paths one path token at a time and would
    # need a different expression, which is consistent as long as :class:`MicroDuckPhysicsCfg` offers only the
    # MJWarp backend.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(ankle_left|ankle_right))?",
        history_length=3,
        track_air_time=True,
    )

    # Self-collision sensor. The walk model has five collidable geometries: the two soles, and ``power_support``
    # (the battery holder on the trunk) plus the ``leg`` mesh on each shin, all three in the MJCF's
    # ``self_collision_only`` class (``contype=2 conaffinity=2``) so they collide with each other but never with
    # the ground. The prebuilt asset disables them to preserve the MJCF's world-contact set. They have no
    # collision role left, so this sensor cannot see them. LOST GUARD-RAIL: shin-versus-battery-holder and
    # shin-versus-shin contacts are unpenalized here. This is a *collider* gap, not a sensing one: those
    # geometries have no collision role on this model, so no sensor can see them. If a future asset restores
    # their self-collision role, widen this sensor the way the stand-up and roulade tasks are widened --
    # ``sensor_shape_prim_expr``/``filter_shape_prim_expr`` over
    # :data:`MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR`, with ``saturate=True`` on the reward -- rather than
    # by growing this body-level pair. Until then this sensor stays exactly as it is.
    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/ankle_left",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/ankle_right"],
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
    """Command specifications for the MDP."""

    # The two bucket fractions are what ``MicroDuckVelocityCommand`` adds over the stock term: a fifth of the
    # environments walk straight forward and 15% turn on the spot.
    base_velocity = mdp.ForwardMaskVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        rel_forward_envs=0.2,
        rel_turn_in_place_envs=0.15,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.ForwardMaskVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

    # A joint-position delta from the stand pose for ``(neck_pitch, head_pitch, head_yaw, head_roll)``; the ranges
    # start almost closed and the ``head_pose_range`` curriculum opens them. Looking around on command is a
    # primary MicroDuck objective, not a disturbance: ``head_pose_tracking`` carries weight 2.0, the same as
    # velocity tracking.
    head_pose = mdp.ZeroablePoseDeltaCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015)),
    )

    # A trunk-pose delta, ``(x, y, z)`` [m] then ``(roll, pitch, yaw)`` [rad].
    body_pose = mdp.ZeroablePoseDeltaCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=(
            (-0.005, 0.005),
            (-0.005, 0.005),
            (-0.005, 0.005),
            (-0.05, 0.05),
            (-0.05, 0.05),
            (-0.05, 0.05),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # The biased variant additionally subtracts the encoder bias the actor's ``joint_pos`` observation adds, which
    # is the loop a real servo closes on its own miscalibrated encoder: the policy commands ``a``, reads ``a``
    # back, and the joint actually settles at ``default + a - bias``. Wiring only one half of that loop would
    # train a permanent joint-space offset into the policy instead of a calibration error it has to tolerate.
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

    The actor reads the robot the way the deployed runtime does -- through a miscalibrated encoder,
    a misaligned IMU, a bus latency and per-step noise -- while the critic reads the true state plus
    the privileged quantities the robot has no sensor for.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group: the 61-wide deploy contract.

        Term order **is** the deployed observation layout, which the runtime
        on the robot rebuilds by hand from its own sensor reads: base angular velocity (3),
        projected gravity (3), joint positions (14), joint velocities (14), the previous action
        (14), then the command block ``[twist(3), head_pose(4), body_pose(6)]``. Moving a term or
        changing its width silently invalidates every trained checkpoint, so the layout is pinned by
        a test rather than left to term declaration order alone.

        Noise magnitudes are MicroDuck's own overrides of the base template,
        an order of magnitude tighter throughout. On top of the noise the actor carries the three
        systematic corruptions per-step noise cannot express:

        * a constant per-robot **encoder bias** on ``joint_pos``, closed against
          :class:`~isaaclab_tasks.contrib.microduck.mdp.actions.ArticulationEncoderBiasAction`;
        * a constant per-robot **IMU mounting misalignment** on the two IMU terms, which share one
          rotation and therefore must agree on ``max_angle_deg``;
        * a stochastic **bus latency**: 0 or 1 control step on the IMU, re-drawn at most every 64
          steps, and a constant 1 step on ``joint_vel``, whose firmware derives velocity from the
          previous position-sample window.

        Isaac Lab applies the noise to the delayed value, where mjlab delays the already-noised one.
        Both give a stale signal plus an independent uniform draw, so the observation distribution
        is the same; see
        :class:`~isaaclab_tasks.contrib.microduck.mdp.observations.delayed_observation`.
        """

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
        head_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        body_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for the value function.

        ``enable_corruption=False`` strips the noise, but it gates neither the delay nor the
        misalignment, so the terms here are the stock, undelayed, unmisaligned ones rather than
        references to the actor's. This group is not a deploy contract, so its width is measured
        rather than declared.

        ``foot_height`` measures the body height above the environment origin, which is the
        clearance over the ground on flat terrain only; on the rough task the critic's foot heights
        follow the trunk over the terrain instead of the terrain under the foot.
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
        foot_height = ObsTerm(func=mdp.foot_origin_height_safe, params={"asset_cfg": _FOOT_BODY_CFG})
        foot_air_time = ObsTerm(func=mdp.foot_group_air_time_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact = ObsTerm(func=mdp.foot_group_contact, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact_forces = ObsTerm(func=mdp.foot_group_contact_forces_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        head_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        body_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events."""

    ##
    # Startup: properties of the individual robot, fixed for its whole life.
    ##

    # On Newton this term samples a single friction coefficient per shape, which is exactly MuJoCo's model, and
    # ignores ``dynamic_friction_range`` and ``num_buckets``; both are supplied because the signature requires
    # them and the PhysX backend would need them. The soles are the only collidable geometries on the robot, and
    # the ground plane combines friction multiplicatively at 1.0, so this range *is* the friction the robot
    # walks on.
    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": _FOOT_MATERIAL_CFG,
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Startup, not reset: a calibration error is a property of the robot, and resampling it every
    # episode would let the policy average it out instead of learning to tolerate it.
    encoder_bias = EventTerm(
        func=mdp.randomize_articulation_encoder_bias,
        mode="startup",
        params={"bias_range": (-0.015, 0.015)},
    )

    # +/-5 % on the trunk. ``recompute_inertia=True`` reproduces that coupling: the inertia is scaled by the same
    # mass ratio.
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
    # Reset: redrawn every episode.
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

    # Trunk centre of mass, +/-3 mm on every axis to start with; the ``com_range`` curriculum opens it to +/-15 mm.
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": _TRUNK_BODY_CFG,
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )

    # The same, on the head bodies, ramped to +/-10 mm. A head that is heavier on one side than the
    # model says is what a real battery or camera mount produces, and the head is the mass the
    # walking controller is least able to compensate for.
    randomize_head_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": _HEAD_BODY_CFG,
            "com_range": {"x": (-0.003, 0.003), "y": (-0.003, 0.003), "z": (-0.003, 0.003)},
        },
    )

    # +/-10 % on the BAM servos' gearbox friction budget. The stock ``randomize_actuator_gains`` cannot express it
    # -- it writes PD gains this model never reads.
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_bam_friction,
        mode="reset",
        params={"scale_range": (0.9, 1.1)},
    )

    # +/-10 % on every joint's armature: the reflected rotor inertia of a servo is a fitted quantity, not a measured
    # one.
    randomize_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
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
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    ##
    # Velocity tracking.
    ##

    track_lin_vel = RewTerm(
        func=mdp.track_linear_velocity_raw,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.1)},
    )
    track_ang_vel = RewTerm(
        func=mdp.track_angular_velocity_raw,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )

    ##
    # Posture.
    ##

    # A bounded Gaussian *reward* on the trunk's gravity tilt, not the stock unbounded L2 penalty.
    upright = RewTerm(
        func=mdp.upright_selected_body,
        weight=2.0,
        params={"std": math.sqrt(0.05), "asset_cfg": _TRUNK_BODY_CFG},
    )

    # Hold the legs near the stand pose, with a tolerance that widens once the robot is asked to walk. The neck and
    # head joints are excluded: they follow the head-pose command instead.
    pose = RewTerm(
        func=mdp.pose_mode_switch,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std_standing": {
                ".*hip_yaw": 0.1,
                ".*hip_roll": 0.05,
                ".*hip_pitch": 0.15,
                ".*knee": 0.15,
                ".*ankle": 0.1,
            },
            "std_walking": {
                ".*hip_yaw": 0.3,
                ".*hip_roll": 0.05,
                ".*hip_pitch": 0.4,
                ".*knee": 0.4,
                ".*ankle": 0.25,
            },
            "std_running": {
                ".*hip_yaw": 0.3,
                ".*hip_roll": 0.05,
                ".*hip_pitch": 0.4,
                ".*knee": 0.4,
                ".*ankle": 0.25,
            },
            "walking_threshold": 0.01,
            "running_threshold": 1.5,
            "asset_cfg": _LEG_JOINT_CFG,
        },
    )

    ##
    # Trunk regularizers.
    ##

    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2_raw, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2_raw, weight=-0.02)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # The ``action_rate_weight`` curriculum ramps this to -1.0 by iteration 1500.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    ##
    # Gait and feet.
    ##

    air_time = RewTerm(
        func=mdp.feet_air_time_windowed_gated,
        weight=3.0,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "threshold_min": 0.125,
            "threshold_max": 0.300,
            "command_threshold": 0.01,
        },
    )
    foot_clearance = RewTerm(
        func=mdp.foot_origin_clearance,
        weight=-2.0,
        params={
            "target_height": MICRODUCK_FOOT_TARGET_HEIGHT,
            "command_name": "base_velocity",
            "asset_cfg": _FOOT_BODY_CFG,
            "command_threshold": 0.01,
        },
    )
    foot_swing_height = RewTerm(
        func=mdp.foot_origin_swing_height,
        weight=_FOOT_SWING_HEIGHT_WEIGHT,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "asset_cfg": _FOOT_BODY_CFG,
            "target_height": MICRODUCK_FOOT_TARGET_HEIGHT,
            "command_name": "base_velocity",
            "command_threshold": 0.01,
        },
    )
    foot_slip = RewTerm(
        func=mdp.foot_slip_raw,
        weight=-0.1,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "asset_cfg": _FOOT_BODY_CFG,
            "command_threshold": 0.01,
        },
    )
    self_collisions = RewTerm(
        func=mdp.self_collision_cost_saturable,
        weight=-1.0,
        params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG},
    )

    ##
    # Head and body pose tracking.
    ##

    head_pose_tracking = RewTerm(
        func=mdp.head_pose_tracking_raw,
        weight=2.0,
        params={"command_name": "head_pose", "std": 0.5, "asset_cfg": _HEAD_JOINT_CFG},
    )
    # Weight 0.0, ramped to +3.0 from iteration 600 by the ``head_pose_bias_weight`` curriculum.
    # The weight is **positive** because the term already returns a non-positive value.
    head_pose_bias = RewTerm(
        func=mdp.head_pose_bias_penalty_gated,
        weight=0.0,
        params={"command_name": "head_pose", "tau_s": 1.0, "asset_cfg": _HEAD_JOINT_CFG},
    )

    body_pose_tracking = RewTerm(
        func=mdp.body_pose_tracking_6d,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "nominal_height": 0.095,
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15.0),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})

    # The default 3.0 would let the robot walk 17 m into the border before terminating.
    out_of_terrain_bounds = DoneTerm(
        func=velocity_mdp.terrain_out_of_bounds,
        time_out=True,
        params={"distance_buffer": 20.3},
    )

    # Catches a *broken* robot rather than a fallen one: a NaN state compares false against every bound, so no other
    # termination would fire on it and the environment would train on garbage forever.
    nan_state = DoneTerm(
        func=mdp.robot_state_is_nan_optional_sensors,
        time_out=False,
        params={"sensor_names": ("contact_forces",)},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=velocity_mdp.terrain_levels_vel)

    action_rate_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": _iterations(0), "weight": -0.1},
                {"step": _iterations(500), "weight": -0.2},
                {"step": _iterations(750), "weight": -0.4},
                {"step": _iterations(1000), "weight": -0.6},
                {"step": _iterations(1250), "weight": -0.8},
                {"step": _iterations(1500), "weight": -1.0},
            ],
        },
    )

    # Held at 0.0 until the robot walks: an early bias penalty rewards holding the head still, which
    # is the opposite of what the head-pose command asks for.
    head_pose_bias_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "head_pose_bias",
            "weight_stages": [
                {"step": _iterations(0), "weight": 0.0},
                {"step": _iterations(600), "weight": 1.0},
                {"step": _iterations(1000), "weight": 2.0},
                {"step": _iterations(1500), "weight": 3.0},
            ],
        },
    )

    standing_envs = CurrTerm(
        func=mdp.standing_envs_stages,
        params={
            "command_name": "base_velocity",
            "standing_stages": [
                {"step": _iterations(0), "rel_standing_envs": 0.02},
                {"step": _iterations(500), "rel_standing_envs": 0.05},
                {"step": _iterations(750), "rel_standing_envs": 0.1},
                {"step": _iterations(1000), "rel_standing_envs": 0.15},
                {"step": _iterations(1500), "rel_standing_envs": 0.2},
                {"step": _iterations(2000), "rel_standing_envs": 0.25},
            ],
        },
    )

    # The final neck-pitch command range can exceed the joint limit, leaving residual tracking error.
    head_pose_range = CurrTerm(
        func=mdp.command_range_stages,
        params={
            "command_name": "head_pose",
            "range_stages": [
                {
                    "step": _iterations(0),
                    "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015)),
                },
                {
                    "step": _iterations(500),
                    "ranges": ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047)),
                },
                {
                    "step": _iterations(1000),
                    "ranges": ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11)),
                },
                {
                    "step": _iterations(1500),
                    "ranges": ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.20, 0.20)),
                },
                {
                    "step": _iterations(2000),
                    "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31)),
                },
            ],
        },
    )

    # A single stage, so the body-pose ranges never move.
    body_pose_range = CurrTerm(
        func=mdp.command_range_stages,
        params={
            "command_name": "body_pose",
            "range_stages": [
                {
                    "step": _iterations(0),
                    "ranges": (
                        (-0.005, 0.005),
                        (-0.005, 0.005),
                        (-0.005, 0.005),
                        (-0.05, 0.05),
                        (-0.05, 0.05),
                        (-0.05, 0.05),
                    ),
                },
            ],
        },
    )

    com_range = CurrTerm(
        func=mdp.event_range_stages,
        params={
            "event_name": "randomize_com",
            "range_stages": [
                {"step": _iterations(0), "range": 0.003},
                {"step": _iterations(500), "range": 0.005},
                {"step": _iterations(1000), "range": 0.01},
                {"step": _iterations(1500), "range": 0.015},
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
class MicroDuckVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """MicroDuck velocity-tracking environment on rough terrain."""

    sim: SimulationCfg = SimulationCfg(physics=MicroDuckPhysicsCfg())

    scene: MicroDuckSceneCfg = MicroDuckSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # 0.005 s physics steps decimated by 4 give the 50 Hz control rate the deployed policy runs
        # at, and 20 s episodes are 1000 control steps.
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        # The decimation above is even, which is what lets the stateful servo delay line be CUDA-graph-captured.
        self.sim.use_newton_actuators = True
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.self_collision is not None:
            self.scene.self_collision.update_period = self.sim.dt
        # the terrain-level curriculum needs the generator to lay its sub-terrains out by difficulty
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = self.curriculum.terrain_levels is not None
        # MicroDuck stands 0.13 m tall, so the stock viewer distance frames empty ground.
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.8, 0.8, 0.4), lookat=(0.0, 0.0, 0.1))
