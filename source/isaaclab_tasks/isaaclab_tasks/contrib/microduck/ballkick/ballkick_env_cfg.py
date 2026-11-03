# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck ball kicking with an analytic floorball prop.

The ball resets in the robot yaw frame after the robot pose is settled. Rewards combine
contact, kick direction, ball speed and robot stability.
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
from isaaclab_tasks.contrib.microduck.assets import (
    MICRODUCK_ALLCOLLISIONS_CFG,
    MICRODUCK_BALL_CFG,
    MICRODUCK_BALL_RADIUS,
)
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR,
    MICRODUCK_FOOT_BODY_NAMES,
    MICRODUCK_HEAD_BODY_NAMES,
    MICRODUCK_HEAD_JOINT_NAMES,
    MICRODUCK_JOINT_NAMES,
    MICRODUCK_LEG_JOINT_NAMES,
    MICRODUCK_STEPS_PER_ITERATION,
    MICRODUCK_TRUNK_BODY_NAME,
)
from isaaclab_tasks.utils import PresetCfg

##
# Which foot kicks
##

MICRODUCK_KICK_FOOT = "right"
"""The foot that swings at the ball."""

MICRODUCK_SUPPORT_FOOT = "left" if MICRODUCK_KICK_FOOT == "right" else "right"
"""The foot that must stay planted through the kick."""

##
# Ball placement
##

MICRODUCK_BALL_OFFSET_X = 0.09
"""Forward offset [m] of the ball centre from the robot root at reset."""

MICRODUCK_BALL_OFFSET_ABS_Y = 0.042
"""Lateral offset [m] of the ball from the robot's midline, toward the kicking foot.

The measured foot-site half-spacing is 0.0418 m, so this puts the ball on the kicking foot's own
line rather than in front of the robot's centre.
"""

MICRODUCK_BALL_POS_NOISE_XY = 0.015
"""Half-width [m] of the uniform noise on both ball offset components."""

MICRODUCK_BALL_TARGET_SPEED = 1.0
"""Ball speed [m/s] the kick rewards are shaped around.

.. note::

    At the shipped constants the kick pays ``12 * min(v, 1.0) - 4 * max(v - 1.0, 0)``: **+12/step**
    at target, zero at 4.0 m/s and a floor of -8/step past 6.0 m/s. That is four times the intended
    at-target payoff and roughly doubles the task's reward mass, so the shared regularizers act at
    about half their intended relative strength while the ball rolls.
"""

MICRODUCK_STAND_HEIGHT = 0.115
"""Standing trunk-height target [m], accounting for servo sag under load.

The geometric sole-grounded stand height is 0.11718 m.
"""

##
# Scene entity selections
##

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_LEG_JOINT_NAMES, preserve_order=True)
_HEAD_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_HEAD_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_HEAD_BODY_NAMES)
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
# no ``preserve_order``: the material randomization resolves body IDs into backend shape ranges and
# documents that callers must not pre-swizzle them
_FOOT_MATERIAL_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES)
_SELF_COLLISION_SENSOR_CFG = SceneEntityCfg("self_collision")
_SUPPORT_FOOT_SENSOR_CFG = SceneEntityCfg("support_foot_ground_contact")
_BALL_CFG = SceneEntityCfg("ball")

_TERRAIN_PRIM_PATH = "/World/ground"
"""Prim path of the ground plane, named so the support-foot sensor can filter against it."""

_SUPPORT_FOOT_COLLIDER_SHAPE_EXPR = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/(.*/)?" + f"{MICRODUCK_SUPPORT_FOOT}_foot_collision/[^/]*"
)
"""Shape-level expression selecting the support foot's sole collider.

Built the same way as :data:`~isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg.
MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR`: shape expressions full-match shape paths, so the
trailing token is what reaches the collider prim below its Xform and the optional ``(.*/)?`` spans
the bodies between the trunk and the ankle.
"""

_IMU_MISALIGNMENT_DEG = 6.0
"""Upper bound [deg] on the IMU mounting-misalignment angle, velocity-matched value."""

_IMU_DELAY_UPDATE_PERIOD = 64
"""Control steps between two draws of the IMU latency."""


def _iterations(count: int) -> int:
    """Convert a PPO iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


##
# Physics preset
##


@configclass
class MicroDuckBallKickPhysicsCfg(PresetCfg):
    """Backend preset for the MicroDuck ball-kick environment."""

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            # Reserve contact and constraint headroom for simultaneous body-ground contact.
            # Constraint limits are per environment; contact storage is shared across environments.
            njmax=128,
            nconmax=36,
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
class MicroDuckBallKickSceneCfg(InteractiveSceneCfg):
    """Scene with the all-collisions MicroDuck robot, a ball, and a ground plane."""

    terrain = TerrainImporterCfg(
        prim_path=_TERRAIN_PRIM_PATH,
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialBaseCfg(
            # neutral, so the friction the robot pushes off is the 1.0 the MJCF authors on the foot
            # soles and the foot-friction randomization scales
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # The kick is a standing task, so the extra colliders matter less than they do on the stand-up and forward-roll
    # tasks -- but a robot that loses its balance mid-swing lands on them, and the self-collision penalty needs
    # them to exist.
    robot = MICRODUCK_ALLCOLLISIONS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # The prop. Its position here is only what it has before the first reset; ``reset_ball`` places
    # it in the robot's yaw frame every episode.
    ball = MICRODUCK_BALL_CFG.replace(prim_path="{ENV_REGEX_NS}/Ball")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(ankle_left|ankle_right))?",
        history_length=3,
        track_air_time=True,
    )

    # The balance signal, and the one sensor this task adds. Filtering against the terrain rather than reading a net
    # force is load-bearing here in a way it is not for the family's other foot sensors: the ball rolls along
    # the ground into the robot's feet, so an unfiltered sole would report "grounded" while the support foot was
    # airborne and merely touching the ball -- exactly the state the term exists to deny. The terrain expression
    # is absolute rather than ``{ENV_REGEX_NS}``-relative because the ground plane is a single shape shared by
    # every environment.
    support_foot_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[_SUPPORT_FOOT_COLLIDER_SHAPE_EXPR],
        filter_shape_prim_expr=[f"{_TERRAIN_PRIM_PATH}/.*"],
    )

    self_collision = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR],
        filter_shape_prim_expr=[MICRODUCK_ALLCOLLISIONS_COLLIDER_SHAPE_EXPR],
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

    The kick takes no command: it is triggered at deployment by the policy switch, and there is
    nothing to steer once it starts. The twist survives only so the deployed 61-wide observation
    keeps its three-wide slot, and the head and body slots are zero padding rather than commands
    (see :class:`ObservationsCfg`).
    """

    # Ranges a hundredth of the velocity task's, resampled at most once per episode. Nothing reads it.
    base_velocity = mdp.ForwardMaskVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_forward_envs=0.2,
        rel_turn_in_place_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.ForwardMaskVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.01, 0.01),
            lin_vel_y=(-0.01, 0.01),
            ang_vel_z=(-0.05, 0.05),
            heading=None,
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Identical to the velocity task's: the same 14 servos in the same deploy order, at the same unit
    scale, closing the same encoder-bias loop.
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

    The actor group is byte-for-byte the velocity task's 61-wide deploy contract, which is the whole
    point of the family: one runtime on the robot feeds every MicroDuck policy from the same buffer.
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
        # Shape placeholders for the deployed vector's head and body command slots, which this task
        # has no commands for. They are deliberately constant zero rather than a live tiny range:
        # the slot exists so a runtime can hot-swap this policy for one that does steer the head.
        head_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 4})
        body_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 6})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for the value function.

        The actor's terms with every corruption removed, plus the base linear velocity, the three
        foot terms the robot has no sensor for, and the two ball terms that make this critic the
        widest in the family at 80.
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
        foot_air_time = ObsTerm(func=mdp.foot_group_air_time_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact = ObsTerm(func=mdp.foot_group_contact, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact_forces = ObsTerm(func=mdp.foot_group_contact_forces_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        head_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 4})
        body_pose_commands = ObsTerm(func=mdp.zero_command_padding, params={"dim": 6})
        ball_position = ObsTerm(func=mdp.ball_pos_in_base, params={"asset_cfg": _BALL_CFG})
        ball_velocity = ObsTerm(func=mdp.ball_vel_in_base, params={"asset_cfg": _BALL_CFG})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Configuration for events.

    The domain-randomization suite is the velocity task's, term for term and range for range, so a
    policy trained here meets the same hardware spread as one trained to walk. What this task adds is
    a standing-only :attr:`set_ground_state` and the ball placement that follows it.

    **Declaration order is behaviour.** Isaac Lab fires reset events in the order they are declared.
    :attr:`set_ground_state` overwrites the height and orientation :attr:`reset_base` wrote, and
    :attr:`reset_ball` then reads the *settled* robot pose to place the ball in its yaw frame. Moving
    the ball placement ahead of either one puts the ball at a heading the robot no longer has -- and
    with the ground-state reset drawing a uniformly random yaw, that means behind the robot on half
    the episodes.
    """

    ##
    # Startup: properties of the individual robot, fixed for its whole life.
    ##

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

    # Only the horizontal spread is live, so only the horizontal spread is configured.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "velocity_range": {}},
    )

    # Non-zero here, unlike every other task in this batch.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )

    # Standing-only: this task reuses the stand-up task's ground-state machinery purely for its noisy
    # upright spawn -- a random yaw, a few degrees of tilt and a height drawn around the standing
    # equilibrium. The three ground buckets are switched off, so their bands and keyframe are left
    # unconfigured rather than filled with values nothing samples.
    set_ground_state = EventTerm(
        func=mdp.reset_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.0,
            "face_up_prob": 0.0,
            "sitting_prob": 0.0,
            "standing_prob": 1.0,
            "standing_z_range": (0.11, 0.12),
            "sitting_tilt_max": math.radians(5.0),
            "asset_cfg": _SERVO_JOINT_CFG,
        },
    )

    # Must come after ``set_ground_state``: the ball is placed in the robot's *reset* yaw frame.
    reset_ball = EventTerm(
        func=mdp.reset_ball_in_front_of_foot,
        mode="reset",
        params={
            "offset": (
                MICRODUCK_BALL_OFFSET_X,
                -MICRODUCK_BALL_OFFSET_ABS_Y if MICRODUCK_KICK_FOOT == "right" else MICRODUCK_BALL_OFFSET_ABS_Y,
            ),
            "noise_xy": MICRODUCK_BALL_POS_NOISE_XY,
            "ball_radius": MICRODUCK_BALL_RADIUS,
            "asset_cfg": _BALL_CFG,
        },
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

    randomize_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_bam_friction,
        mode="reset",
        params={"scale_range": (0.9, 1.1)},
    )

    ##
    # Interval.
    ##

    # Ramped from nothing by the ``push_magnitude`` curriculum, as on the stand-up task: a robot
    # still learning to swing one leg cannot also be shoved.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Twelve terms, the smallest stack in the family, and they split cleanly in two:
    """

    ##
    # The kick.
    ##

    ball_forward_velocity = RewTerm(
        func=mdp.ball_forward_velocity,
        weight=12.0,
        params={"max_speed": MICRODUCK_BALL_TARGET_SPEED, "asset_cfg": _BALL_CFG},
    )
    # Returns a non-negative overshoot, so the weight is what makes it a penalty.
    ball_speed_overshoot = RewTerm(
        func=mdp.ball_speed_overshoot_penalty,
        weight=-4.0,
        params={"target_speed": MICRODUCK_BALL_TARGET_SPEED, "max_penalty": 5.0, "asset_cfg": _BALL_CFG},
    )

    ##
    # The stand: stay balanced on the support foot while the other one swings.
    ##

    support_foot_grounded = RewTerm(
        func=mdp.single_foot_grounded_reward,
        weight=2.0,
        params={"sensor_cfg": _SUPPORT_FOOT_SENSOR_CFG},
    )
    pose_stand_legs = RewTerm(
        func=mdp.joint_pose_gaussian,
        weight=2.0,
        params={"std": 0.5, "asset_cfg": _LEG_JOINT_CFG},
    )
    # There is no head-pose command on this task at all.
    pose_stand_neck = RewTerm(
        func=mdp.joint_pose_gaussian,
        weight=1.0,
        params={"std": 0.3, "asset_cfg": _HEAD_JOINT_CFG},
    )
    height_stand = RewTerm(
        func=mdp.root_height_gaussian,
        weight=1.0,
        params={"std": 0.04, "target_height": MICRODUCK_STAND_HEIGHT},
    )
    # The velocity task's exact recipe: a bounded Gaussian *reward* on the trunk's gravity tilt,
    # not the stock unbounded L2 penalty.
    upright = RewTerm(
        func=mdp.upright_selected_body,
        weight=2.0,
        params={"std": math.sqrt(0.05), "asset_cfg": _TRUNK_BODY_CFG},
    )

    ##
    # Sim-to-real regularizers, matched to the velocity task.
    ##

    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2_raw, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2_raw, weight=-0.02)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # Ramped to -1.0 by iteration 1500.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    self_collisions = RewTerm(
        func=mdp.self_collision_cost_saturable,
        weight=-1.0,
        params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG, "saturate": True},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.

    Unlike the stand-up and forward-roll tasks, the tilt termination is **kept**: the robot starts
    standing and is meant to still be standing when the episode ends, so falling over is a failure
    rather than a phase of the manoeuvre.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})

    # Catches a *broken* robot rather than a fallen one. The guard only changes behaviour in states that are already
    # broken, and the matching half of it is the pair of NaN-safe critic terms in :class:`ObservationsCfg`. The
    # ball is not covered by any state check on either stack, which is why the two ball observations guard
    # themselves.
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
                {"step": _iterations(0), "weight": -0.1},
                {"step": _iterations(500), "weight": -0.2},
                {"step": _iterations(750), "weight": -0.4},
                {"step": _iterations(1000), "weight": -0.6},
                {"step": _iterations(1250), "weight": -0.8},
                {"step": _iterations(1500), "weight": -1.0},
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

    push_magnitude = CurrTerm(
        func=mdp.event_param_stages,
        params={
            "event_name": "push_robot",
            "inclusive": False,
            "param_stages": [
                {"step": _iterations(0), "params": {"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}}},
                {"step": _iterations(500), "params": {"velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}}},
                {"step": _iterations(1000), "params": {"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}}},
            ],
        },
    )


##
# Environment configuration
##


@configclass
class MicroDuckBallKickFlatEnvCfg(ManagerBasedRLEnvCfg):
    """MicroDuck ball-kick environment on flat ground."""

    sim: SimulationCfg = SimulationCfg(physics=MicroDuckBallKickPhysicsCfg())

    scene: MicroDuckBallKickSceneCfg = MicroDuckBallKickSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # 0.005 s physics steps decimated by 4 give the 50 Hz control rate the deployed policy runs at.
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 0.005
        # The decimation above is even, which is what lets the stateful servo delay line be CUDA-graph-captured.
        self.sim.use_newton_actuators = True
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.support_foot_ground_contact is not None:
            self.scene.support_foot_ground_contact.update_period = self.sim.dt
        if self.scene.self_collision is not None:
            self.scene.self_collision.update_period = self.sim.dt
        # MicroDuck stands 0.13 m tall, so the stock viewer distance frames empty ground.
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.8, 0.8, 0.4), lookat=(0.0, 0.0, 0.1))
