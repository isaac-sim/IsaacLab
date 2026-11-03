# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ball-kick environment for the Pollen Robotics MicroDuck biped.

Ported from `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_,
which trains MicroDuck on mjlab. Every value taken from upstream cites a section of
``historical tasks3 source extraction``, the extraction of the pinned upstream
checkouts; the sections that describe the shared mjlab base template live in the companion
``historical original source extraction`` and are cited as "reference section N".

The robot starts standing at a random heading with a ball on the ground just in front of one foot,
and has five seconds to kick it away at about a walking pace and stay on its feet. There is no
command, no phase clock and no curriculum on the kick itself: the kick reward is live from the first
step and an earlier kick collects more rolling reward, so the policy kicks immediately.

Three things make this task structurally unlike the rest of the family:

* **It is the only two-entity environment.** The scene carries
  :data:`~isaaclab_tasks.contrib.microduck.assets.MICRODUCK_BALL_CFG` next to the robot, and a reset event places
  the ball in
  the robot's own yaw frame after the robot's pose is settled. That event ordering is behaviour, not
  housekeeping -- see :class:`EventsCfg`.
* **The actor is blind to the ball.** The deployed robot has no ball sensing, so the ball reaches
  the critic only and the actor keeps the family's 61-wide deploy contract byte for byte. The policy
  learns to kick from proprioception and from where the ball reliably *is* at reset.
* **The task is one-footed on purpose.** ``KICK_FOOT`` picks the swinging foot and the other one is
  rewarded for staying planted, which is why left-right symmetry is off here where the forward-roll
  task turns it on.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
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

import isaaclab_tasks.contrib.microduck.pollen_mdp as mdp
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
# Which foot kicks (addendum section 6.1)
##

MICRODUCK_KICK_FOOT = "right"
"""The foot that swings at the ball.

Upstream asserts this is ``"right"`` or ``"left"`` and ships ``"right"``; the mirrored task is a
different policy, not a different episode, which is also why left-right symmetry stays off here. The
support foot is the other one, and everything below is derived from this constant rather than
restated, so flipping it flips the ball offset, the support sensor and the run name together.
"""

MICRODUCK_SUPPORT_FOOT = "left" if MICRODUCK_KICK_FOOT == "right" else "right"
"""The foot that must stay planted through the kick."""

##
# Ball placement (addendum sections 2.3 and 6.1)
##

MICRODUCK_BALL_OFFSET_X = 0.09
"""Forward offset [m] of the ball centre from the robot root at reset.

Upstream's own justification reproduces exactly against the model. The toe tip of a foot collider
reaches x = 0.0340 m at the stand pose, and with the 0.035 m ball radius and the +/-0.015 m noise
below the ball's rear surface never comes closer than x = 0.040 m, so the spawn keeps **6 mm** of
clearance in the worst case. What matters when re-tuning is that margin rather than the literal
0.09: upstream records that ``0.08 +/- 0.02`` penetrated the toe at reset and the solver ejected the
ball, paying a kick reward for no kick.
"""

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

    **Upstream's own commentary disagrees with this constant, and the constant is what is ported.**
    ``microduck_ball_kick_env_cfg.py:223-237`` describes a landscape "peaking at BALL_TARGET_SPEED
    (0.25 m/s -- a gentle tap)", says "Weight 12.0 = 3.0/target so the at-target payoff stays
    approximately +3/step", and says the net reward "only hits 0 at ~1.0 m/s, 4x the target". All
    three statements are consistent with a target of 0.25 and none of them with the 1.0 the file
    actually sets (``:95``); the target was raised and the weights were not rescaled, which
    ``:90-94`` explicitly warns against ("if you change the target, rescale the weights with it").

    At the shipped constants the kick pays ``12 * min(v, 1.0) - 4 * max(v - 1.0, 0)``: **+12/step**
    at target, zero at 4.0 m/s and a floor of -8/step past 6.0 m/s. That is four times the intended
    at-target payoff and roughly doubles the task's reward mass, so the shared regularizers act at
    about half their intended relative strength while the ball rolls.

    The code is reproduced rather than the prose, because the shipped weights are what upstream's
    runs were trained against; the discrepancy is reported upstream separately. Rescaling
    :attr:`RewardsCfg.ball_forward_velocity` to 3.0 and
    :attr:`RewardsCfg.ball_speed_overshoot` to 1.0 restores the documented landscape.
"""

MICRODUCK_STAND_HEIGHT = 0.115
"""Trunk height [m] the robot is asked to hold through the kick.

The same value the stand-up and sit-stand tasks use: 2 mm below the 0.11718 m the model reaches
geometrically at the stand pose with the soles down, which is the servos sagging under load.
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
"""Upper bound [deg] on the IMU mounting-misalignment angle, upstream's velocity-matched value."""

_IMU_DELAY_UPDATE_PERIOD = 64
"""Control steps between two draws of the IMU latency (reference section 8)."""


def _iterations(count: int) -> int:
    """Convert an upstream PPO iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


##
# Physics preset
##


@configclass
class MicroDuckBallKickPhysicsCfg(PresetCfg):
    """Backend preset for the MicroDuck ball-kick environment.

    MicroDuck is trained on MuJoCo upstream, so MJWarp is the reference backend and the only one
    offered; it is also the only backend that can run this task as configured, because the
    environment sets ``sim.use_newton_actuators = True`` and the solver-hosted BAM model is rejected
    on the PhysX family's host adapter. See
    :class:`~isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg.MicroDuckPhysicsCfg`.

    This is the one task in the family whose contact budget upstream *does* revisit -- it raises
    ``nconmax`` from the template's 35 to 50 for the ball's own contacts (addendum section 6.2) --
    and, as on its siblings, the port measures rather than transcribes.
    """

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            # Measured, not inherited. Profiling under random actions -- the regime where the
            # robots stumble, sprawl on the floor and grind the ball into every collider, with the
            # tilt termination dropped and the pushes forced to full magnitude -- peaks at
            # **30 contacts and 86 constraints** per environment. The profile was run at both 256
            # and 2048 environments and the two agree exactly, which is what says the tail has been
            # sampled. Logs:
            # ``artifacts/microduck/profile_microduck_contacts_ballkick_{256,2048}envs.log``, from
            # ``artifacts/microduck/profile_microduck_contacts.py``.
            #
            # That is three contacts above the stand-up task's peak on the same robot: the ball adds
            # its own ground contact plus whatever it is pressed against. ``njmax`` is a hard
            # per-environment cap and carries the margin; ``nconmax`` is a per-environment share of
            # one shared buffer and cannot overflow at the measured peak, so it sits just above it.
            njmax=128,
            nconmax=36,
            # upstream's flat solver profile, which this task inherits unchanged -- it raises only
            # the contact budget, not the iteration counts
            iterations=10,
            ls_iterations=20,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        # upstream steps MuJoCo once per 0.005 s physics tick (reference section 1)
        num_substeps=1,
        default_shape_cfg=NewtonShapeCfg(margin=0.0),
    )
    default = newton_mjwarp


##
# Scene definition
##


@configclass
class MicroDuckBallKickSceneCfg(InteractiveSceneCfg):
    """Scene with the all-collisions MicroDuck robot, a ball, and a ground plane.

    Upstream requires the robot to be the *first* scene entity, because its reset events write
    absolute ``qpos`` columns and adding the ball ahead of the robot would shift them (addendum
    section 13.23). Isaac Lab addresses each asset through its own view, so the ordering constraint
    does not carry over -- but the ball is still declared after the robot, because the reset event
    that places it reads the robot's settled pose.
    """

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

    # The all-collisions model, as upstream uses here. The kick is a standing task, so the extra
    # colliders matter less than they do on the stand-up and forward-roll tasks -- but a robot that
    # loses its balance mid-swing lands on them, and the self-collision penalty needs them to exist.
    robot = MICRODUCK_ALLCOLLISIONS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # The prop. Its position here is only what it has before the first reset; ``reset_ball`` places
    # it in the robot's yaw frame every episode.
    ball = MICRODUCK_BALL_CFG.replace(prim_path="{ENV_REGEX_NS}/Ball")

    # Feet and trunk, as every task in the family does: upstream tracks ground contact on the two
    # soles and drops the base template's terrain and foot-height scanners wholesale.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/(ankle_left|ankle_right))?",
        history_length=3,
        track_air_time=True,
    )

    # The balance signal, and the one sensor this task adds. Upstream matches the support foot's
    # sole *geom* against the terrain *body* and reads whether its single contact slot found
    # anything; this is that sensor, on the Newton backend's shape-level expressions.
    #
    # Filtering against the terrain rather than reading a net force is load-bearing here in a way it
    # is not for the family's other foot sensors: the ball rolls along the ground into the robot's
    # feet, so an unfiltered sole would report "grounded" while the support foot was airborne and
    # merely touching the ball -- exactly the state the term exists to deny.
    #
    # The terrain expression is absolute rather than ``{ENV_REGEX_NS}``-relative because the ground
    # plane is a single shape shared by every environment.
    support_foot_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        sensor_shape_prim_expr=[_SUPPORT_FOOT_COLLIDER_SHAPE_EXPR],
        filter_shape_prim_expr=[f"{_TERRAIN_PRIM_PATH}/.*"],
    )

    # Self-collision sensor, identical to the stand-up and forward-roll tasks': the model's ten
    # enabled colliders against each other, many-to-many, which is upstream's trunk-subtree-against-
    # itself sensor. The reward saturates it back to upstream's 0-or-1 scale.
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
    """Command specifications for the MDP (addendum section 6.5).

    The kick takes no command: it is triggered at deployment by the policy switch, and there is
    nothing to steer once it starts. The twist survives only so the deployed 61-wide observation
    keeps its three-wide slot, and the head and body slots are zero padding rather than commands
    (see :class:`ObservationsCfg`).
    """

    # Ranges a hundredth of the velocity task's, resampled at most once per episode. Nothing reads
    # it. ``rel_forward_envs = 0.2`` is inherited from upstream's base template and never overridden
    # there (addendum section 7.22): a fifth of the resamples get a commanded surge of 0.3 m/s that
    # no reward acts on, which is a quirk of the observation distribution rather than of the
    # behaviour, and is reproduced rather than quietly cleaned up.
    base_velocity = mdp.MicroDuckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_forward_envs=0.2,
        rel_turn_in_place_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.MicroDuckVelocityCommandCfg.Ranges(
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

    joint_pos = mdp.BiasedJointPositionActionCfg(
        asset_name="robot",
        joint_names=MICRODUCK_JOINT_NAMES,
        preserve_order=True,
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP (addendum sections 6.4 and 11).

    The actor group is byte-for-byte the velocity task's 61-wide deploy contract, which is the whole
    point of the family: one runtime on the robot feeds every MicroDuck policy from the same buffer.

    **The actor never sees the ball**, and that is upstream's design rather than an omission: the
    real robot has no ball sensor, so a policy trained on ball state could not be deployed. The
    critic does see it, which is the asymmetric half of the actor-critic and is what lets the value
    function predict the kick payoff.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group: the 61-wide deploy contract.

        See :class:`~isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg.ObservationsCfg` for
        what each corruption models; the terms, their order, their noise and their delays are
        upstream's velocity values, which this task copies deliberately for sim-to-real parity.
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
            func=mdp.joint_pos_rel_biased,
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
        """Privileged observations for the value function (addendum section 11.2).

        The actor's terms with every corruption removed, plus the base linear velocity, the three
        foot terms the robot has no sensor for, and the two ball terms that make this critic the
        widest in the family at 80.

        The two sensor-derived foot terms are the NaN-guarded variants, as on the stand-up and
        forward-roll ports and unlike upstream here; see :class:`TerminationsCfg` for why.
        """

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel_biased, params={"asset_cfg": _SERVO_JOINT_CFG, "biased": False})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _SERVO_JOINT_CFG})
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        foot_air_time = ObsTerm(func=mdp.foot_air_time_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact = ObsTerm(func=mdp.foot_contact, params={"sensor_cfg": _FOOT_SENSOR_CFG})
        foot_contact_forces = ObsTerm(func=mdp.foot_contact_forces_safe, params={"sensor_cfg": _FOOT_SENSOR_CFG})
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
    """Configuration for events (addendum section 6.6).

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
        func=mdp.randomize_encoder_bias,
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

    # Upstream also randomizes the base height and yaw here, and ``set_ground_state`` then
    # overwrites both. Only the horizontal spread is live, so only the horizontal spread is
    # configured.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}, "velocity_range": {}},
    )

    # Non-zero here, unlike every other task in this batch. Upstream's reason is a deployment one:
    # the ball-kick policy is handed off from the walking or velocity-stand policy, whose settled
    # stand does not reproduce the stand pose exactly, so the kick has to start from a stand it did
    # not choose.
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
            # Upstream spells this ``sitting_tilt_max`` and applies it to the standing bucket too --
            # the two share one upright orientation sampler -- so at ``standing_prob = 1.0`` it is
            # the +/-5 degrees of pitch and roll the stand spawns with.
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
    """Reward terms for the MDP (addendum section 6.3).

    Twelve terms, the smallest stack in the family, and they split cleanly in two:

    * **The kick**, ``ball_forward_velocity`` and ``ball_speed_overshoot``, which together form a
      one-sided plateau around :data:`MICRODUCK_BALL_TARGET_SPEED`. Read
      :data:`MICRODUCK_BALL_TARGET_SPEED` before re-tuning either weight -- upstream's own
      commentary describes a different landscape from the one its constants produce.
    * **The stand**, everything else: hold the legs and the neck at the stand pose, hold the trunk
      at :data:`MICRODUCK_STAND_HEIGHT` and upright, and keep the support foot planted. The kick is
      worth up to +12 per step against a standing stack of 8, so this half is what stops the policy
      from throwing itself at the ball.

    ``dof_pos_limits`` is upstream's silently inherited regularizer: it is never mentioned in the
    ball-kick configuration and survives only because it is not in the deletion list (addendum
    section 13.18), so it is stated here rather than left to be discovered.
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
    # Tighter than the legs': the head is not part of the kick, and upstream holds it still rather
    # than steering it. There is no head-pose command on this task at all.
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
        func=mdp.upright,
        weight=2.0,
        params={"std": math.sqrt(0.05), "asset_cfg": _TRUNK_BODY_CFG},
    )

    ##
    # Sim-to-real regularizers, matched to the velocity task.
    ##

    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2, weight=-0.02)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # Ramped to -1.0 by iteration 1500. Upstream's own note is a useful tuning hint: the kick is a
    # fast one-shot swing, so if the converged kick is too weak, softening the ramp end from -1.0 to
    # -0.6 is the first knob to try.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    # ``saturate`` keeps the many-to-many sensor on upstream's 0/1 scale, so the weight is the
    # penalty for touching yourself at all rather than a per-collider tariff.
    self_collisions = RewTerm(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG, "saturate": True},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP (addendum section 6.7).

    Unlike the stand-up and forward-roll tasks, the tilt termination is **kept**: the robot starts
    standing and is meant to still be standing when the episode ends, so falling over is a failure
    rather than a phase of the manoeuvre.

    Upstream also inherits a terrain-bounds termination, which returns all-false on a ground plane
    and is therefore dead on this flat-only task (addendum section 7.24); it is not carried over.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})

    # Catches a *broken* robot rather than a fallen one.
    #
    # Deviation from upstream, deliberately and in line with the stand-up, forward-roll and roller
    # ports: upstream leaves this term's sensor list empty here, which the extraction reads as drift
    # rather than design and recommends closing everywhere in the port (addendum section 14.4). The
    # guard only changes behaviour in states that are already broken, and the matching half of it is
    # the pair of NaN-safe critic terms in :class:`ObservationsCfg`.
    #
    # It reads the *robot*, as upstream's does. The ball is not covered by any state check on either
    # stack, which is why the two ball observations guard themselves.
    nan_state = DoneTerm(
        func=mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("contact_forces",)},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP (addendum section 6.8).

    **Nothing schedules the kick.** ``ball_forward_velocity`` is at full weight from step 0, which is
    upstream's documented intent: the reward is available from t=0 and an earlier kick collects more
    ball-rolling reward, so the policy kicks immediately and the shaping problem is keeping it
    upright afterwards rather than getting it to try.

    What is scheduled is the same domain-randomization and smoothness ramp the rest of the family
    uses. Upstream's schedules are written in PPO iterations; :func:`_iterations` converts them to
    the global environment-step count these terms compare against.
    """

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

    # Upstream drives this one with an exclusive step comparison where its event-parameter helper
    # uses an inclusive one (addendum section 7.6). The inconsistency is reproduced rather than
    # smoothed over, so a stage table transcribed from upstream schedules identically here.
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
    """MicroDuck ball-kick environment on flat ground.

    There is no rough variant, as there is none upstream: the factory takes no terrain argument at
    all, and its registration says why -- a ball on rough terrain is a different task.
    """

    sim: SimulationCfg = SimulationCfg(physics=MicroDuckBallKickPhysicsCfg())
    # ``env_spacing`` is upstream's scene extent (reference section 1).
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
        # 0.005 s physics steps decimated by 4 give the 50 Hz control rate the deployed policy runs
        # at. Episodes are 5 s -- 250 control steps -- which is upstream's "kick, then several
        # seconds of ball-rolling reward, then settle back", and the shortest window in the family.
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 0.005
        # Run the BAM servos through the backend-native path, as the rest of the family does and as
        # upstream does. The decimation above is even, which is what lets the stateful servo delay
        # line be CUDA-graph-captured.
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
