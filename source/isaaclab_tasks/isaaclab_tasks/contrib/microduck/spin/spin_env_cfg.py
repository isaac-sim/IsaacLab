# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spin trick environment for the roller-skating Pollen Robotics MicroDuck biped.

Ported from `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_,
which trains MicroDuck on mjlab. Every value taken from upstream cites a section of
``historical tasks3 source extraction``, the extraction of the pinned upstream
checkouts; the sections that describe the roller recipe live in the companion
``historical tasks2 source extraction`` and are cited as "addendum section N".

The trick is one counter-clockwise turn on the spot: launch, hold about three radians a second,
brake, and rest standing -- four segments on a four-second clock, run five times per twenty-second
episode. The area under that trapezoid is 6.3 rad, which is one turn per cycle to a quarter of a
percent. The phase rides in the three-wide twist slot of the shared 61-wide observation, exactly as
the crouch-glide trick's does.

Structurally this is the crouch-glide task with a different objective, and the two differences that
are not cosmetic are worth stating:

* **The objective is a rate, not a pose.** :attr:`RewardsCfg.spin_rate_track` and
  :attr:`RewardsCfg.spin_rate_l1` score the trunk's yaw rate against the envelope. The rate is read
  in the *body* frame, because that is what the robot's own gyro reports and therefore what the
  policy observes.
* **``angular_momentum`` is dropped, and it is the only task in the batch that drops it.** The term
  charges the norm of the whole angular-momentum vector, so on this task it would fight the
  objective directly. ``body_ang_vel`` survives because it only charges roll and pitch, which damps
  the wobble without touching the rotation (section 10.2).

Everything else is shape. Two shaping terms hint at the mechanism -- a leg scissor and a wheel
differential -- and both are gated by the envelope normalized to ``[0, 1]``, so they fade out with
the rotation and the robot arrives at the standing rest in a neutral station rather than mid-pump.
The scissor is additionally *decayed by curriculum*, which is the only training wheel in the family
that is deliberately removed once it has done its job.

Like the crouch-glide task, this environment is built as a delta on
:class:`~isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg.MicroDuckVelocityRollersFlatEnvCfg`
where upstream rebuilds it from the raw mjlab template; see
:mod:`~isaaclab_tasks.contrib.microduck.rollercrouch.rollercrouch_env_cfg` for why, and for the
wheel-friction interlock the two share.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.microduck.pollen_mdp as mdp
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import (
    MICRODUCK_FOOT_BODY_NAMES,
    MICRODUCK_FOOT_NORMAL_AXIS,
    MICRODUCK_TIRE_BODY_NAMES,
    MICRODUCK_TIRES_PER_FOOT,
    MicroDuckVelocityRollersFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import EventsCfg as RollersEventsCfg
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    MICRODUCK_JOINT_NAMES,
    MICRODUCK_STEPS_PER_ITERATION,
    MICRODUCK_TRUNK_BODY_NAME,
)

##
# The envelope (section 10.1)
##

MICRODUCK_SPIN_PERIOD = 4.0
"""Length [s] of one spin cycle. At the inherited twenty-second episode this is five turns."""

MICRODUCK_SPIN_RATE_MAX = 3.0
"""Peak yaw rate [rad/s] of the envelope's hold segment, counter-clockwise."""

MICRODUCK_SPIN_ACCEL_END = 0.125
MICRODUCK_SPIN_HOLD_END = 0.525
MICRODUCK_SPIN_BRAKE_END = 0.650
"""Segment boundaries of the cycle, as fractions of the period (section 10.1).

At a four-second period they are a 0.5 s launch, a 1.6 s hold, a 0.5 s brake and a 1.4 s standing
rest. The area under the resulting trapezoid is ``2.1 * MICRODUCK_SPIN_RATE_MAX = 6.30 rad``, which
is 1.0027 turns per cycle.
"""

MICRODUCK_SPIN_WHEEL_OMEGA_SCALE = 17.0
"""Wheel-rate difference [rad/s] the differential hint's ``tanh`` saturates near.

**KNOWN-UNREPRODUCIBLE DERIVATION, CONSTANT REPRODUCED VERBATIM.** Upstream derives this from
``2 * rate_max * half_track / wheel_radius`` with a half-track it states as 0.0499 m and a radius of
0.0175 m. Measured on the pinned roller model the half-track is 0.03925 m at the foot sites and
0.04056 m at the tire centres, and the tire radius is 0.0150 m, so neither input reproduces and
neither does the result (section 13.9). The constant is what the deployed policy trained against and
is kept unchanged; only the arithmetic behind it is not carried over.
"""

MICRODUCK_SPIN_LAUNCH_DRIFT_SCALE = 0.2
"""Fraction of the drift cost charged during the launch, so the robot may shuffle to get started."""

MICRODUCK_ENTRY_VELOCITY_X = (0.0, 0.3)
"""Forward speed [m/s] the robot is spawned rolling at (section 10.5).

Wider at the bottom than the crouch-glide task's ``(0.2, 0.5)`` and reaching zero, because the
deployed button can be pressed standing still *or* rolling slowly: the policy has to learn to kill
whatever residual momentum it has before it launches the rotation.
"""

##
# Scene entity selections
##

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_FOOT_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_TIRE_BODY_NAMES, preserve_order=True)
_SELF_COLLISION_SENSOR_CFG = SceneEntityCfg("self_collision")

MICRODUCK_SPIN_NECK_JOINT_NAMES = ["neck_pitch", "head_pitch", "head_roll"]
"""The neck and head joints held near their stand pose -- ``head_yaw`` deliberately excluded.

Upstream leaves the yaw free so it can serve as an inertia flywheel for launching the rotation
(section 10.2), and it is the only task in the family that scopes this penalty at all.
"""

_SPIN_NECK_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_SPIN_NECK_JOINT_NAMES, preserve_order=True)

_LEFT_WHEEL_JOINT_CFG = SceneEntityCfg(
    "robot", joint_names=["passive_LF_wheel", "passive_LR_wheel"], preserve_order=True
)
_RIGHT_WHEEL_JOINT_CFG = SceneEntityCfg(
    "robot", joint_names=["passive_RF_wheel", "passive_RR_wheel"], preserve_order=True
)
"""The two bogies' wheel hinges, averaged separately so their difference is the yaw differential.

Upstream selects the wheels with ``^passive_.*`` here where its sibling tasks use
``^passive_.*wheel``, which its own conventions forbid: on a model carrying backlash hinges the
looser pattern picks those up too (section 13.4). On the plain roller model the two are equivalent
and there is no live defect; naming the four hinges makes the divergence unexpressible rather than
merely harmless, which is what the rest of this port does everywhere.
"""

_LEFT_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=["left_hip_pitch", "left_knee"], preserve_order=True)
_RIGHT_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=["right_hip_pitch", "right_knee"], preserve_order=True)
"""The sagittal leg joints the scissor hint pairs, position by position."""

_SPIN_ENVELOPE = {
    "rate_max": MICRODUCK_SPIN_RATE_MAX,
    "accel_end": MICRODUCK_SPIN_ACCEL_END,
    "hold_end": MICRODUCK_SPIN_HOLD_END,
    "brake_end": MICRODUCK_SPIN_BRAKE_END,
}
"""The envelope the four phase-driven terms share, spelled once."""


def _iterations(count: int) -> int:
    """Convert an upstream PPO iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP (section 10.4).

    The crouch-glide task's clock at a four-second period: the twist slot carries the cycle phase as
    ``(cos, sin, 0)``, nothing the robot does moves it, and ``randomize_phase`` is False so every
    episode starts standing at phase 0 -- which is what the deployed button press does.
    """

    base_velocity = mdp.GroundPickPhaseCommandCfg(
        asset_name="robot",
        resampling_time_range=(MICRODUCK_SPIN_PERIOD, MICRODUCK_SPIN_PERIOD),
        heading_command=False,
        debug_vis=False,
        period=MICRODUCK_SPIN_PERIOD,
        randomize_phase=False,
        ranges=mdp.GroundPickPhaseCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=None,
        ),
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP (section 10.2).

    Fourteen terms. Upstream keeps three of the mjlab base template's through a ``keep`` set --
    :attr:`upright`, :attr:`body_ang_vel` and :attr:`action_rate_l2` -- and **not**
    ``angular_momentum``, which every other task in the batch keeps: it charges the norm of the whole
    angular-momentum vector and would fight a spin head-on, where ``body_ang_vel`` charges roll and
    pitch only and therefore damps the wobble without touching the rotation.

    Read the rest in three groups:

    * **The objective.** :attr:`spin_rate_track` and :attr:`spin_rate_l1` score the trunk's yaw rate
      against the envelope at a Gaussian and an L1 width -- the same two-layer shape the crouch-glide
      task uses on its pose. :attr:`spin_stay_in_place` is what makes it a spin rather than a pivot
      around one skate.
    * **The mechanism hints.** :attr:`spin_wheel_differential` and :attr:`leg_antisymmetry` are the
      two ways a skater produces a rotation, offered as shaping and gated by the envelope so neither
      shapes the standing rest. The scissor is decayed by curriculum once it has done its job.
    * **The regularizers**, which are the roller family's, plus :attr:`spin_grounded` -- both blades
      down while the rotation runs.

    Sign convention: :attr:`spin_rate_l1` and :attr:`leg_antisymmetry` negate themselves and
    therefore take **positive** weights, while :attr:`spin_stay_in_place` returns a cost and takes a
    negative one.
    """

    ##
    # Base-template terms upstream keeps (section 10.2).
    ##

    upright = RewTerm(func=mdp.upright, weight=2.0, params={"std": math.sqrt(0.2), "asset_cfg": _TRUNK_BODY_CFG})
    # Roll and pitch only, which is why this one survives where ``angular_momentum`` does not.
    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    # Upstream declares -1.0 here and -0.5 at stage 0 of the curriculum that owns this weight, and the
    # curriculum manager runs before the first reward evaluation, so the declared literal is dead
    # (section 13.12). The live value is the one stated.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.5)

    ##
    # The objective: track the yaw-rate envelope, on the spot.
    ##

    spin_rate_track = RewTerm(
        func=mdp.spin_rate_track,
        weight=6.0,
        params={"command_name": "base_velocity", "std": 1.5, "asset_cfg": _TRUNK_BODY_CFG, **_SPIN_ENVELOPE},
    )
    # Negates itself, hence the positive weight.
    spin_rate_l1 = RewTerm(
        func=mdp.spin_rate_l1,
        weight=0.5,
        params={"command_name": "base_velocity", "asset_cfg": _TRUNK_BODY_CFG, **_SPIN_ENVELOPE},
    )
    # Returns a cost, hence the negative weight. Raised from -1.0 after a calibration run at 500
    # iterations measured the trunk translating at about 0.35 m/s -- roughly the yaw rate times the
    # half-track, which is the signature of a pivot on one skate rather than a centred spin.
    spin_stay_in_place = RewTerm(
        func=mdp.spin_stay_in_place,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "launch_scale": MICRODUCK_SPIN_LAUNCH_DRIFT_SCALE,
            "accel_end": MICRODUCK_SPIN_ACCEL_END,
            "asset_cfg": _TRUNK_BODY_CFG,
        },
    )

    ##
    # The mechanism hints, both gated by the envelope.
    ##

    spin_wheel_differential = RewTerm(
        func=mdp.spin_wheel_differential,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "left_wheel_cfg": _LEFT_WHEEL_JOINT_CFG,
            "right_wheel_cfg": _RIGHT_WHEEL_JOINT_CFG,
            "omega_scale": MICRODUCK_SPIN_WHEEL_OMEGA_SCALE,
            **_SPIN_ENVELOPE,
        },
    )
    # Negates itself, hence the positive weight. Decayed to 0.25 by iteration 3000 -- the only reward
    # curriculum in the family that ramps *down*, which is deliberate: the hint launches the right
    # mechanism and is then removed so the policy can refine its own pumping frequency.
    leg_antisymmetry = RewTerm(
        func=mdp.leg_antisymmetry,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "left_joint_cfg": _LEFT_LEG_JOINT_CFG,
            "right_joint_cfg": _RIGHT_LEG_JOINT_CFG,
            **_SPIN_ENVELOPE,
        },
    )
    spin_grounded = RewTerm(
        func=mdp.spin_grounded,
        weight=0.5,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT,
            **_SPIN_ENVELOPE,
        },
    )

    ##
    # The roller family's regularizers.
    ##

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
    self_collisions = RewTerm(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG}
    )
    neck_action_rate_l2 = RewTerm(
        func=mdp.joint_action_rate_l2,
        weight=-0.5,
        # the *action* vector's neck columns, which include head_yaw: upstream scopes only the
        # position penalty below, not the rate one
        params={
            "action_name": "joint_pos",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["neck_pitch", "head_pitch", "head_yaw", "head_roll"], preserve_order=True
            ),
        },
    )
    # Scoped to exclude ``head_yaw``, which is the flywheel. This is the only place in the family
    # where this penalty is narrowed (section 10.2).
    neck_joint_pos_l2 = RewTerm(func=mdp.joint_pose_l2, weight=-0.2, params={"asset_cfg": _SPIN_NECK_JOINT_CFG})
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-3, params={"asset_cfg": _SERVO_JOINT_CFG})


@configclass
class EventsCfg(RollersEventsCfg):
    """Configuration for events (section 10.5).

    The roller task's randomization suite with one change: the robot is spawned rolling forwards at
    up to 0.3 m/s, and the band reaches zero -- the deployed button can be pressed from a standstill
    or from a slow roll.

    :attr:`randomize_wheel_friction` is inherited **degenerate**, with no curriculum to ramp it, so
    this environment trains on perfectly frictionless bearings; see
    :class:`~isaaclab_tasks.contrib.microduck.rollercrouch.rollercrouch_env_cfg.EventsCfg` for the
    full account of that defect and of why the event is nonetheless kept.
    """

    def __post_init__(self):
        # Injected through the root reset, not through a reset-mode push: a push *adds* to the
        # current root velocity, which on an environment that has already diverged sends the free
        # joint to NaN. Upstream restates the crouch task's warning verbatim here (section 10.5).
        self.reset_base.params["velocity_range"] = {"x": MICRODUCK_ENTRY_VELOCITY_X}


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP (section 10.7).

    Four schedules: the crouch-glide task's three, plus the decay that removes the leg-scissor hint.
    That decay is worth naming -- it is the family's only reward weight that ramps *down*, and the
    pattern it implements is "training wheel, then removed" rather than "introduce late".

    Upstream's schedules are written in PPO iterations; :func:`_iterations` converts them to the
    global environment-step count these terms compare against.
    """

    action_rate_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": _iterations(0), "weight": -0.5},
                {"step": _iterations(250), "weight": -0.8},
                {"step": _iterations(500), "weight": -1.0},
            ],
        },
    )

    leg_antisym_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "leg_antisymmetry",
            "weight_stages": [
                {"step": _iterations(0), "weight": 1.0},
                {"step": _iterations(1500), "weight": 0.5},
                {"step": _iterations(3000), "weight": 0.25},
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
class MicroDuckSpinFlatEnvCfg(MicroDuckVelocityRollersFlatEnvCfg):
    """MicroDuck spin trick environment on flat ground.

    The scene, the sensors, the action space, both observation groups and the terminations are the
    roller task's, which is what upstream's standalone rebuild arrives at term for term (sections
    10.5, 10.6 and 11).
    """

    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Measured, not inherited: three radians a second about a 39 mm half-track drags four small
        # tire patches across the floor with continuously rotating contact normals, which is the
        # regime the extraction singles out as the one where these budgets matter most -- and the one
        # upstream leaves at the mjlab template's unexamined 35. Profiled under random actions with
        # the tilt termination removed and the pushes forced to full magnitude, so the robots sprawl
        # and every collider reaches the floor, at 256, 2048 and 4096 environments: **90 constraints
        # and 30 contacts** per environment at the peak, against the skating task's 83 and 26 on the
        # same model. Logs:
        # ``artifacts/microduck/profile_microduck_contacts_spin_{256,2048,4096}envs.log``, from
        # ``artifacts/microduck/profile_microduck_contacts.py``.
        #
        # ``njmax`` is a hard per-environment cap and carries the wider margin; ``nconmax`` is a
        # per-environment share of one shared buffer and cannot overflow at the measured peak, so it
        # sits just above it, at the same 1.2x the rest of the family uses.
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 176
        newton_mjwarp.solver_cfg.nconmax = 36
        self.sim.physics.default = newton_mjwarp
