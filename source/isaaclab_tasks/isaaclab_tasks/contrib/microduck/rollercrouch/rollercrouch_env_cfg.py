# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Crouch-glide trick environment for the roller-skating Pollen Robotics MicroDuck biped.

Ported from `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_,
which trains MicroDuck on mjlab. Every value taken from upstream cites a section of
``historical tasks3 source extraction``, the extraction of the pinned upstream
checkouts; the sections that describe the roller recipe live in the companion
``historical tasks2 source extraction`` and are cited as "addendum section N".

The trick is a *shoot-the-duck*: the robot arrives rolling, folds into a deep crouch, holds it for
two seconds while it glides on the momentum it brought, and stands back up -- four segments on a
five-second clock, run four times per twenty-second episode. The phase of that clock rides in the
three-wide twist slot of the shared 61-wide observation, exactly as the ground-pick gesture's does,
so the policy is told where in the trick it is and nothing it does can move the clock.

Two structural properties follow and shape the whole recipe:

* **The pose is the objective.** There is no height trapezoid and no velocity target: two terms score
  the fourteen servos against a phase-blended interpolation between a standing keyframe and a
  crouched one, and everything else shapes how the robot gets there. Both keyframes were read off the
  physical robot rather than designed in simulation, which is where this task's two known defects
  come from -- see :data:`MICRODUCK_CROUCH_POSE` and :data:`MICRODUCK_STAND_POSE`.
* **The momentum is an input, not an output.** :attr:`EventsCfg.reset_base` spawns the robot already
  rolling at 0.2-0.5 m/s and :attr:`RewardsCfg.forward_speed` pays for keeping it. The entry velocity
  is injected through the root reset rather than through a reset-mode push, and that is a fixed
  regression rather than a preference: a push adds to the *current* root velocity, which on a
  diverged environment sends the free joint to NaN. Upstream locks the same distinction with its own
  regression test, and its spin task restates the warning verbatim (section 8.7).

This environment is built as a delta on
:class:`~isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg.MicroDuckVelocityRollersFlatEnvCfg`,
where upstream rebuilds it from the raw mjlab template and states the randomization suite, the
sensors and both observation groups again by hand. The extraction verified that what upstream states
again *is* the roller recipe term for term (section 8.7), so the two constructions agree in value and
differ only in how much is restated; the parity tests transcribe upstream's tables independently, so
an inherited value that drifted from upstream's would fail rather than agree with itself.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.contrib.microduck.mdp as mdp
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import (
    MICRODUCK_FOOT_BODY_NAMES,
    MICRODUCK_FOOT_NORMAL_AXIS,
    MICRODUCK_TIRE_BODY_NAMES,
    MICRODUCK_TIRES_PER_FOOT,
    MicroDuckVelocityRollersFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import EventsCfg as RollersEventsCfg
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    MICRODUCK_HEAD_JOINT_NAMES,
    MICRODUCK_JOINT_NAMES,
    MICRODUCK_STEPS_PER_ITERATION,
    MICRODUCK_TRUNK_BODY_NAME,
)

##
# The cycle (section 8.1)
##

MICRODUCK_CROUCH_PERIOD = 5.0
"""Length [s] of one crouch-glide cycle.

At the inherited twenty-second episode this is four complete cycles. It is also a **deployment
contract**: upstream's runtime is launched with a matching ``--ground-pick-period``, and a policy
trained on a different clock reads a phase that does not mean what it meant in training.
"""

MICRODUCK_CROUCH_DESCENT_END = 0.10
MICRODUCK_CROUCH_HOLD_END = 0.50
MICRODUCK_CROUCH_RISE_END = 0.60
"""Segment boundaries of the cycle, as fractions of the period (section 8.1).

At a five-second period they are a 0.5 s descent, a 2.0 s crouched glide, a 0.5 s rise and a 2.0 s
standing rest. The two long segments are the ones the trick is judged on; the two short ones are the
transitions between them.
"""

MICRODUCK_CROUCH_POSE_STD = 0.4
"""Per-joint Gaussian tolerance [rad] on the blended pose target.

Generous on purpose: the knees travel about 1.5 rad between the two keyframes, so a tight kernel
would be flat across most of the fold and leave no gradient to descend.
"""

MICRODUCK_CROUCH_LEAN_PITCH = 0.08
"""Sine of the forward trunk lean asked for during the crouch, about 4.6 degrees."""

MICRODUCK_ENTRY_VELOCITY_X = (0.2, 0.5)
"""Forward speed [m/s] the robot is spawned rolling at (section 8.7).

The trick is a glide, so the momentum has to be there before the fold starts. See
:attr:`EventsCfg.reset_base` for why it is injected through the root reset and not through a push.
"""

##
# The two keyframes (section 8.2)
##

MICRODUCK_STAND_POSE = {
    "left_hip_yaw": -0.0476,
    "left_hip_roll": -0.0629,
    "left_hip_pitch": -0.2869,
    "left_knee": 0.9618,
    "left_ankle": 1.1674,
    "neck_pitch": 0.6029,
    "head_pitch": 0.543,
    "head_yaw": -0.069,
    "head_roll": -0.0414,
    "right_hip_yaw": -0.0337,
    "right_hip_roll": -0.0061,
    "right_hip_pitch": 0.1534,
    "right_knee": -0.9725,
    "right_ankle": -1.0646,
}
"""The standing keyframe [rad] the cycle departs from and returns to, read off the physical robot.

**KNOWN-STALE COMMENTS, POSE REPRODUCED VERBATIM.** Upstream's two comments on this dictionary claim
it is the simulator's HOME pose and ask that it be kept close to HOME, so that the deployed runtime
can hand back cleanly to the roller policy, which restarts from HOME. Measured on the pinned roller
MJCF, neither is true (section 13.7): the knees are 55 and 56 degrees off HOME, the ankles 41 and 35
degrees, and the pose stands the trunk at 0.1133 m against HOME's 0.1407 m -- a 27 mm crouch. The
values are what the deployed policy was trained against and are reproduced unchanged; only the
claims about them are not carried over.

The consequence is visible at phase 0: :attr:`EventsCfg.reset_base` spawns the trunk in the roller
task's HOME-height band, about 25 mm above the pose the reward immediately asks for, so every episode
opens with an unrewarded settle.
"""

MICRODUCK_CROUCH_POSE = {
    "left_hip_yaw": -0.0184,
    "left_hip_roll": 0.0307,
    "left_hip_pitch": 1.4082,
    "left_knee": 1.5248,
    "left_ankle": -0.0675,
    "neck_pitch": 1.0937,
    "head_pitch": 1.2149,
    "head_yaw": -0.0184,
    "head_roll": -0.0368,
    "right_hip_yaw": 0.0184,
    "right_hip_roll": -0.0169,
    "right_hip_pitch": -1.4757,
    "right_knee": -1.5907,
    "right_ankle": 0.0568,
}
"""The crouched keyframe [rad] the cycle folds into, read off the physical robot.

**KNOWN-DEFECT POSE, REPRODUCED VERBATIM (upstream issue draft 018).** Two of these targets lie
outside the compiled model's hard joint limits, measured on the pinned roller MJCF (section 13.6):

* ``neck_pitch`` at 1.0937 rad against an upper stop of 1.0472 rad -- 2.7 degrees past it, and
  0.18 rad past the *soft* bound the 0.9 limit factor imposes;
* ``right_knee`` at -1.5907 rad against a lower stop of -1.5708 rad -- 1.1 degrees past it.

Both pose rewards therefore charge a residual on those two joints that no policy can zero, so the
"hold the crouch" reward can never saturate and the policy is asked to drive two joints into their
stops with a low-stiffness servo. The magnitudes are small -- the Gaussian loses about 1.4 % and the
L1 charges a constant ``0.047/14`` per step -- so this is a transfer hazard rather than a training
blocker, which is why it is reproduced rather than clamped: the deployed policy was trained against
these targets, and re-deriving the pose in simulation is a retune with its own training run.

The pose is also **not left-right mirrored** (``|left_knee|`` 1.5248 against ``|right_knee|`` 1.5907,
``left_hip_pitch`` 1.4082 against ``|right_hip_pitch|`` 1.4757), which is what a raw capture from a
physical robot looks like.

``test_microduck_rollercrouch_env.py`` reads the *compiled model's* joint limits and records the
violation, so the defect is pinned by measurement rather than by a transcribed constant.
"""

##
# Scene entity selections
##

_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)
_HEAD_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)
_TRUNK_BODY_CFG = SceneEntityCfg("robot", body_names=[MICRODUCK_TRUNK_BODY_NAME])
_FOOT_BODY_CFG = SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES, preserve_order=True)
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=MICRODUCK_TIRE_BODY_NAMES, preserve_order=True)
_SELF_COLLISION_SENSOR_CFG = SceneEntityCfg("self_collision")

_POSE_SEGMENTS = {
    "descent_end": MICRODUCK_CROUCH_DESCENT_END,
    "hold_end": MICRODUCK_CROUCH_HOLD_END,
    "rise_end": MICRODUCK_CROUCH_RISE_END,
}
"""The segment boundaries the three phase-gated terms share, spelled once."""


def _iterations(count: int) -> int:
    """Convert an upstream PPO iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP (section 8.6).

    One command, and it is a clock rather than a velocity: the twist slot carries the cycle phase as
    ``(cos, sin, 0)``, so the wrap from the end of a cycle to its start is continuous rather than the
    largest jump in the observation. Nothing the robot does moves it.

    ``randomize_phase`` is **False**, unlike the ground-pick task's. Every episode starts standing at
    phase 0, which is what the deployed runtime does -- a button press launches the cycle from the
    standing station -- and it is also what stops the policy learning "stay low" from spawns that are
    already low.
    """

    base_velocity = mdp.GroundPickPhaseCommandCfg(
        asset_name="robot",
        resampling_time_range=(MICRODUCK_CROUCH_PERIOD, MICRODUCK_CROUCH_PERIOD),
        heading_command=False,
        debug_vis=False,
        period=MICRODUCK_CROUCH_PERIOD,
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
    """Reward terms for the MDP (section 8.4).

    Twelve terms, the leanest recipe in the MicroDuck family. Upstream keeps four of the mjlab base
    template's -- :attr:`upright`, :attr:`body_ang_vel`, :attr:`angular_momentum` and
    :attr:`action_rate_l2` -- through a ``keep`` set, which is what also removes the
    ``dof_pos_limits`` penalty every ``del``-list task silently inherits. The other eight are the
    trick.

    Read them in three groups:

    * **The pose.** :attr:`crouch_glide_pose` and :attr:`crouch_glide_pose_l1` are the same
      phase-blended target at a Gaussian and an L1 width. The Gaussian saturates once the fold is
      roughly right and the L1 carries a constant gradient the rest of the way, which is what keeps
      a shallow crouch from being a comfortable local optimum.
    * **The glide.** :attr:`forward_speed` pays for rolling forward at every phase, crouch included,
      and is the only reason the fold is a glide rather than a squat. :attr:`crouch_forward_lean`
      asks the trunk to lean into it while the crouch is held.
    * **The regularizers**, which are the roller family's: flat blades, no self-contact, a quiet neck
      and a torque budget.

    Sign convention: :attr:`crouch_glide_pose_l1` negates itself and therefore takes a **positive**
    weight.
    """

    ##
    # Base-template terms upstream keeps (section 8.4).
    ##

    # ``std`` is the base template's ``sqrt(0.2)``: upstream narrows it for MicroDuck in the walking
    # recipe only, and this task re-derives from the template and never revisits it.
    upright = RewTerm(
        func=mdp.upright_selected_body, weight=2.0, params={"std": math.sqrt(0.2), "asset_cfg": _TRUNK_BODY_CFG}
    )
    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2_raw, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2_raw, weight=-0.02)
    # Upstream declares -1.0 here and -0.5 at stage 0 of the curriculum that owns this weight, and
    # the curriculum manager runs before the first reward evaluation, so the declared literal is dead
    # (section 13.12). The live value is the one stated.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.5)

    ##
    # The pose: a phase-blended interpolation between the two keyframes.
    ##

    crouch_glide_pose = RewTerm(
        func=mdp.crouch_glide_pose_gaussian,
        weight=6.0,
        params={
            "command_name": "base_velocity",
            "crouch_pose": MICRODUCK_CROUCH_POSE,
            "stand_pose": MICRODUCK_STAND_POSE,
            "std": MICRODUCK_CROUCH_POSE_STD,
            "asset_cfg": _SERVO_JOINT_CFG,
            **_POSE_SEGMENTS,
        },
    )
    # Negates itself, hence the positive weight.
    crouch_glide_pose_l1 = RewTerm(
        func=mdp.crouch_glide_pose_l1,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "crouch_pose": MICRODUCK_CROUCH_POSE,
            "stand_pose": MICRODUCK_STAND_POSE,
            "asset_cfg": _SERVO_JOINT_CFG,
            **_POSE_SEGMENTS,
        },
    )

    ##
    # The glide.
    ##

    # Command-independent by design: on a phase command there is no speed to gate on, and the trick
    # needs the momentum to survive the fold rather than to be re-earned after it.
    forward_speed = RewTerm(func=mdp.forward_speed_reward, weight=1.0, params={"vel_ref": 0.2})
    crouch_forward_lean = RewTerm(
        func=mdp.crouch_forward_lean,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "target_pitch": MICRODUCK_CROUCH_LEAN_PITCH,
            "std": 0.1,
            "asset_cfg": _TRUNK_BODY_CFG,
            **_POSE_SEGMENTS,
        },
    )

    ##
    # The roller family's regularizers.
    ##

    # Gated per foot by that foot's own contact, as on the skating task.
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
        func=mdp.self_collision_cost_saturable, weight=-1.0, params={"sensor_cfg": _SELF_COLLISION_SENSOR_CFG}
    )
    neck_action_rate_l2 = RewTerm(
        func=mdp.joint_action_rate_l2,
        weight=-0.5,
        params={"action_name": "joint_pos", "asset_cfg": _HEAD_JOINT_CFG},
    )
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-3, params={"asset_cfg": _SERVO_JOINT_CFG})


@configclass
class EventsCfg(RollersEventsCfg):
    """Configuration for events (section 8.7).

    The roller task's randomization suite, which is what upstream states again here term for term, with
    one change: the robot is spawned already rolling forwards.

    :attr:`randomize_wheel_friction` is inherited **degenerate** -- its range is ``(0.0, 0.0)`` and,
    unlike on the skating task, no curriculum ramps it. Upstream copied the event and not the
    schedule, so this environment trains on perfectly frictionless bearings for its whole run
    (section 13.3, upstream issue draft 017). That is reproduced verbatim rather than fixed: the
    deployed policy was trained on free bearings, and a glide is exactly the behaviour a bearing-drag
    change would alter. It is a sim-to-real optimism in the degree of freedom the trick depends on,
    not a training defect, and correcting it is a retune with its own run.

    Note:
        Upstream's own ``expand_bam_friction_fields`` startup event is missing from this environment
        and from the spin one, and the invariant it protects is met only by accident: the degenerate
        wheel event above is upstream's *sole* declarer of ``dof_frictionloss``, so deleting it as
        dead code would make upstream's BAM actuator raise at the first multi-environment step
        (section 13.1). That interlock does not exist in this port, because Isaac Lab's BAM actuator
        owns per-environment friction storage unconditionally rather than expanding a shared model
        field -- which is why the event has no counterpart in any MicroDuck task here. The wheel event
        is kept for its own sake, and ``test_microduck_rollercrouch_env.py`` pins both halves: that
        the per-environment storage exists without it, and that it is nonetheless still registered.
    """

    def __post_init__(self):
        # Entry momentum, injected through the root reset. **Not** through a reset-mode push: a push
        # *adds* to the current root velocity, which on an environment that has already diverged
        # sends the free joint to NaN. Upstream fixed exactly this and locks it with a regression
        # test, and its spin task restates the warning verbatim (section 8.7).
        self.reset_base.params["velocity_range"] = {"x": MICRODUCK_ENTRY_VELOCITY_X}


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP (section 8.9).

    Three schedules, the leanest in the family: one reward ramp and the two centre-of-mass ramps the
    whole family shares. There is deliberately no wheel-friction schedule -- see :class:`EventsCfg`
    for what that costs and why it is reproduced.

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
class MicroDuckRollerCrouchFlatEnvCfg(MicroDuckVelocityRollersFlatEnvCfg):
    """MicroDuck crouch-glide trick environment on flat ground.

    The scene, the sensors, the action space, both observation groups and the terminations are the
    roller task's, which is what upstream's standalone rebuild arrives at term for term (section 8.7,
    8.8 and 11). What this class carries is the trick: a phase clock instead of a throttle, a pose
    objective instead of a stride, entry momentum at the reset, and three curricula instead of four.
    """

    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Measured, not inherited: the crouch folds the knees to about 1.5 rad and drops the trunk to
        # a measured 0.078 m, which is a contact set the skating profile does not cover. Profiled
        # under random actions with the tilt termination removed and the pushes forced to full
        # magnitude, so the robots sprawl and every collider reaches the floor, at 256, 2048 and 4096
        # environments: **90 constraints and 29 contacts** per environment at the peak, against the
        # skating task's 83 and 26 on the same model. The three scales agree to one contact, which is
        # what says the tail has been sampled. Logs:
        # ``artifacts/microduck/profile_microduck_contacts_rollercrouch_{256,2048,4096}envs.log``,
        # from ``artifacts/microduck/profile_microduck_contacts.py``.
        #
        # ``njmax`` is a hard per-environment cap and carries the wider margin; ``nconmax`` is a
        # per-environment share of one shared buffer and cannot overflow at the measured peak, so it
        # sits just above it, at the same 1.2x the rest of the family uses.
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 176
        newton_mjwarp.solver_cfg.nconmax = 36
        self.sim.physics.default = newton_mjwarp

        # The robot folds to about 0.078 m at the bottom of the trick, so the viewer looks lower than
        # the skating task's.
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.8, 0.8, 0.35), lookat=(0.0, 0.0, 0.08))
