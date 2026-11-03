# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-commanded crouch-and-glide behavior on passive roller skates."""

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
# The cycle
##

MICRODUCK_CROUCH_PERIOD = 5.0
"""Length [s] of one crouch-glide cycle."""

MICRODUCK_CROUCH_DESCENT_END = 0.10
MICRODUCK_CROUCH_HOLD_END = 0.50
MICRODUCK_CROUCH_RISE_END = 0.60
"""Segment boundaries of the cycle, as fractions of the period.

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
"""Forward speed [m/s] the robot is spawned rolling at.

The trick is a glide, so the momentum has to be there before the fold starts. See
:attr:`EventsCfg.reset_base` for why it is injected through the root reset and not through a push.
"""

##
# The two keyframes
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
"""Crouch keyframe [rad].

The neck-pitch target 1.0937 rad exceeds its 1.0472 rad upper stop, and the right-knee target
-1.5907 rad exceeds the -1.5708 rad lower stop. Pose rewards retain residual error at those
limits and may command the joints against their stops. The leg targets are asymmetric.
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
    """Convert a PPO iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP.

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
    """Reward terms for the MDP.

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

    upright = RewTerm(
        func=mdp.upright_selected_body, weight=2.0, params={"std": math.sqrt(0.2), "asset_cfg": _TRUNK_BODY_CFG}
    )
    body_ang_vel = RewTerm(func=mdp.body_ang_vel_xy_l2_raw, weight=-0.05, params={"asset_cfg": _TRUNK_BODY_CFG})
    angular_momentum = RewTerm(func=mdp.angular_momentum_l2_raw, weight=-0.02)
    # The live value is the one stated.
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
    """Configuration for events."""

    def __post_init__(self):
        # Entry momentum, injected through the root reset. **Not** through a reset-mode push: a push *adds* to the
        # current root velocity, which on an environment that has already diverged sends the free joint to
        # NaN.
        self.reset_base.params["velocity_range"] = {"x": MICRODUCK_ENTRY_VELOCITY_X}


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.

    Three schedules, the leanest in the family: one reward ramp and the two centre-of-mass ramps the
    whole family shares. There is deliberately no wheel-friction schedule -- see :class:`EventsCfg`
    for what that costs and why it is reproduced.
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
    """MicroDuck crouch-glide trick environment on flat ground."""

    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Reserve contact and constraint headroom for simultaneous body-ground contact.

        # Constraint limits are per environment; contact storage is shared across environments.
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 176
        newton_mjwarp.solver_cfg.nconmax = 36
        self.sim.physics.default = newton_mjwarp

        # The robot folds to about 0.078 m at the bottom of the trick, so the viewer looks lower than
        # the skating task's.
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.8, 0.8, 0.35), lookat=(0.0, 0.0, 0.08))
