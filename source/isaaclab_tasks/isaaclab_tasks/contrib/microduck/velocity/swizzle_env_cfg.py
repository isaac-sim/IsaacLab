# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Swizzle skating with lateral leg motion, forward-speed and balance rewards."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.contrib.microduck.mdp as mdp
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import (
    MICRODUCK_TIRES_PER_FOOT,
    MicroDuckVelocityRollersFlatEnvCfg,
)
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import CommandsCfg as RollersCommandsCfg
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import CurriculumCfg as RollersCurriculumCfg
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import ObservationsCfg as RollersObservationsCfg
from isaaclab_tasks.contrib.microduck.velocity.rollers_env_cfg import RewardsCfg as RollersRewardsCfg
from isaaclab_tasks.contrib.microduck.velocity.velocity_env_cfg import (
    MICRODUCK_HEAD_JOINT_NAMES,
    MICRODUCK_LEG_JOINT_NAMES,
    MICRODUCK_STEPS_PER_ITERATION,
)

##
# Scene entity selections
##

_LEG_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_LEG_JOINT_NAMES, preserve_order=True)
_HEAD_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_HEAD_JOINT_NAMES, preserve_order=True)
_FOOT_SENSOR_CFG = SceneEntityCfg(
    "contact_forces", body_names=["tire", "tire_2", "tire_3", "tire_4"], preserve_order=True
)

_LEFT_LEG_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"],
    preserve_order=True,
)
_RIGHT_LEG_JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=["right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"],
    preserve_order=True,
)
"""The two legs' joints, paired position by position."""


def _iterations(count: int) -> int:
    """Convert a PPO iteration count into the global environment-step count."""
    return count * MICRODUCK_STEPS_PER_ITERATION


##
# MDP settings
##


@configclass
class CommandsCfg(RollersCommandsCfg):
    """Command specifications for the MDP.

    The stride task's command with both clamps opened, plus a head-pose command:

    The yaw clamp is read once, when the command term is constructed, so it has to be a
    configuration value -- which is why the heading *weights* ramp by curriculum and the range does
    not.
    """

    # Joint-position deltas from the stand pose for (neck_pitch, head_pitch, head_yaw, head_roll).
    # These ranges are the ``head_pose_range`` curriculum's first stage, which holds them tiny until
    # the swizzle is solid and then opens them to the family's full envelope.
    head_pose = mdp.ZeroablePoseDeltaCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015)),
    )

    def __post_init__(self):
        self.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class ObservationsCfg(RollersObservationsCfg):
    """Observation specifications for the MDP.

    The stride task's groups with one term replaced on each: the four zero-padded head-command
    columns now carry the real command. Re-assigning an existing field keeps its declared position,
    so the actor stays the family's 61-wide deploy contract and the critic stays 78 wide -- the swap
    changes the *content* of a slot, not the layout. The body-pose slot stays zero-padded, because
    this task has no trunk-pose command.
    """

    @configclass
    class PolicyCfg(RollersObservationsCfg.PolicyCfg):
        """Observations for the policy group: the 61-wide deploy contract."""

        head_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})

    @configclass
    class CriticCfg(RollersObservationsCfg.CriticCfg):
        """Privileged observations for the value function."""

        head_pose_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg(RollersRewardsCfg):
    """Reward terms for the MDP.

    Eighteen terms: the stride recipe's twenty-one, minus seven, plus four.

    **Removed, because they exist to suppress this gait.** :attr:`single_support` pays for one blade
    down and charges two; :attr:`glide` pays a quiet single-support coast; :attr:`skating_air_time`
    pays each swing; :attr:`gait_symmetry` polices the stride's left-right *alternation*, which is
    the opposite of this gait's simultaneity; :attr:`hip_roll_neutral` closes the splayed stance the
    hourglass needs open. :attr:`braking` goes because a negative throttle now means backwards, and
    :attr:`neck_joint_pos_l2` because it would fight the head command.

    **Added.** :attr:`leg_symmetry` is the gait definition and :attr:`grounded` keeps both blades
    down; :attr:`heading_tracking` and :attr:`head_pose_tracking` are the two re-enabled
    capabilities, and both start at weight zero and are ramped in by curricula once the stroke is
    solid.

    The eleven surviving stride terms keep their weights, with two edits: :attr:`pose` is narrowed to
    the legs and :attr:`wheel_speed` is made bidirectional.
    """

    ##
    # The stride terms this gait is the negation of.
    ##

    single_support = None
    glide = None
    skating_air_time = None
    gait_symmetry = None
    hip_roll_neutral = None
    braking = None

    ##
    # The two head-to-stand-pose pullers: one removed, one narrowed.
    ##

    neck_joint_pos_l2 = None

    # Narrowed from all eighteen joints to the ten leg joints, which changes the value of the reward and not only
    # its scope: the kernel is a *mean* over the selection, so the same joint errors now divide by ten instead
    # of by eighteen. The wheels' 999.0 tolerance entries go with the wheels.
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
            },
            "std_walking": {
                ".*hip_yaw.*": 0.3,
                ".*hip_roll.*": 0.6,
                ".*hip_pitch.*": 0.4,
                ".*knee.*": 0.4,
                ".*ankle.*": 0.25,
            },
            "std_running": {
                ".*hip_yaw.*": 0.5,
                ".*hip_roll.*": 0.8,
                ".*hip_pitch.*": 0.8,
                ".*knee.*": 0.8,
                ".*ankle.*": 0.5,
            },
            "walking_threshold": 0.01,
            "running_threshold": 0.5,
            "asset_cfg": _LEG_JOINT_CFG,
        },
    )

    ##
    # The gait.
    ##

    leg_symmetry = RewTerm(
        func=mdp.leg_symmetry_reward,
        weight=2.0,
        params={"left_joint_cfg": _LEFT_LEG_JOINT_CFG, "right_joint_cfg": _RIGHT_LEG_JOINT_CFG},
    )
    grounded = RewTerm(
        func=mdp.grounded_reward,
        weight=1.0,
        params={
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "command_name": "base_velocity",
            "bodies_per_foot": MICRODUCK_TIRES_PER_FOOT,
        },
    )

    ##
    # The two capabilities, both introduced by curriculum.
    ##

    # Weight 0.0, ramped to 3.0 by iteration 2500 as :attr:`heading_hold` is ramped out.
    heading_tracking = RewTerm(
        func=mdp.heading_tracking_reward,
        weight=0.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    # Weight 0.0, ramped to 4.0 by iteration 3000, alongside the command range that feeds it.
    head_pose_tracking = RewTerm(
        func=mdp.head_pose_tracking_raw,
        weight=0.0,
        params={"command_name": "head_pose", "std": 0.5, "asset_cfg": _HEAD_JOINT_CFG},
    )

    def __post_init__(self):
        # A negative throttle means *go backwards*, not *brake*, so the wheel reward pays for spin
        # aligned with the commanded sign rather than for forward spin alone. Everything else about
        # the term -- including its deliberately stale 0.0175 m radius -- is the stride task's.
        self.wheel_speed.params["bidirectional"] = True


@configclass
class CurriculumCfg(RollersCurriculumCfg):
    """Curriculum terms for the MDP.

    The stride task's four schedules plus four more, which is where this task's two re-enabled
    capabilities actually arrive. They are deliberately late: both the heading hand-over and the head
    command are held out until the swizzle itself is solid, because a policy that has not yet found
    the stroke will take a cheap yaw or head reward over a hard locomotion one.
    """

    ##
    # The heading hand-over: hold the spawn heading, then track a commanded one.
    ##

    heading_hold_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "heading_hold",
            "weight_stages": [
                {"step": _iterations(0), "weight": 1.0},
                # straight-line skating until the swizzle solidifies
                {"step": _iterations(1000), "weight": 1.0},
                {"step": _iterations(1750), "weight": 0.5},
                {"step": _iterations(2500), "weight": 0.0},
            ],
        },
    )

    heading_tracking_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "heading_tracking",
            "weight_stages": [
                {"step": _iterations(0), "weight": 0.0},
                {"step": _iterations(1000), "weight": 0.0},
                {"step": _iterations(1750), "weight": 1.5},
                {"step": _iterations(2500), "weight": 3.0},
            ],
        },
    )

    ##
    # The head command: the weight and the range open together.
    ##

    head_pose_tracking_weight = CurrTerm(
        func=mdp.reward_weight_stages,
        params={
            "reward_name": "head_pose_tracking",
            "weight_stages": [
                {"step": _iterations(0), "weight": 0.0},
                {"step": _iterations(1500), "weight": 0.0},
                {"step": _iterations(2250), "weight": 2.0},
                {"step": _iterations(3000), "weight": 4.0},
            ],
        },
    )

    head_pose_range = CurrTerm(
        func=mdp.command_range_stages,
        params={
            "command_name": "head_pose",
            "range_stages": [
                {"step": _iterations(0), "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": _iterations(1500), "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": _iterations(2250), "ranges": ((-0.55, 0.55), (-0.55, 0.55), (-0.70, 0.70), (-0.15, 0.15))},
                {"step": _iterations(3000), "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))},
            ],
        },
    )


##
# Environment configuration
##


@configclass
class MicroDuckVelocitySwizzleEnvCfg(MicroDuckVelocityRollersFlatEnvCfg):
    """MicroDuck swizzle-skating environment on flat ground."""

    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
