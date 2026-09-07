# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The w100 gait arm under domain randomization.

:class:`~rough_29dof_posture_env_cfg.G129DofRoughHipL2AirTime100EnvCfg` is the best-looking policy
this line has produced -- plate feet, hip deviation penalised L2 at -1.0, air-time weight 1.0 -- but
it is trained on one nominal robot on one nominal floor, so nothing about it is known to survive
contact with hardware. These two arms add what the sim-to-real line already uses, on top of that
gait rather than on top of the DR29 teacher.

* :class:`G129DofRoughAirTime100DREnvCfg` adds the randomization: the four startup draws
  (ground friction, joint armature, joint friction, actuator stiffness) plus a push twice as strong
  and twice as frequent as the stock one.
* :class:`G129DofRoughAirTime100DRHistoryEnvCfg` adds a five-frame observation history on top. A
  policy that cannot see the past cannot identify which robot it is driving, which is the mechanism
  randomization is supposed to teach; the DR29 line has always carried both together, so this arm
  separates the two rather than assuming the pairing.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.velocity.mdp as mdp

from .rough_29dof_posture_env_cfg import G129DofRoughHipL2AirTime100EnvCfg

_TORSO_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")

_ARM_JOINTS = (
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
)

_BODY_JOINTS = (
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_hip_yaw_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
    *_TORSO_JOINTS,
    *_ARM_JOINTS,
)
"""The 29 motors the robot commands -- everything except the Dex3 fingers.

Scoped this way rather than to ``.*`` for the reason the DR29 task gives: the fingers are a separate
device this policy never drives, and a finger link's inertia is small enough that scaling its
armature is numerically hostile for no transfer benefit.
"""

_HISTORY_LENGTH = 5
"""Frames of proprioception the history arm stacks, matching the DR29 tasks."""

_HISTORY_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
)
"""Policy observation terms the history arm stacks -- proprioception, not the height scan."""

_HARD_PUSH = {
    "interval_range_s": (5.0, 10.0),
    "velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6), "yaw": (-0.5, 0.5)},
}
"""The push the 37-joint DR task uses, against the stock 10-15 s at +-0.5 m/s in x and y alone.

Twice as often, a fifth stronger, and with a yaw kick. Measured on the DR29 teacher this was the
term that carried the gain -- ``success_rate`` 0.962 to 0.994 and ``fall_rate`` 0.130 to 0.062 --
while self-collision on its own bought nothing and cost 17 cm of pelvis height.
"""


def _add_randomization(cfg) -> None:
    """Add the DR29 startup draws and the stronger push to ``cfg``, in place.

    Every added term is ``startup`` mode: friction is a property of the floor, armature is reflected
    rotor inertia, joint friction is a property of the harmonic drive, and a position-loop gain is a
    property of the unit. None of them changes between two runs of the same robot, so one draw per
    environment is the honest model and 4096 environments are 4096 robots.

    Args:
        cfg: Environment configuration to randomize, modified in place.
    """
    cfg.events.physics_material.params.update(
        {
            # The stock values are 0.8 static / 0.6 dynamic, so these are those spans scaled 0.5-2x.
            "static_friction_range": (0.4, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "num_buckets": 64,
            "make_consistent": True,
        }
    )

    cfg.events.joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(_BODY_JOINTS)),
            "armature_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    cfg.events.joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            # The USD assumes frictionless joints; the official MuJoCo G1 carries 0.2 N*m of
            # frictionloss on every joint, so the span brackets that rather than stopping short.
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(_BODY_JOINTS)),
            "friction_distribution_params": (0.0, 0.3),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    cfg.events.actuator_stiffness = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            # Stiffness only -- damping stays nominal so the effective damping ratio moves with the
            # draw, which is what a real position loop does when its gain is off.
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(_BODY_JOINTS)),
            "stiffness_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    cfg.events.push_robot.interval_range_s = _HARD_PUSH["interval_range_s"]
    cfg.events.push_robot.params["velocity_range"] = _HARD_PUSH["velocity_range"]


@configclass
class G129DofRoughAirTime100DREnvCfg(G129DofRoughHipL2AirTime100EnvCfg):
    """The w100 gait with the DR29 randomization set and the stronger push."""

    def __post_init__(self):
        super().__post_init__()
        _add_randomization(self)


@configclass
class G129DofRoughAirTime100DRHistoryEnvCfg(G129DofRoughAirTime100DREnvCfg):
    """The randomized arm, plus five frames of proprioception in the policy observation."""

    def __post_init__(self):
        super().__post_init__()

        # Per term rather than group-level: a group-level history would stack the height scan too,
        # which is 187 numbers per frame against 76 for all of proprioception put together, and a
        # terrain map five steps old is not what the past is being added for. History is buffered
        # per term and the terms concatenated afterwards, so the layout is term by term --
        # [ang_vel(t-4..t), gravity(t-4..t), ...] -- not frame by frame. A deployment stack has to
        # reproduce that order, not just the frame count.
        for term in _HISTORY_TERMS:
            obs_term = getattr(self.observations.policy, term)
            obs_term.history_length = _HISTORY_LENGTH
            obs_term.flatten_history_dim = True


_WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

_WAIST_L2_WEIGHTS = {"t1": -1.0, "t2": -3.0}
"""Weights for the waist deviation penalty, replacing the stock L1 at -0.1.

Measured on d1, the randomized arm leans its upper body back 26 degrees at the waist while its
pelvis stays upright within a degree -- ``success_rate`` 0.995, ``pelvis_height`` 0.82 and root
pitch -0.8 degrees all miss it completely, because every one of them measures the pelvis. Leaning
back parks the torso's mass behind the hips, which is both steadier under a push and cheaper: the
same policy draws 180 W against w100's 237 W.

The stock penalty cannot argue with that. L1 at -0.1 prices 26 degrees at 0.046 per step. L2 at
-1.0 prices it at 0.21, and -3.0 at 0.63, which brackets the point where holding the torso upright
costs more than the lean saves. This is the same correction that fixed the splayed hips, where L1
at -0.1 lost to L2 at -1.0.
"""


def _add_waist_l2(cfg, weight: float) -> None:
    """Reprice waist deviation as L2 at ``weight``, in place."""
    from .rough_29dof_wbc_env_cfg import joint_deviation_l2  # noqa: PLC0415

    cfg.rewards.joint_deviation_torso.func = joint_deviation_l2
    cfg.rewards.joint_deviation_torso.weight = weight
    cfg.rewards.joint_deviation_torso.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=_WAIST_JOINTS)}


@configclass
class G129DofRoughAirTime100DRWaist1EnvCfg(G129DofRoughAirTime100DREnvCfg):
    """The randomized arm with waist deviation priced L2 at -1.0."""

    def __post_init__(self):
        super().__post_init__()
        _add_waist_l2(self, _WAIST_L2_WEIGHTS["t1"])


@configclass
class G129DofRoughAirTime100DRWaist3EnvCfg(G129DofRoughAirTime100DREnvCfg):
    """The randomized arm with waist deviation priced L2 at -3.0."""

    def __post_init__(self):
        super().__post_init__()
        _add_waist_l2(self, _WAIST_L2_WEIGHTS["t2"])


@configclass
class G129DofRoughAirTime100HistoryEnvCfg(G129DofRoughHipL2AirTime100EnvCfg):
    """The w100 gait with five frames of proprioception and nothing else changed.

    d2 added the history on top of the randomization and the stronger push, and came out worse than
    d1 on every number that moved -- ``success_rate`` 0.648 against 0.791, and a waist leaned back
    30 degrees against 26. That confounds two changes. This arm carries the history alone, on the
    stock randomization w100 already runs, so whether the extra frames help or hurt is answerable
    without the push in the way.
    """

    def __post_init__(self):
        super().__post_init__()

        # Proprioception only; the height scan stays single-frame. See
        # :class:`G129DofRoughAirTime100DRHistoryEnvCfg` for why.
        for term in _HISTORY_TERMS:
            obs_term = getattr(self.observations.policy, term)
            obs_term.history_length = _HISTORY_LENGTH
            obs_term.flatten_history_dim = True
