# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pricing left/right gait asymmetry on top of the w100 gait.

w100 walks well but not evenly: it lifts the right foot more than the left, which reads as a limp.
Nothing in the reward set asks the two feet to behave alike. ``feet_air_time_positive_biped``
rewards the duration of the current single-stance phase, takes the minimum over the two feet and
clamps at its threshold, so it cannot distinguish a brisk symmetric gait from standing on one leg
with the other held up -- the latter sits at the clamp while the former's timers reset at every
touchdown.

Measured on w100 under evaluation conditions, the two feet differ by 0.091 s in swing duration and
0.181 s in stance duration. Those are the numbers the weights below are set against.

An earlier attempt at this failed, and the way it failed shaped these arms. Weights of -0.5, -2.0
and -5.0 on the swing-difference term all produced ``success_rate`` 0.000 with the single-stance
share collapsing -- but they were stacked on the air-time-2.0 arm, which was already the unstable
end of that sweep, and its imbalance was roughly four times w100's, so the same weight bought four
times the penalty. Rebuilt on w100 the same numbers mean something different, and the third arm
changes the shape of the term rather than only its weight.
"""

import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .rough_29dof_posture_env_cfg import G129DofRoughHipL2AirTime100EnvCfg

_FEET = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")


def _moving(env, command_name: str) -> torch.Tensor:
    """Environments whose command asks them to move. Standing still is not a gait fault."""
    return torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1


def _pair(buffer, body_ids) -> torch.Tensor:
    """The two feet's values from a contact-sensor buffer, as a ``(num_envs, 2)`` tensor."""
    data = buffer.torch if hasattr(buffer, "torch") else buffer
    pair = data[:, body_ids]
    if pair.shape[1] != 2:
        raise ValueError(f"expected two feet, sensor_cfg selected {pair.shape[1]}")
    return pair


def feet_swing_imbalance(env, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity") -> torch.Tensor:
    """Penalise a gait whose two feet swing for different durations [s].

    ``last_air_time`` holds the duration of the swing that ended at the most recent touchdown, so it
    is constant between touchdowns and only moves when a foot lands. That makes it far less noisy
    than comparing the two feet's running timers, which at any instant have one foot down and one
    foot up and whose raw difference says nothing about the gait.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    swing = _pair(sensor.data.last_air_time, sensor_cfg.body_ids)
    return torch.abs(swing[:, 0] - swing[:, 1]) * _moving(env, command_name)


def feet_stride_imbalance(env, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity") -> torch.Tensor:
    """Penalise asymmetry in both halves of the stride [s].

    The swing-only form leaves half the gait unpriced: two feet can swing for the same time and
    still stand for very different times, which is what a limp looks like when the robot hurries off
    one leg. Measured on w100 the stance difference is 0.181 s against the swing's 0.091 s, so it is
    the larger of the two and the one a swing-only term ignores.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    swing = _pair(sensor.data.last_air_time, sensor_cfg.body_ids)
    stance = _pair(sensor.data.last_contact_time, sensor_cfg.body_ids)
    imbalance = torch.abs(swing[:, 0] - swing[:, 1]) + torch.abs(stance[:, 0] - stance[:, 1])
    return imbalance * _moving(env, command_name)


_SWING_WEIGHTS = {"y1": -1.0, "y2": -3.0}
"""Weights for :func:`feet_swing_imbalance`, against w100's measured 0.091 s.

They cost 0.09 and 0.27 per step, which is between the task's regularizers -- ``dof_torques_l2`` is
worth about 0.002 -- and its tracking reward. Above that the term stops arguing with the gait and
starts replacing it, which is how the previous attempt reached ``success_rate`` 0.000.
"""

_STRIDE_WEIGHT = -1.0
"""Weight for :func:`feet_stride_imbalance`, against its measured 0.272 s: 0.27 per step.

Deliberately the same cost as ``y2`` so the two shapes are compared at equal pressure rather than at
equal weight.
"""


@configclass
class G129DofRoughAirTime100Sym1EnvCfg(G129DofRoughHipL2AirTime100EnvCfg):
    """w100 plus a swing-duration symmetry penalty at -1.0."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_swing_imbalance = RewTerm(
            func=feet_swing_imbalance,
            weight=_SWING_WEIGHTS["y1"],
            params={"sensor_cfg": _FEET},
        )


@configclass
class G129DofRoughAirTime100Sym2EnvCfg(G129DofRoughHipL2AirTime100EnvCfg):
    """w100 plus a swing-duration symmetry penalty at -3.0."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_swing_imbalance = RewTerm(
            func=feet_swing_imbalance,
            weight=_SWING_WEIGHTS["y2"],
            params={"sensor_cfg": _FEET},
        )


@configclass
class G129DofRoughAirTime100Sym3EnvCfg(G129DofRoughHipL2AirTime100EnvCfg):
    """w100 plus a symmetry penalty covering both swing and stance, at -1.0."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_stride_imbalance = RewTerm(
            func=feet_stride_imbalance,
            weight=_STRIDE_WEIGHT,
            params={"sensor_cfg": _FEET},
        )
