# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pricing mechanical power on top of the w100 gait.

w100 -- plate feet, hip deviation L2 at -1.0, air-time weight 1.0 -- walks with short, quick steps,
and short quick steps are expensive: every one is another acceleration and deceleration of a leg
that a longer stride would have carried through. The task already penalises torque
(``dof_torques_l2`` at -1.5e-7) and joint acceleration, but neither is energy. A motor holding a
static load draws torque at no mechanical power, and a joint spinning freely draws none either;
what costs is the product, and only the product distinguishes a shuffle from a stride.

Measured on w100 under evaluation conditions, the leg joints draw a mean of 234 W and peak at
4.5 kW. That is the scale the weights below are chosen against, not a guess.
"""

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .rough_29dof_posture_env_cfg import G129DofRoughHipL2AirTime100EnvCfg

_POWER_JOINTS = [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
"""The joints that do the walking, matching the scope of the task's existing torque penalty.

Scoped to the legs and not to ``.*`` because the arms and hands are held still by deviation
penalties rather than driven, and the Dex3 finger joints -- 1.5e-06 kg*m^2 of inertia against a
position loop stiff enough for the legs -- chatter at a power the same expression reports as 10 kW
across all 43 joints against 234 W across the legs. Summing those in would make the term mostly a
measurement of finger noise.
"""


def joint_mechanical_power(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Total mechanical power drawn by the selected joints [W].

    The absolute value is deliberate. Where torque and velocity oppose each other the joint is
    braking, and a real motor dissipates that energy in its windings rather than recovering it, so
    braking is a cost and not a credit.

    Args:
        env: Environment the reward is computed for.
        asset_cfg: Articulation and joints to sum over.

    Returns:
        Per-environment power [W], shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torque * velocity), dim=1)


_POWER_WEIGHTS = {"p1": -1.0e-4, "p4": -2.0e-4, "p5": -3.0e-4, "p2": -5.0e-4, "p3": -2.0e-3}
"""Weights bracketing the useful range rather than guessing at it.

At w100's measured 234 W these cost 0.023, 0.047, 0.070, 0.12 and 0.47 per step, against a tracking
reward near 1.0 and an ``action_rate_l2`` near 0.6. Measured at 6000 iterations, -1e-4 and -5e-4
both reach ``success_rate`` 1.000 while cutting leg power 33% and 45%, and -2e-3 reaches 0.194 at
20 W -- the term stopped being a regularizer and became the objective, which is the failure the
bracket was built to find.

p4 and p5 fill the gap between the two that worked. -5e-4 buys the larger saving but flexes the
hips to -22 degrees against -12 at -1e-4, so the question they answer is where the saving stops
being free.
"""


def _add_power_penalty(cfg, weight: float) -> None:
    """Add the power penalty to ``cfg`` at ``weight``, in place."""
    cfg.rewards.joint_power = RewTerm(
        func=joint_mechanical_power,
        weight=weight,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=_POWER_JOINTS)},
    )


@configclass
class G129DofRoughAirTime100Power1EnvCfg(G129DofRoughHipL2AirTime100EnvCfg):
    """w100 plus a leg power penalty at -1e-4."""

    def __post_init__(self):
        super().__post_init__()
        _add_power_penalty(self, _POWER_WEIGHTS["p1"])


@configclass
class G129DofRoughAirTime100Power4EnvCfg(G129DofRoughHipL2AirTime100EnvCfg):
    """w100 plus a leg power penalty at -2e-4."""

    def __post_init__(self):
        super().__post_init__()
        _add_power_penalty(self, _POWER_WEIGHTS["p4"])


@configclass
class G129DofRoughAirTime100Power5EnvCfg(G129DofRoughHipL2AirTime100EnvCfg):
    """w100 plus a leg power penalty at -3e-4."""

    def __post_init__(self):
        super().__post_init__()
        _add_power_penalty(self, _POWER_WEIGHTS["p5"])


@configclass
class G129DofRoughAirTime100Power2EnvCfg(G129DofRoughHipL2AirTime100EnvCfg):
    """w100 plus a leg power penalty at -5e-4."""

    def __post_init__(self):
        super().__post_init__()
        _add_power_penalty(self, _POWER_WEIGHTS["p2"])


@configclass
class G129DofRoughAirTime100Power3EnvCfg(G129DofRoughHipL2AirTime100EnvCfg):
    """w100 plus a leg power penalty at -2e-3."""

    def __post_init__(self):
        super().__post_init__()
        _add_power_penalty(self, _POWER_WEIGHTS["p3"])
