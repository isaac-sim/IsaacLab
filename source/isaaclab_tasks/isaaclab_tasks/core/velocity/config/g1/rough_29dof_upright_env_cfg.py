# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Two ways to stop the pelvis leaning, on top of the per-joint action scale.

``ms`` -- ``su`` with mjlab's per-joint action scale -- is the best base this line has: 0.996 mean
success at sd 0.001, and it removed the gross leg asymmetry (knee 33-38 degrees apart on ``w100``,
under 3 degrees here). What it leaves is a pelvis rolled 2.8 degrees on two of its three seeds,
measured on flat ground under a pinned straight command.

Two things in this reward set explain that, and each gets an arm:

**The tilt is barely priced.** ``flat_orientation_l2`` at -1.0 charges ``sin^2(tilt)``, which is
0.0024 per step at 2.8 degrees against a tracking reward near 1.0. mjlab prices the same quantity as
``exp(-|g_xy|^2 / 0.2)`` at +1.0, whose slope at zero tilt is 5x steeper -- 0.0118 at 2.8 degrees,
and 1% of a unit reward is reached at 2.57 degrees rather than 5.74. It is also a bounded *reward*
rather than an unbounded penalty, so being upright is worth up to +1.0 per step, on the same scale
as velocity tracking, instead of being worth nothing.

**The tilt has an unpriced way out.** ``joint_deviation_hip`` covers hip roll and yaw,
``joint_deviation_torso`` covers the waist, and nothing at all covers the ankles. Measured on
``ms``/s42: ankle roll is left -3.43 and right +9.00 degrees, a common mode of +2.79, against a
pelvis roll of -2.79 -- the same number with the opposite sign, because rolling both ankles the same
way *is* tilting the body. So the second arm prices the common mode ``theta_L + theta_R`` and leaves
the difference alone: the difference is what lets each foot meet a side slope, and penalizing it
would fight rough terrain for no reason.

The third arm runs both, because they close different halves of the same leak and the interesting
failure is that one makes the other unnecessary.
"""

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .rough_29dof_mjlab_env_cfg import G129DofRoughMjlabScaleEnvCfg

_UPRIGHT_STD = 0.4472135954999579
"""``sqrt(0.2)``, mjlab's value. The reward is ``exp(-|g_xy|^2 / std^2)``, so ``std^2 = 0.2``."""

_UPRIGHT_WEIGHT = 1.0
_ANKLE_COMMON_WEIGHT = -5.0
"""Weight on the squared ankle-roll common mode [1/rad^2].

At the 2.79 degrees of common mode measured on ``ms``, ``(theta_L + theta_R)^2`` is 0.0095, so this
costs 0.048 per step -- about a twentieth of a perfect tracking reward. Enough to be seen against
the 0.0024 the tilt currently costs, and far from the scale that made the hip-pitch term at -0.5
unlearnable.
"""


def _as_tensor(value):
    """Return ``value`` as a plain tensor, unwrapping this repo's ``ProxyArray`` where present."""
    return value.torch if hasattr(value, "torch") else value


def upright_exp(env, std: float = _UPRIGHT_STD, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Bounded reward for holding the base vertical: ``exp(-|g_xy|^2 / std^2)``.

    mjlab's ``upright`` in the same form, on the root rather than on ``torso_link``. The root is
    what was measured leaning here, and the waist is already held by ``joint_deviation_torso`` at
    L2 -1.0, which is the term that stops the torso from paying for a rolled pelvis.

    Args:
        env: The environment.
        std: Tolerance on the projected-gravity magnitude; ``std**2`` divides the squared tilt.
        asset_cfg: The articulation whose base orientation is rewarded.

    Returns:
        Per-environment reward in ``(0, 1]``, one when perfectly upright.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    gravity = _as_tensor(asset.data.projected_gravity_b)
    return torch.exp(-torch.sum(torch.square(gravity[:, :2]), dim=1) / std**2)


def ankle_roll_common_mode_l2(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"])
) -> torch.Tensor:
    """Squared common mode of the two ankle-roll joints, ``(theta_L + theta_R)^2``.

    Under a left-right mirror an ankle-roll angle changes sign, so a symmetric stance has
    ``theta_L = -theta_R`` and the sum is zero; rolling both ankles the same way in body coordinates
    makes the sum twice the tilt. Summing is order-independent, so which ankle the config resolves
    first does not matter.

    Args:
        env: The environment.
        asset_cfg: The articulation and its two ankle-roll joints.

    Returns:
        Per-environment penalty magnitude, shape ``(num_envs,)``.

    Raises:
        ValueError: If the config does not resolve to exactly two joints.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    angles = _as_tensor(asset.data.joint_pos)[:, asset_cfg.joint_ids]
    if angles.shape[1] != 2:
        raise ValueError(f"expected two ankle-roll joints, resolved {angles.shape[1]}")
    return torch.square(angles.sum(dim=1))


def _add_upright(cfg) -> None:
    """Reprice the base tilt in mjlab's exponential form, in place."""
    cfg.rewards.flat_orientation_l2.weight = 0.0
    cfg.rewards.upright = RewTerm(
        func=upright_exp,
        weight=_UPRIGHT_WEIGHT,
        params={"std": _UPRIGHT_STD, "asset_cfg": SceneEntityCfg("robot")},
    )


def _add_ankle_common(cfg) -> None:
    """Price the ankle-roll common mode, in place."""
    cfg.rewards.ankle_roll_common = RewTerm(
        func=ankle_roll_common_mode_l2,
        weight=_ANKLE_COMMON_WEIGHT,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"])},
    )


@configclass
class G129DofRoughUprightEnvCfg(G129DofRoughMjlabScaleEnvCfg):
    """``ms`` with the tilt repriced as mjlab's bounded upright reward."""

    def __post_init__(self):
        super().__post_init__()
        _add_upright(self)


@configclass
class G129DofRoughAnkleRollEnvCfg(G129DofRoughMjlabScaleEnvCfg):
    """``ms`` with the ankle-roll common mode priced."""

    def __post_init__(self):
        super().__post_init__()
        _add_ankle_common(self)


@configclass
class G129DofRoughUprightAnkleEnvCfg(G129DofRoughMjlabScaleEnvCfg):
    """``ms`` with both."""

    def __post_init__(self):
        super().__post_init__()
        _add_upright(self)
        _add_ankle_common(self)
