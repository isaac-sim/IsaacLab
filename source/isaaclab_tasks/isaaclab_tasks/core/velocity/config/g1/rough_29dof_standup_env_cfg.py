# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A gradient out of the crouch, instead of only a wall at the bottom of it.

The warm-up arm gets eight of the nine cross-backend runs to a gait at 6000 iterations, and the one
that fails, fails the same way every time. Read off ``ww_hf_newton`` at 5400 iterations:

===============================  ===========  ===========
term                              s42 (works)  s43 (fails)
===============================  ===========  ===========
``Episode_Termination/base_height``   0.019        0.177
``Curriculum/terrain_levels``         5.75         2.21
``Train/mean_episode_length``       996.2        915.0
``Metrics/success_rate``              1.000        0.000
===============================  ===========  ===========

The failing run is not diverging and not dying -- it survives 915 of 1000 steps. It is walking
crouched, losing a sixth of its episodes to the height floor, and because the terrain curriculum
promotes on distance walked it never gets past level 2. Both seeds sit at a 100% termination rate
the moment the warm-up ends at iteration 500; the difference is only whether they climb out.

Nothing in the task pushes the pelvis up. The floor at :data:`MINIMUM_PELVIS_HEIGHT` is a
termination, so it prices being 1 cm too low exactly like being 20 cm too low, and it gives no
direction at all while the robot is above it -- a policy at 0.45 m has no reason to prefer 0.60. The
ramped-floor variant, which lowers that wall early and raises it over 1500 iterations, does worse
(1/4 on heightfield against 2/3): a softer wall is still a wall, and it buys the policy time to
entrench the crouch rather than time to leave it.

So add the missing gradient: a one-sided quadratic on the pelvis-height *deficit*, measured against
the same terrain-relative reference the termination uses. Above :data:`_STAND_HEIGHT` it is exactly
zero, so a healthy gait pays nothing and w100's posture is untouched; below it the penalty grows
smoothly toward the floor the episode would otherwise end on.
"""

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.configclass import configclass

from .rough_29dof_waistonly_env_cfg import G129DofRoughAirTime100WaistWarmupEnvCfg

_STAND_HEIGHT = 0.65
"""Pelvis clearance above which the penalty is zero [m].

The robot spawns at 0.793 m and the episode ends below 0.4 m, so this sits in the upper half of the
live range: high enough that the crouch the failing seeds settle into is penalized, low enough that
the flexion of a normal stride is not. One-sided, so it can only push up.
"""

_STAND_WEIGHTS = {"s1": -10.0, "s3": -30.0}
"""Weights to try on the deficit.

At the 0.13 m deficit the stuck seed walks with, -10 costs 0.17 per step against the 1.0 that
perfect velocity tracking pays, and -30 costs 0.51. The first is a nudge, the second is comparable
to the whole tracking term; which of those is needed to leave a crouch that is otherwise stable is
the question the sweep answers.
"""


def pelvis_height_deficit_l2(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Squared shortfall of the root below ``target_height`` above the ground beneath it.

    The ground reference is the median ray hit, matching
    :func:`~.rough_29dof_env_cfg.pelvis_below_terrain_clearance` -- world z would price standing in a
    dip as a crouch, and a mean is dragged by the few rays that land on a ledge.

    Args:
        env: The environment.
        target_height: Clearance at and above which this term is zero [m].
        asset_cfg: Articulation whose root height is measured.
        sensor_cfg: Ray caster defining the ground beneath the robot.

    Returns:
        Per-environment penalty magnitude, shape ``(num_envs,)``, zero where the robot stands tall.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    hits = sensor.data.ray_hits_w.torch[..., 2]
    ground = torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0).median(dim=1).values
    clearance = asset.data.root_pos_w.torch[:, 2] - ground
    return torch.clamp(target_height - clearance, min=0.0).square()


def _add_stand_height(cfg, weight: float) -> None:
    """Attach the one-sided pelvis-height penalty to ``cfg`` at ``weight``."""
    cfg.rewards.pelvis_height = RewTerm(
        func=pelvis_height_deficit_l2,
        weight=weight,
        params={
            "target_height": _STAND_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )


@configclass
class G129DofRoughStandUpEnvCfg(G129DofRoughAirTime100WaistWarmupEnvCfg):
    """The warm-up arm plus the pelvis-height deficit penalty at -10."""

    def __post_init__(self):
        super().__post_init__()
        _add_stand_height(self, _STAND_WEIGHTS["s1"])


@configclass
class G129DofRoughStandUpStrongEnvCfg(G129DofRoughAirTime100WaistWarmupEnvCfg):
    """The warm-up arm plus the pelvis-height deficit penalty at -30."""

    def __post_init__(self):
        super().__post_init__()
        _add_stand_height(self, _STAND_WEIGHTS["s3"])
