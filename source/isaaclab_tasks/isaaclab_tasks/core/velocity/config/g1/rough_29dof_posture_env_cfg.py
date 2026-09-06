# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Why the shipped G1 splays its legs once its feet are fixed, and what stops it.

Replacing the shipped asset's four 5 mm foot spheres with ``g1_minimal.usd``'s sole plate takes
``Isaac-Velocity-Rough-G1-29Dof`` from ``success_rate`` 0.000 to 0.87-0.94, but every such policy
walks with its legs splayed almost flat: ``Episode_Reward/joint_deviation_hip`` sits at -0.18 to
-0.21 where the old asset's is -0.017, and the recorded clips agree with that number rather than
with the success rate.

An asset ladder ruled the asset out. Matching the old robot's leg joint limits, link masses and
joint origins changed nothing, and pinning shut the six degrees of freedom the old robot does not
have made it slightly worse. What was never held fixed is the *task*: the old asset's reference arm
runs the stock ``Isaac-Velocity-Rough-G1``, while every shipped-asset arm runs the 29-DoF task,
which adds a terrain-relative ``base_height`` termination. These configs separate the two.

* :class:`G129DofRoughNoHeightEnvCfg` drops that termination. If the splay goes, the termination
  induced it -- plausibly because splaying is the cheapest way to lower the centre of mass without
  tripping a floor that only checks pelvis height.
* :class:`G1RoughOldAssetHeightEnvCfg` is the reverse: the old asset, which does not splay, given
  the termination. If it starts splaying, that settles it from the other side.
* :class:`G129DofRoughHipL2EnvCfg` is the candidate fix rather than a diagnosis: hip deviation
  penalised the way WBC-AGILE penalises it, L2 at -1.0 against the stock L1 at -0.1.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .rough_29dof_env_cfg import G129DofRoughEnvCfg
from .rough_29dof_wbc_env_cfg import joint_deviation_l2
from .rough_env_cfg import G1RoughEnvCfg

_HIP_JOINTS = [".*_hip_roll_joint", ".*_hip_yaw_joint"]

_HIP_L2_WEIGHT = -1.0
"""WBC-AGILE's ``hip_pos_pen`` weight, against the stock task's -0.1 on an L1 term."""


@configclass
class G129DofRoughNoHeightEnvCfg(G129DofRoughEnvCfg):
    """The 29-DoF task without its base-height termination.

    Removing it re-opens the crouch the termination exists to close, so a low ``success_rate`` here
    is not a surprise and not the measurement -- ``joint_deviation_hip`` is.
    """

    def __post_init__(self):
        super().__post_init__()
        self.terminations.base_height = None


@configclass
class G1RoughOldAssetHeightEnvCfg(G1RoughEnvCfg):
    """The 37-joint asset on its own stock task, plus the 29-DoF task's base-height termination.

    The joint names the termination reads -- the root body and the height scanner -- exist on both
    robots, so nothing else has to be retargeted.
    """

    def __post_init__(self):
        super().__post_init__()

        from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: PLC0415

        from .rough_29dof_env_cfg import MINIMUM_PELVIS_HEIGHT, pelvis_below_terrain_clearance  # noqa: PLC0415

        self.terminations.base_height = DoneTerm(
            func=pelvis_below_terrain_clearance,
            params={
                "minimum_height": MINIMUM_PELVIS_HEIGHT,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )


@configclass
class G129DofRoughHipL2EnvCfg(G129DofRoughEnvCfg):
    """The 29-DoF task with hip deviation priced the way WBC-AGILE prices it."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_deviation_hip.func = joint_deviation_l2
        self.rewards.joint_deviation_hip.weight = _HIP_L2_WEIGHT
        self.rewards.joint_deviation_hip.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=_HIP_JOINTS)
        }


_AIR_TIME_SWEEP = {"c1": 1.0, "c2": 2.0, "c3": 4.0}
"""Multiples 4x, 8x and 16x of the stock ``feet_air_time`` weight of 0.25.

:class:`G129DofRoughHipL2EnvCfg` fixes the splay but still shuffles: its ``feet_air_time`` sits at
0.0086 against the 0.4 s the term can pay for. On the shipped asset's spheres, 1.0 was where
``success_rate`` peaked and 2.0 bought longer steps without buying tracking; a sole plate has far
more contact area to push against, so the useful range may run higher, and 4.0 brackets that from
above rather than assuming it.
"""


@configclass
class G129DofRoughHipL2AirTime4EnvCfg(G129DofRoughHipL2EnvCfg):
    """The splay fix plus four times the stock air-time weight."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time.weight = _AIR_TIME_SWEEP["c1"]


@configclass
class G129DofRoughHipL2AirTime8EnvCfg(G129DofRoughHipL2EnvCfg):
    """The splay fix plus eight times the stock air-time weight."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time.weight = _AIR_TIME_SWEEP["c2"]


@configclass
class G129DofRoughHipL2AirTime16EnvCfg(G129DofRoughHipL2EnvCfg):
    """The splay fix plus sixteen times the stock air-time weight."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time.weight = _AIR_TIME_SWEEP["c3"]
