# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for forward-kinematics-based reachable pose commands."""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands import UniformPoseCommandCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .fk_pose_command import FkReachablePoseCommand

KinematicChainEntry = tuple[
    str,
    tuple[float, float, float],
    tuple[float, float, float],
]

_HALF_PI = math.pi / 2.0
A1_RIGHT_CHAIN: list[KinematicChainEntry] = [
    (".*joint1.*", (0.0, 0.0, 0.0315), (0.0, 0.0, 0.0)),
    (".*joint2.*", (0.0, 0.0, 0.12745), (_HALF_PI, _HALF_PI, 0.0)),
    (".*joint3.*", (0.0, -0.09325, 0.0), (_HALF_PI, -_HALF_PI, 0.0)),
    (".*joint4.*", (0.0, 0.001, 0.14025), (-_HALF_PI, 0.0, 0.0)),
    (".*joint5.*", (0.0, -0.088, -0.001), (_HALF_PI, 0.0, 0.0)),
    (".*joint6.*", (0.0, -0.0005, 0.133), (_HALF_PI, 0.0, 0.0)),
    (".*joint7.*", (0.0, 0.112, -0.0005), (_HALF_PI, 0.0, math.pi)),
]
"""A1 right-arm kinematic chain from ``base_link`` through ``Link7``."""


@configclass
class FkReachablePoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for pose targets sampled by A1 forward kinematics."""

    class_type: type[FkReachablePoseCommand] | str = "{DIR}.fk_pose_command:FkReachablePoseCommand"

    chain: list[KinematicChainEntry] = MISSING
    """Kinematic chain entries ordered from the robot base to the end effector."""

    joint_range_scale: float = 1.0
    """Centered fraction of each joint's full position range used for sampling."""
