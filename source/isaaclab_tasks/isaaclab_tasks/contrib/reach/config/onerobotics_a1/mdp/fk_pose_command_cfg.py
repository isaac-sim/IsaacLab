# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for forward-kinematics-based reachable pose commands."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands import UniformPoseCommandCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .fk_pose_command import FkReachablePoseCommand

Vector3 = tuple[float, float, float]
FixedTransform = tuple[Vector3, Vector3]
LegacyKinematicChainEntry = tuple[
    str,
    Vector3,
    Vector3,
]
KinematicChainEntry = LegacyKinematicChainEntry | tuple[str, Vector3, Vector3, Vector3]

A1_RIGHT_CHAIN: list[KinematicChainEntry] = [
    (".*joint1.*", (0.0, 0.0, 0.0125), (0.0, 0.0, 0.0)),
    (".*joint2.*", (0.0, 0.0, 0.12745), (1.5708, 1.5708, 0.0)),
    (".*joint3.*", (0.0, -0.09325, -0.0001), (1.5708, 1.5708, 0.0)),
    (
        ".*joint4.*",
        (0.0, -0.000999999996676215, 0.140249999995695),
        (1.57079632678358, 0.0, -6.73358220455809e-5),
    ),
    (".*joint5.*", (0.0, 0.08800000000195, -0.00100000000141642), (-1.57079632679444, 0.0, 0.0)),
    (".*joint6.*", (0.0, 0.0005, 0.133), (-1.5708, 0.0, 0.0)),
    (".*joint7.*", (0.0, -0.112, 0.00025), (1.5708, 0.0, 0.0)),
]
"""A1 2026 right-arm kinematic chain from ``base_link`` through ``Link7``."""

A1_BIMANUAL_RIGHT_FIXED_MOUNT: FixedTransform = (
    (-0.080655000000000004, -0.035999999999999997, 0.33600000000000002),
    (1.5707963267948966, -1.5707963267948966, 0.0),
)
"""Fixed transform from the bimanual ``base_link`` to ``Link_r0``."""

A1_BIMANUAL_RIGHT_CHAIN: list[KinematicChainEntry] = [
    (
        ".*joint_r1.*",
        (0.0, 0.0, 0.060249999999999998),
        (0.0, 0.0, 0.0),
        (-3.6732051033465739e-06, 0.0, 0.99999999999325373),
    ),
    (
        ".*joint_r2.*",
        (-3.625453437003069e-07, 0.0, 0.098699999999334154),
        (2.3561944901889715, 1.570791132098422, 0.78539816340082136),
        (0.0, 0.0, 1.0),
    ),
    (
        ".*joint_r3.*",
        (0.0, -0.1215, 0.0),
        (-1.5708, -1.5707926535897934, -3.1415926535897931),
        (-3.6732051033217936e-06, -3.6731916108860622e-06, 0.99999999998650746),
    ),
    (
        ".*joint_r4.*",
        (-4.1176794502467211e-07, -0.00010086139745906392, 0.11199999962951541),
        (-1.5707963267948966, 0.0, 0.0),
        (-1.3492410950664359e-11, -7.3463967141830754e-06, 0.99999999997301503),
    ),
    (
        ".*joint_r5.*",
        (-4.0772576782128141e-07, -0.11100000073420146, 9.9634549962015071e-05),
        (1.5707999999865074, 1.3492410951172447e-11, -3.6732051033052743e-06),
        (0.0, 0.0, 1.0),
    ),
    (".*joint_r6.*", (0.0, 0.0, 0.11), (1.5708, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (".*joint_r7.*", (0.0, 0.112, 0.0), (1.5708, 0.0, 3.1416), (0.0, 0.0, 1.0)),
]
"""Right-arm chain from ``Link_r0`` through ``Link_r7`` in the bimanual source model."""

A1_BIMANUAL_LEFT_FIXED_MOUNT: FixedTransform = (
    (-0.080655000000000004, 0.035999999999999997, 0.33600000000000002),
    (-1.5707963267948966, -1.5707963267948966, 0.0),
)
"""Fixed transform from the bimanual ``base_link`` to ``Link_l0``."""

A1_BIMANUAL_LEFT_CHAIN: list[KinematicChainEntry] = [
    (
        ".*joint_l1.*",
        (0.0, 0.0, 0.060249999999999998),
        (0.0, 0.0, 0.0),
        (-3.6732051033465739e-06, 0.0, 0.99999999999325373),
    ),
    (
        ".*joint_l2.*",
        (-3.625453437003069e-07, 0.0, 0.098699999999334154),
        (-1.5707963267948966, -1.5707963267948966, 0.0),
        (-3.6732185957575245e-06, -3.6731916108612818e-06, 0.99999999998650746),
    ),
    (
        ".*joint_l3.*",
        (-4.4592709819402209e-07, -0.12149999963104151, -0.00010044629441870738),
        (1.5707963267948966, -1.5707963267948966, 0.0),
        (-3.6735589314205621e-06, 3.6732050849805432e-06, 0.99999999998650602),
    ),
    (
        ".*joint_l4.*",
        (-4.2148360030222731e-07, 0.00010086139846659004, 0.11199999962947839),
        (1.5707963264005926, -3.6731646048592994e-06, 0.0001073464236997842),
        (0.0, 0.0, 1.0),
    ),
    (".*joint_l5.*", (0.0, 0.111, 0.00010045), (-1.5708, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (".*joint_l6.*", (0.0, 0.0, 0.11), (-1.5708, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (".*joint_l7.*", (0.0, -0.112, 0.0), (1.5708, 6.7338e-05, 0.0), (0.0, 0.0, 1.0)),
]
"""Left-arm chain from ``Link_l0`` through ``Link_l7`` in the bimanual source model."""


@configclass
class FkReachablePoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for pose targets sampled by A1 forward kinematics."""

    class_type: type[FkReachablePoseCommand] | str = "{DIR}.fk_pose_command:FkReachablePoseCommand"

    chain: list[KinematicChainEntry] = MISSING
    """Kinematic chain entries ordered from the robot base to the end effector."""

    fixed_transform: FixedTransform | None = None
    """Optional fixed transform applied before the first actuated joint.

    The legacy three-field chain with no fixed transform retains its original
    z-axis-only forward-kinematics path exactly.
    """

    joint_range_scale: float = 1.0
    """Centered fraction of each joint's full position range used for sampling."""
