# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal clone payload definitions shared by the OvPhysX cloner and manager."""

from collections.abc import Sequence

CloneTransform = tuple[float, float, float, float, float, float, float]


def clone_transforms_from_positions(positions: Sequence[Sequence[float]]) -> list[CloneTransform]:
    """Return xyzw clone transforms with identity rotations for world positions."""
    return [(float(x), float(y), float(z), 0.0, 0.0, 0.0, 1.0) for x, y, z in positions]
