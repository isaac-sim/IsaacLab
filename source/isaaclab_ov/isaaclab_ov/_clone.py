# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal clone payload definitions shared by the OvPhysX cloner and manager."""

from collections.abc import Sequence
from dataclasses import dataclass

CloneTransform = tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class CloneRecipe:
    """One source-to-target runtime clone operation."""

    source: str
    targets: tuple[str, ...]
    transforms: tuple[CloneTransform, ...]
    env_ids: tuple[int, ...] | None = None


def clone_transforms_from_positions(positions: Sequence[Sequence[float]]) -> list[CloneTransform]:
    """Return xyzw clone transforms with identity rotations for world positions."""
    return [(float(x), float(y), float(z), 0.0, 0.0, 0.0, 1.0) for x, y, z in positions]
