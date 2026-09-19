# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Internal clone payload definitions shared by the OvPhysX cloner and manager."""

import re
from collections.abc import Sequence
from fnmatch import fnmatchcase

CloneTransform = tuple[float, float, float, float, float, float, float]
CloneRecipe = tuple[str, list[str], list[CloneTransform], list[int] | None]


def ordered_clone_paths(paths: list[str], patterns: list[str]) -> list[str]:
    """Order binding rows numerically within each requested body pattern."""

    def order_key(path: str) -> tuple:
        group = next((i for i, pattern in enumerate(patterns) if fnmatchcase(path, pattern)), len(patterns))
        return group, tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path))

    return sorted(paths, key=order_key)


def clone_transforms_from_positions(positions: Sequence[Sequence[float]]) -> list[CloneTransform]:
    """Return xyzw clone transforms with identity rotations for world positions."""
    return [(float(x), float(y), float(z), 0.0, 0.0, 0.0, 1.0) for x, y, z in positions]
