# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence


def make_unique_names(names: Sequence[str]) -> list[str]:
    """Preserve input order while suffixing duplicate names without collisions."""
    authored_names = set(names)
    used_names: set[str] = set()
    unique_names: list[str] = []
    for name in names:
        if name not in used_names:
            unique_name = name
        else:
            suffix = 0
            while True:
                unique_name = f"{name}_{suffix}"
                suffix += 1
                if unique_name not in authored_names and unique_name not in used_names:
                    break
        unique_names.append(unique_name)
        used_names.add(unique_name)
    return unique_names


def get_unique_names_at_indices(names: Sequence[str], indices: Sequence[int]) -> list[str]:
    """Return unique names at the specified indices in the original name sequence."""
    unique_names = make_unique_names(names)
    return [unique_names[index] for index in indices]
