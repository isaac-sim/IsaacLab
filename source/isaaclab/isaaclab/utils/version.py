# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility function for version comparison."""


def compare_versions(v1: str, v2: str) -> int:
    parts1 = list(map(int, v1.split(".")))
    parts2 = list(map(int, v2.split(".")))

    # Pad the shorter version with zeros (e.g. 1.2 vs 1.2.0)
    length = max(len(parts1), len(parts2))
    parts1 += [0] * (length - len(parts1))
    parts2 += [0] * (length - len(parts2))

    if parts1 > parts2:
        return 1  # v1 is greater
    elif parts1 < parts2:
        return -1  # v2 is greater
    else:
        return 0  # versions are equal
