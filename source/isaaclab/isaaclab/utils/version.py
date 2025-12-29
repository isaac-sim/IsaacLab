# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utility function for versioning."""


def compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings and return the comparison result.

    The version strings are expected to be in the format "x.y.z" where x, y,
    and z are integers. The version strings are compared lexicographically.

    Args:
        v1: The first version string.
        v2: The second version string.

    Returns:
        An integer indicating the comparison result:

        - :attr:`1` if v1 is greater
        - :attr:`-1` if v2 is greater
        - :attr:`0` if v1 and v2 are equal
    """
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
