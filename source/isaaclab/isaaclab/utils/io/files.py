# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local file lookup utilities."""

import glob
import os


def latest_file(directory: str, pattern: str = "*") -> str | None:
    """Return the most recently modified entry in a directory matching a glob pattern, or ``None``.

    Args:
        directory: The directory to search.
        pattern: A :mod:`glob` pattern relative to ``directory``. It may descend into sub-directories.
    """
    matches = glob.glob(os.path.join(directory, pattern))
    return max(matches, key=os.path.getmtime) if matches else None
