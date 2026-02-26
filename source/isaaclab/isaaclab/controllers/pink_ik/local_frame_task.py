# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated compatibility shim for Pink task imports.

Prefer importing from ``isaaclab.controllers.pink_ik.pink_tasks``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`isaaclab.controllers.pink_ik.local_frame_task` is deprecated; "
    "import from `isaaclab.controllers.pink_ik.pink_tasks` instead.",
    DeprecationWarning,
    stacklevel=2,
)
