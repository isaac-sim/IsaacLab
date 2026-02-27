# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .newton_replicate import newton_replicate

from isaaclab.utils.module import lazy_export

lazy_export(
    ("newton_replicate", "newton_replicate"),
)
