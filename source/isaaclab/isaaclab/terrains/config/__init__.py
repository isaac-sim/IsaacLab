# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-defined terrain configurations for the terrain generator."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .rough import *  # noqa: F403

from isaaclab.utils.module import cascading_export

cascading_export(
    submodules=["rough"],
)
