# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment environments for manipulation tasks.

These environments are designed for real-world deployment of manipulation tasks.
They containconfigurations and implementations that have been tested
and deployed on physical robots.

The deploy module includes:
- Reach environments for end-effector pose tracking

"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .reach import *  # noqa: F403

from isaaclab.utils.module import cascading_export

cascading_export(
    submodules=["reach"],
)
