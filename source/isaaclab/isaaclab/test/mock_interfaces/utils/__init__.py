# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for creating and using mock interfaces."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .mock_generator import MockArticulationBuilder, MockSensorBuilder
    from .mock_wrench_composer import MockWrenchComposer
    from .patching import patch_articulation, patch_sensor

from isaaclab.utils.module import lazy_export

lazy_export(
    ("mock_generator", ["MockArticulationBuilder", "MockSensorBuilder"]),
    ("mock_wrench_composer", "MockWrenchComposer"),
    ("patching", ["patch_articulation", "patch_sensor"]),
)
