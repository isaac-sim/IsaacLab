# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared physical constants for cube-stacking tasks."""

FRANKA_STACK_ARM_WORKSPACE_LOWER: tuple[float, ...] = (
    -0.303,
    -0.200,
    -0.328,
    -2.759,
    -0.124,
    2.393,
    0.271,
)
"""Lower joint-position boundary of the validated Franka stacking workspace [rad]."""

FRANKA_STACK_ARM_WORKSPACE_UPPER: tuple[float, ...] = (
    0.450,
    0.603,
    0.147,
    -2.000,
    0.345,
    3.112,
    1.200,
)
"""Upper joint-position boundary of the validated Franka stacking workspace [rad]."""
