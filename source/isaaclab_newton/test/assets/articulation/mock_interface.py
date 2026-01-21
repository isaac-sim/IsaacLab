# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock interfaces for testing and benchmarking Articulation and ArticulationData classes.

This module re-exports shared mock classes from the common module and provides
articulation-specific factory functions.
"""

from __future__ import annotations

# Re-export all shared mock classes for backward compatibility
from common.mock_newton import (
    MockArticulationTensorAPI,
    MockNewtonArticulationView,
    MockNewtonModel,
    MockSharedMetaDataType,
)
from common.mock_newton import create_mock_newton_manager as _create_mock_newton_manager

__all__ = [
    "MockNewtonModel",
    "MockNewtonArticulationView",
    "MockSharedMetaDataType",
    "MockArticulationTensorAPI",
    "create_mock_newton_manager",
]


def create_mock_newton_manager(gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)):
    """Create a mock NewtonManager for testing ArticulationData.

    Args:
        gravity: Gravity vector to use for the mock model.

    Returns:
        A context manager that patches the NewtonManager for ArticulationData.
    """
    return _create_mock_newton_manager(
        "isaaclab_newton.assets.articulation.articulation_data.NewtonManager",
        gravity,
    )
