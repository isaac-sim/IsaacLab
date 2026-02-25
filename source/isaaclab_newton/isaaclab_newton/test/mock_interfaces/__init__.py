# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock interfaces for Newton simulation views.

This module provides mock implementations of Newton simulation components for unit testing
without requiring an actual simulation environment.
"""

from .factories import (
    create_mock_articulation_view,
    create_mock_humanoid_view,
    create_mock_quadruped_view,
)
from .mock_newton import (
    MockNewtonContactSensor,
    MockNewtonModel,
    MockWrenchComposer,
    create_mock_newton_manager,
)
from .views import MockNewtonArticulationView

__all__ = [
    # Views
    "MockNewtonArticulationView",
    # Other mocks
    "MockNewtonModel",
    "MockWrenchComposer",
    "MockNewtonContactSensor",
    # Factory functions
    "create_mock_articulation_view",
    "create_mock_quadruped_view",
    "create_mock_humanoid_view",
    "create_mock_newton_manager",
]
