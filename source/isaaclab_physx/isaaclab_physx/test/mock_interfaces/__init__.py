# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock interfaces for PhysX TensorAPI views.

This module provides mock implementations of PhysX TensorAPI views for unit testing
without requiring Isaac Sim or GPU simulation.
"""

from .factories import (
    create_mock_articulation_view,
    create_mock_humanoid_view,
    create_mock_quadruped_view,
    create_mock_rigid_body_view,
    create_mock_rigid_contact_view,
)
from .views import MockArticulationView, MockRigidBodyView, MockRigidContactView

__all__ = [
    # Views
    "MockRigidBodyView",
    "MockArticulationView",
    "MockRigidContactView",
    # Factories
    "create_mock_rigid_body_view",
    "create_mock_articulation_view",
    "create_mock_rigid_contact_view",
    "create_mock_quadruped_view",
    "create_mock_humanoid_view",
]
