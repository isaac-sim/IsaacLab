# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock interfaces for PhysX TensorAPI views.

This module provides mock implementations of PhysX TensorAPI views for unit testing
without requiring Isaac Sim or GPU simulation.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .factories import create_mock_articulation_view, create_mock_humanoid_view, create_mock_quadruped_view, create_mock_rigid_body_view, create_mock_rigid_contact_view
    from .views import MockArticulationView, MockArticulationViewWarp, MockRigidBodyView, MockRigidBodyViewWarp, MockRigidContactView, MockRigidContactViewWarp

from isaaclab.utils.module import lazy_export

lazy_export(
    ("factories", [
        "create_mock_articulation_view",
        "create_mock_humanoid_view",
        "create_mock_quadruped_view",
        "create_mock_rigid_body_view",
        "create_mock_rigid_contact_view",
    ]),
    ("views", [
        "MockArticulationView",
        "MockArticulationViewWarp",
        "MockRigidBodyView",
        "MockRigidBodyViewWarp",
        "MockRigidContactView",
        "MockRigidContactViewWarp",
    ]),
)
