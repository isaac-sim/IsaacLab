# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock PhysX TensorAPI views."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .mock_articulation_view import MockArticulationView
    from .mock_articulation_view_warp import MockArticulationViewWarp
    from .mock_rigid_body_view import MockRigidBodyView
    from .mock_rigid_body_view_warp import MockRigidBodyViewWarp
    from .mock_rigid_contact_view import MockRigidContactView
    from .mock_rigid_contact_view_warp import MockRigidContactViewWarp

from isaaclab.utils.module import lazy_export

lazy_export(
    ("mock_articulation_view", "MockArticulationView"),
    ("mock_articulation_view_warp", "MockArticulationViewWarp"),
    ("mock_rigid_body_view", "MockRigidBodyView"),
    ("mock_rigid_body_view_warp", "MockRigidBodyViewWarp"),
    ("mock_rigid_contact_view", "MockRigidContactView"),
    ("mock_rigid_contact_view_warp", "MockRigidContactViewWarp"),
)
