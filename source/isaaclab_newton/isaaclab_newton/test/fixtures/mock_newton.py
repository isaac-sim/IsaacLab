# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton fixture components used by isolated tests and benchmarks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import warp as wp


class MockNewtonModel:
    """Mock Newton model that provides gravity and articulation topology."""

    def __init__(
        self,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        device: str = "cpu",
        num_instances: int = 1,
        num_bodies: int = 1,
        num_joints: int = 0,
        is_fixed_base: bool = False,
    ):
        self.world_count = num_instances
        self._gravity = wp.array([gravity] * (num_instances + 1), dtype=wp.vec3f, device=device)
        num_dofs = num_joints + (0 if is_fixed_base else 6)
        self.articulation_count = num_instances
        self.max_joints_per_articulation = num_bodies
        self.max_dofs_per_articulation = num_dofs
        self.joint_dof_count = num_instances * num_dofs
        self.body_count = num_instances * num_bodies

    @property
    def gravity(self):
        return self._gravity


def create_mock_newton_manager(
    patch_path: str,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    num_instances: int = 1,
    num_bodies: int = 1,
    num_joints: int = 0,
    is_fixed_base: bool = False,
):
    """Create a mock NewtonManager for testing.

    Args:
        patch_path: The module path to patch
            (e.g., "isaaclab_newton.assets.articulation.articulation_data.NewtonManager").
        gravity: Gravity vector to use for the mock model.
        num_instances: Number of articulation instances in the mock model.
        num_bodies: Number of bodies in each mock articulation.
        num_joints: Number of joint degrees of freedom in each mock articulation.
        is_fixed_base: Whether the mock articulation has a fixed base.

    Returns:
        A context manager that patches the NewtonManager.
    """
    mock_model = MockNewtonModel(
        gravity,
        num_instances=num_instances,
        num_bodies=num_bodies,
        num_joints=num_joints,
        is_fixed_base=is_fixed_base,
    )
    mock_state = MagicMock()
    mock_control = MagicMock()

    return patch(
        patch_path,
        **{
            "get_model.return_value": mock_model,
            "get_state_0.return_value": mock_state,
            "get_control.return_value": mock_control,
            "get_dt.return_value": 0.01,
        },
    )
