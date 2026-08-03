# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared Newton simulation mocks."""

from isaaclab_newton.test.mock_interfaces import MockNewtonModel


def test_mock_newton_model_derives_articulation_topology() -> None:
    """Derive the model-wide scratch-buffer topology from the mock articulation shape."""
    model = MockNewtonModel(num_instances=3, num_bodies=7, num_joints=6)

    assert model.articulation_count == 3
    assert model.max_joints_per_articulation == 7
    assert model.max_dofs_per_articulation == 12
    assert model.joint_dof_count == 36
    assert model.body_count == 21
