# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton mock factory functions."""

from isaaclab_newton.test.mock_interfaces.factories import (
    create_mock_articulation_view,
    create_mock_humanoid_view,
    create_mock_quadruped_view,
)
from isaaclab_newton.test.mock_interfaces.views import MockNewtonArticulationView


class TestFactories:
    """Tests for Newton mock factory functions."""

    def test_create_mock_articulation_view_basic(self):
        """Test basic factory usage."""
        view = create_mock_articulation_view(count=4, num_joints=12, num_bodies=13)
        assert view.count == 4
        assert view.joint_dof_count == 12
        assert view.link_count == 13
        assert isinstance(view, MockNewtonArticulationView)

    def test_create_mock_articulation_view_fixed_base(self):
        """Test factory with fixed base flag."""
        view = create_mock_articulation_view(count=2, num_joints=6, num_bodies=7, is_fixed_base=True)
        assert view.count == 2
        assert view.is_fixed_base is True

    def test_create_mock_articulation_view_custom_names(self):
        """Test factory with custom names."""
        joint_names = ["j1", "j2"]
        body_names = ["b1", "b2", "b3"]
        view = create_mock_articulation_view(
            count=1, num_joints=2, num_bodies=3, joint_names=joint_names, body_names=body_names
        )
        assert view.joint_dof_names == joint_names
        assert view.body_names == body_names

    def test_create_mock_quadruped_view(self):
        """Test quadruped factory."""
        view = create_mock_quadruped_view(count=4)
        assert view.count == 4
        assert view.joint_dof_count == 12
        assert view.link_count == 13
        assert view.is_fixed_base is False
        assert "FL_hip_joint" in view.joint_dof_names
        assert "base" in view.body_names
        assert isinstance(view, MockNewtonArticulationView)

    def test_create_mock_humanoid_view(self):
        """Test humanoid factory."""
        view = create_mock_humanoid_view(count=2)
        assert view.count == 2
        assert view.joint_dof_count == 21
        assert view.link_count == 22
        assert view.is_fixed_base is False
        assert "left_elbow" in view.joint_dof_names
        assert "pelvis" in view.body_names
        assert isinstance(view, MockNewtonArticulationView)
