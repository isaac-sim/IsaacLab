# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for factory backend switching between torch and warp."""

from isaaclab_physx.test.mock_interfaces.factories import (
    create_mock_articulation_view,
    create_mock_humanoid_view,
    create_mock_quadruped_view,
    create_mock_rigid_body_view,
    create_mock_rigid_contact_view,
)
from isaaclab_physx.test.mock_interfaces.views import (
    MockArticulationView,
    MockArticulationViewWarp,
    MockRigidBodyView,
    MockRigidBodyViewWarp,
    MockRigidContactView,
    MockRigidContactViewWarp,
)


class TestRigidBodyViewBackend:
    """Tests for rigid body view backend switching."""

    def test_factory_creates_torch_by_default(self):
        """Test that factory creates torch backend by default."""
        view = create_mock_rigid_body_view(count=4)
        assert isinstance(view, MockRigidBodyView)
        assert not isinstance(view, MockRigidBodyViewWarp)
        assert view._backend == "torch"

    def test_factory_creates_torch_when_specified(self):
        """Test that factory creates torch backend when explicitly specified."""
        view = create_mock_rigid_body_view(count=4, backend="torch")
        assert isinstance(view, MockRigidBodyView)
        assert not isinstance(view, MockRigidBodyViewWarp)
        assert view._backend == "torch"

    def test_factory_creates_warp_when_specified(self):
        """Test that factory creates warp backend when specified."""
        view = create_mock_rigid_body_view(count=4, backend="warp")
        assert isinstance(view, MockRigidBodyViewWarp)
        assert not isinstance(view, MockRigidBodyView)
        assert view._backend == "warp"


class TestArticulationViewBackend:
    """Tests for articulation view backend switching."""

    def test_factory_creates_torch_by_default(self):
        """Test that factory creates torch backend by default."""
        view = create_mock_articulation_view(count=4, num_dofs=12, num_links=13)
        assert isinstance(view, MockArticulationView)
        assert not isinstance(view, MockArticulationViewWarp)

    def test_factory_creates_torch_when_specified(self):
        """Test that factory creates torch backend when explicitly specified."""
        view = create_mock_articulation_view(count=4, num_dofs=12, num_links=13, backend="torch")
        assert isinstance(view, MockArticulationView)
        assert not isinstance(view, MockArticulationViewWarp)

    def test_factory_creates_warp_when_specified(self):
        """Test that factory creates warp backend when specified."""
        view = create_mock_articulation_view(count=4, num_dofs=12, num_links=13, backend="warp")
        assert isinstance(view, MockArticulationViewWarp)
        assert not isinstance(view, MockArticulationView)


class TestRigidContactViewBackend:
    """Tests for rigid contact view backend switching."""

    def test_factory_creates_torch_by_default(self):
        """Test that factory creates torch backend by default."""
        view = create_mock_rigid_contact_view(count=4, num_bodies=5, filter_count=3)
        assert isinstance(view, MockRigidContactView)
        assert not isinstance(view, MockRigidContactViewWarp)

    def test_factory_creates_torch_when_specified(self):
        """Test that factory creates torch backend when explicitly specified."""
        view = create_mock_rigid_contact_view(count=4, num_bodies=5, filter_count=3, backend="torch")
        assert isinstance(view, MockRigidContactView)
        assert not isinstance(view, MockRigidContactViewWarp)

    def test_factory_creates_warp_when_specified(self):
        """Test that factory creates warp backend when specified."""
        view = create_mock_rigid_contact_view(count=4, num_bodies=5, filter_count=3, backend="warp")
        assert isinstance(view, MockRigidContactViewWarp)
        assert not isinstance(view, MockRigidContactView)


class TestQuadrupedViewBackend:
    """Tests for quadruped view backend switching."""

    def test_factory_creates_torch_by_default(self):
        """Test that factory creates torch backend by default."""
        view = create_mock_quadruped_view(count=4)
        assert isinstance(view, MockArticulationView)
        assert not isinstance(view, MockArticulationViewWarp)

    def test_factory_creates_warp_when_specified(self):
        """Test that factory creates warp backend when specified."""
        view = create_mock_quadruped_view(count=4, backend="warp")
        assert isinstance(view, MockArticulationViewWarp)
        assert not isinstance(view, MockArticulationView)
        # Verify quadruped structure is preserved
        assert view.shared_metatype.dof_count == 12
        assert view.shared_metatype.link_count == 13


class TestHumanoidViewBackend:
    """Tests for humanoid view backend switching."""

    def test_factory_creates_torch_by_default(self):
        """Test that factory creates torch backend by default."""
        view = create_mock_humanoid_view(count=2)
        assert isinstance(view, MockArticulationView)
        assert not isinstance(view, MockArticulationViewWarp)

    def test_factory_creates_warp_when_specified(self):
        """Test that factory creates warp backend when specified."""
        view = create_mock_humanoid_view(count=2, backend="warp")
        assert isinstance(view, MockArticulationViewWarp)
        assert not isinstance(view, MockArticulationView)
        # Verify humanoid structure is preserved
        assert view.shared_metatype.dof_count == 21
        assert view.shared_metatype.link_count == 22


class TestBackendConsistency:
    """Tests for backend consistency across views."""

    def test_warp_views_have_backend_attribute(self):
        """Test that warp view types have _backend attribute set to 'warp'."""
        warp_rb = create_mock_rigid_body_view(count=1, backend="warp")
        warp_art = create_mock_articulation_view(count=1, backend="warp")
        warp_contact = create_mock_rigid_contact_view(count=1, num_bodies=1, filter_count=0, backend="warp")

        assert hasattr(warp_rb, "_backend")
        assert hasattr(warp_art, "_backend")
        assert hasattr(warp_contact, "_backend")

        assert warp_rb._backend == "warp"
        assert warp_art._backend == "warp"
        assert warp_contact._backend == "warp"

    def test_torch_rigid_body_view_has_backend_attribute(self):
        """Test that torch MockRigidBodyView has _backend attribute."""
        torch_rb = create_mock_rigid_body_view(count=1, backend="torch")
        assert hasattr(torch_rb, "_backend")
        assert torch_rb._backend == "torch"
