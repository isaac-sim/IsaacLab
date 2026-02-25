# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MockRigidBodyView."""

import pytest
import torch
from isaaclab_physx.test.mock_interfaces.factories import create_mock_rigid_body_view
from isaaclab_physx.test.mock_interfaces.views import MockRigidBodyView


class TestMockRigidBodyViewInit:
    """Tests for MockRigidBodyView initialization."""

    def test_default_init(self):
        """Test default initialization."""
        view = MockRigidBodyView()
        assert view.count == 1
        assert len(view.prim_paths) == 1
        assert view._backend == "torch"

    def test_custom_count(self):
        """Test initialization with custom count."""
        view = MockRigidBodyView(count=4)
        assert view.count == 4
        assert len(view.prim_paths) == 4

    def test_custom_prim_paths(self):
        """Test initialization with custom prim paths."""
        paths = ["/World/Body_A", "/World/Body_B"]
        view = MockRigidBodyView(count=2, prim_paths=paths)
        assert view.prim_paths == paths


class TestMockRigidBodyViewGetters:
    """Tests for MockRigidBodyView getter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances."""
        return MockRigidBodyView(count=4, device="cpu")

    def test_get_transforms_shape(self, view):
        """Test transforms shape."""
        transforms = view.get_transforms()
        assert transforms.shape == (4, 7)

    def test_get_transforms_default_quaternion(self, view):
        """Test that default quaternion is identity (xyzw format)."""
        transforms = view.get_transforms()
        # xyzw format: [x, y, z, w] = [0, 0, 0, 1] for identity
        assert torch.allclose(transforms[:, 3:6], torch.zeros(4, 3))  # xyz = 0
        assert torch.allclose(transforms[:, 6], torch.ones(4))  # w = 1

    def test_get_velocities_shape(self, view):
        """Test velocities shape."""
        velocities = view.get_velocities()
        assert velocities.shape == (4, 6)

    def test_get_accelerations_shape(self, view):
        """Test accelerations shape."""
        accelerations = view.get_accelerations()
        assert accelerations.shape == (4, 6)

    def test_get_masses_shape(self, view):
        """Test masses shape."""
        masses = view.get_masses()
        assert masses.shape == (4, 1)

    def test_get_masses_default_value(self, view):
        """Test that default mass is 1."""
        masses = view.get_masses()
        assert torch.allclose(masses, torch.ones(4, 1))

    def test_get_coms_shape(self, view):
        """Test centers of mass shape."""
        coms = view.get_coms()
        assert coms.shape == (4, 7)

    def test_get_inertias_shape(self, view):
        """Test inertias shape."""
        inertias = view.get_inertias()
        assert inertias.shape == (4, 9)

    def test_get_inertias_default_value(self, view):
        """Test that default inertia is identity."""
        inertias = view.get_inertias()
        expected = torch.eye(3).repeat(4, 1).reshape(4, 9)
        assert torch.allclose(inertias, expected)

    def test_getters_return_clones(self, view):
        """Test that getters return clones, not references."""
        transforms1 = view.get_transforms()
        transforms1[0, 0] = 999.0
        transforms2 = view.get_transforms()
        assert transforms2[0, 0] != 999.0


class TestMockRigidBodyViewSetters:
    """Tests for MockRigidBodyView setter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances."""
        return MockRigidBodyView(count=4, device="cpu")

    def test_set_transforms(self, view):
        """Test setting transforms."""
        new_transforms = torch.randn(4, 7)
        view.set_transforms(new_transforms)
        result = view.get_transforms()
        assert torch.allclose(result, new_transforms)

    def test_set_transforms_with_indices(self, view):
        """Test setting transforms with indices."""
        new_transforms = torch.randn(2, 7)
        indices = torch.tensor([0, 2])
        view.set_transforms(new_transforms, indices=indices)
        result = view.get_transforms()
        assert torch.allclose(result[0], new_transforms[0])
        assert torch.allclose(result[2], new_transforms[1])

    def test_set_velocities(self, view):
        """Test setting velocities."""
        new_velocities = torch.randn(4, 6)
        view.set_velocities(new_velocities)
        result = view.get_velocities()
        assert torch.allclose(result, new_velocities)

    def test_set_masses(self, view):
        """Test setting masses."""
        new_masses = torch.randn(4, 1, 1).abs()
        view.set_masses(new_masses)
        result = view.get_masses()
        assert torch.allclose(result, new_masses)


class TestMockRigidBodyViewMockSetters:
    """Tests for MockRigidBodyView mock setter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances."""
        return MockRigidBodyView(count=4, device="cpu")

    def test_set_mock_transforms(self, view):
        """Test mock transform setter."""
        mock_data = torch.randn(4, 7)
        view.set_mock_transforms(mock_data)
        result = view.get_transforms()
        assert torch.allclose(result, mock_data)

    def test_set_mock_velocities(self, view):
        """Test mock velocity setter."""
        mock_data = torch.randn(4, 6)
        view.set_mock_velocities(mock_data)
        result = view.get_velocities()
        assert torch.allclose(result, mock_data)

    def test_set_mock_accelerations(self, view):
        """Test mock acceleration setter."""
        mock_data = torch.randn(4, 6)
        view.set_mock_accelerations(mock_data)
        result = view.get_accelerations()
        assert torch.allclose(result, mock_data)

    def test_set_mock_masses(self, view):
        """Test mock mass setter."""
        mock_data = torch.randn(4, 1, 1).abs()
        view.set_mock_masses(mock_data)
        result = view.get_masses()
        assert torch.allclose(result, mock_data)

    def test_set_mock_coms(self, view):
        """Test mock center of mass setter."""
        mock_data = torch.randn(4, 1, 7)
        view.set_mock_coms(mock_data)
        result = view.get_coms()
        assert torch.allclose(result, mock_data)

    def test_set_mock_inertias(self, view):
        """Test mock inertia setter."""
        mock_data = torch.randn(4, 1, 3, 3)
        view.set_mock_inertias(mock_data)
        result = view.get_inertias()
        assert torch.allclose(result, mock_data)


class TestMockRigidBodyViewActions:
    """Tests for MockRigidBodyView action methods."""

    def test_apply_forces_and_torques_at_position_noop(self):
        """Test that apply_forces_and_torques_at_position is a no-op."""
        view = MockRigidBodyView(count=4)
        # Should not raise
        view.apply_forces_and_torques_at_position(
            forces=torch.randn(4, 3),
            torques=torch.randn(4, 3),
            positions=torch.randn(4, 3),
        )


class TestMockRigidBodyViewFactory:
    """Tests for create_mock_rigid_body_view factory function."""

    def test_factory_basic(self):
        """Test basic factory usage."""
        view = create_mock_rigid_body_view(count=4)
        assert view.count == 4

    def test_factory_with_prim_paths(self):
        """Test factory with custom prim paths."""
        paths = ["/World/A", "/World/B"]
        view = create_mock_rigid_body_view(count=2, prim_paths=paths)
        assert view.prim_paths == paths
