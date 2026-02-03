# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MockRigidContactView."""

import pytest
import torch
from isaaclab_physx.test.mock_interfaces.factories import create_mock_rigid_contact_view
from isaaclab_physx.test.mock_interfaces.views import MockRigidContactView


class TestMockRigidContactViewInit:
    """Tests for MockRigidContactView initialization."""

    def test_default_init(self):
        """Test default initialization."""
        view = MockRigidContactView()
        assert view.count == 1
        assert view.num_bodies == 1
        assert view.filter_count == 0

    def test_custom_configuration(self):
        """Test initialization with custom configuration."""
        view = MockRigidContactView(
            count=4,
            num_bodies=5,
            filter_count=3,
        )
        assert view.count == 4
        assert view.num_bodies == 5
        assert view.filter_count == 3


class TestMockRigidContactViewGetters:
    """Tests for MockRigidContactView getter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 5 bodies, 3 filters."""
        return MockRigidContactView(count=4, num_bodies=5, filter_count=3, device="cpu")

    def test_get_net_contact_forces_shape(self, view):
        """Test net contact forces shape."""
        forces = view.get_net_contact_forces(dt=0.01)
        # Shape: (N*B, 3) = (4*5, 3) = (20, 3)
        assert forces.shape == (20, 3)

    def test_get_contact_force_matrix_shape(self, view):
        """Test contact force matrix shape."""
        matrix = view.get_contact_force_matrix(dt=0.01)
        # Shape: (N*B, F, 3) = (4*5, 3, 3) = (20, 3, 3)
        assert matrix.shape == (20, 3, 3)

    def test_get_contact_data_shapes(self, view):
        """Test contact data tuple shapes."""
        data = view.get_contact_data(dt=0.01)
        assert len(data) == 6
        positions, normals, impulses, separations, num_found, patch_id = data

        total_bodies = 4 * 5  # count * num_bodies
        max_contacts = 16  # default

        assert positions.shape == (total_bodies, max_contacts, 3)
        assert normals.shape == (total_bodies, max_contacts, 3)
        assert impulses.shape == (total_bodies, max_contacts, 3)
        assert separations.shape == (total_bodies, max_contacts)
        assert num_found.shape == (total_bodies,)
        assert patch_id.shape == (total_bodies, max_contacts)

    def test_get_friction_data_shapes(self, view):
        """Test friction data tuple shapes."""
        data = view.get_friction_data(dt=0.01)
        assert len(data) == 4
        forces, impulses, points, patch_id = data

        total_bodies = 4 * 5
        max_contacts = 16

        assert forces.shape == (total_bodies, max_contacts, 3)
        assert impulses.shape == (total_bodies, max_contacts, 3)
        assert points.shape == (total_bodies, max_contacts, 3)
        assert patch_id.shape == (total_bodies, max_contacts)

    def test_getters_return_clones(self, view):
        """Test that getters return clones, not references."""
        forces1 = view.get_net_contact_forces(0.01)
        forces1[0, 0] = 999.0
        forces2 = view.get_net_contact_forces(0.01)
        assert forces2[0, 0] != 999.0


class TestMockRigidContactViewMockSetters:
    """Tests for MockRigidContactView mock setter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 5 bodies, 3 filters."""
        return MockRigidContactView(count=4, num_bodies=5, filter_count=3, device="cpu")

    def test_set_mock_net_contact_forces(self, view):
        """Test mock net contact forces setter."""
        mock_data = torch.randn(20, 3)
        view.set_mock_net_contact_forces(mock_data)
        result = view.get_net_contact_forces(0.01)
        assert torch.allclose(result, mock_data)

    def test_set_mock_contact_force_matrix(self, view):
        """Test mock contact force matrix setter."""
        mock_data = torch.randn(20, 3, 3)
        view.set_mock_contact_force_matrix(mock_data)
        result = view.get_contact_force_matrix(0.01)
        assert torch.allclose(result, mock_data)

    def test_set_mock_contact_data_partial(self, view):
        """Test setting partial contact data."""
        mock_positions = torch.randn(20, 16, 3)
        mock_normals = torch.randn(20, 16, 3)
        view.set_mock_contact_data(positions=mock_positions, normals=mock_normals)

        positions, normals, _, _, _, _ = view.get_contact_data(0.01)
        assert torch.allclose(positions, mock_positions)
        assert torch.allclose(normals, mock_normals)

    def test_set_mock_contact_data_full(self, view):
        """Test setting full contact data."""
        total_bodies = 20
        max_contacts = 16

        mock_positions = torch.randn(total_bodies, max_contacts, 3)
        mock_normals = torch.randn(total_bodies, max_contacts, 3)
        mock_impulses = torch.randn(total_bodies, max_contacts, 3)
        mock_separations = torch.randn(total_bodies, max_contacts)
        mock_num_found = torch.randint(0, max_contacts, (total_bodies,), dtype=torch.int32)
        mock_patch_id = torch.randint(0, 10, (total_bodies, max_contacts), dtype=torch.int32)

        view.set_mock_contact_data(
            positions=mock_positions,
            normals=mock_normals,
            impulses=mock_impulses,
            separations=mock_separations,
            num_found=mock_num_found,
            patch_id=mock_patch_id,
        )

        positions, normals, impulses, separations, num_found, patch_id = view.get_contact_data(0.01)

        assert torch.allclose(positions, mock_positions)
        assert torch.allclose(normals, mock_normals)
        assert torch.allclose(impulses, mock_impulses)
        assert torch.allclose(separations, mock_separations)
        assert torch.equal(num_found, mock_num_found)
        assert torch.equal(patch_id, mock_patch_id)

    def test_set_mock_friction_data(self, view):
        """Test setting friction data."""
        total_bodies = 20
        max_contacts = 16

        mock_forces = torch.randn(total_bodies, max_contacts, 3)
        mock_impulses = torch.randn(total_bodies, max_contacts, 3)

        view.set_mock_friction_data(forces=mock_forces, impulses=mock_impulses)

        forces, impulses, _, _ = view.get_friction_data(0.01)
        assert torch.allclose(forces, mock_forces)
        assert torch.allclose(impulses, mock_impulses)


class TestMockRigidContactViewFactory:
    """Tests for create_mock_rigid_contact_view factory function."""

    def test_factory_basic(self):
        """Test basic factory usage."""
        view = create_mock_rigid_contact_view(count=4, num_bodies=5, filter_count=3)
        assert view.count == 4
        assert view.num_bodies == 5
        assert view.filter_count == 3

    def test_factory_custom_max_contacts(self):
        """Test factory with custom max contact data count."""
        view = create_mock_rigid_contact_view(count=2, num_bodies=3, max_contact_data_count=32)
        _, _, _, separations, _, _ = view.get_contact_data(0.01)
        assert separations.shape[1] == 32


class TestMockRigidContactViewZeroFilterCount:
    """Tests for MockRigidContactView with zero filter count."""

    def test_contact_force_matrix_zero_filters(self):
        """Test contact force matrix with zero filters."""
        view = MockRigidContactView(count=4, num_bodies=5, filter_count=0)
        matrix = view.get_contact_force_matrix(0.01)
        # Shape: (N*B, F, 3) = (20, 0, 3)
        assert matrix.shape == (20, 0, 3)
