# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MockRigidContactViewWarp."""

import numpy as np
import pytest
import warp as wp
from isaaclab_physx.test.mock_interfaces.factories import create_mock_rigid_contact_view
from isaaclab_physx.test.mock_interfaces.views import MockRigidContactViewWarp


class TestMockRigidContactViewWarpInit:
    """Tests for MockRigidContactViewWarp initialization."""

    def test_default_init(self):
        """Test default initialization."""
        view = MockRigidContactViewWarp()
        assert view.count == 1
        assert view.num_bodies == 1
        assert view.filter_count == 0
        assert view._backend == "warp"

    def test_custom_configuration(self):
        """Test initialization with custom configuration."""
        view = MockRigidContactViewWarp(
            count=4,
            num_bodies=5,
            filter_count=3,
        )
        assert view.count == 4
        assert view.num_bodies == 5
        assert view.filter_count == 3


class TestMockRigidContactViewWarpGetters:
    """Tests for MockRigidContactViewWarp getter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 5 bodies, 3 filters."""
        return MockRigidContactViewWarp(count=4, num_bodies=5, filter_count=3, device="cpu")

    def test_get_net_contact_forces_shape(self, view):
        """Test net contact forces shape - should be (N*B, 3) with wp.float32 dtype."""
        forces = view.get_net_contact_forces(dt=0.01)
        # Shape: (N*B, 3) = (4*5, 3) = (20, 3)
        assert forces.shape == (20, 3)
        assert forces.dtype == wp.float32

    def test_get_contact_force_matrix_shape(self, view):
        """Test contact force matrix shape - should be (N*B, F, 3) with wp.float32 dtype."""
        matrix = view.get_contact_force_matrix(dt=0.01)
        # Shape: (N*B, F, 3) = (4*5, 3, 3) = (20, 3, 3)
        assert matrix.shape == (20, 3, 3)
        assert matrix.dtype == wp.float32

    def test_get_contact_data_shapes(self, view):
        """Test contact data tuple shapes."""
        data = view.get_contact_data(dt=0.01)
        assert len(data) == 6
        positions, normals, impulses, separations, num_found, patch_id = data

        total_bodies = 4 * 5  # count * num_bodies
        max_contacts = 16  # default

        assert positions.shape == (total_bodies, max_contacts, 3)
        assert positions.dtype == wp.float32
        assert normals.shape == (total_bodies, max_contacts, 3)
        assert normals.dtype == wp.float32
        assert impulses.shape == (total_bodies, max_contacts, 3)
        assert impulses.dtype == wp.float32
        assert separations.shape == (total_bodies, max_contacts)
        assert separations.dtype == wp.float32
        assert num_found.shape == (total_bodies,)
        assert num_found.dtype == wp.int32
        assert patch_id.shape == (total_bodies, max_contacts)
        assert patch_id.dtype == wp.int32

    def test_get_friction_data_shapes(self, view):
        """Test friction data tuple shapes."""
        data = view.get_friction_data(dt=0.01)
        assert len(data) == 4
        forces, impulses, points, patch_id = data

        total_bodies = 4 * 5
        max_contacts = 16

        assert forces.shape == (total_bodies, max_contacts, 3)
        assert forces.dtype == wp.float32
        assert impulses.shape == (total_bodies, max_contacts, 3)
        assert impulses.dtype == wp.float32
        assert points.shape == (total_bodies, max_contacts, 3)
        assert points.dtype == wp.float32
        assert patch_id.shape == (total_bodies, max_contacts)
        assert patch_id.dtype == wp.int32

    def test_getters_return_clones(self, view):
        """Test that getters return clones, not references."""
        forces1 = view.get_net_contact_forces(0.01)
        forces1_np = forces1.numpy()
        forces1_np[0, 0] = 999.0
        forces2 = view.get_net_contact_forces(0.01)
        forces2_np = forces2.numpy()
        assert forces2_np[0, 0] != 999.0


class TestMockRigidContactViewWarpMockSetters:
    """Tests for MockRigidContactViewWarp mock setter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 5 bodies, 3 filters."""
        return MockRigidContactViewWarp(count=4, num_bodies=5, filter_count=3, device="cpu")

    def test_set_mock_net_contact_forces(self, view):
        """Test mock net contact forces setter."""
        mock_data = np.random.randn(20, 3).astype(np.float32)
        mock_forces = wp.array(mock_data, dtype=wp.float32, device="cpu")
        view.set_mock_net_contact_forces(mock_forces)
        result = view.get_net_contact_forces(0.01)
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, mock_data, rtol=1e-5)

    def test_set_mock_contact_force_matrix(self, view):
        """Test mock contact force matrix setter."""
        mock_data = np.random.randn(20, 3, 3).astype(np.float32)
        mock_matrix = wp.array(mock_data, dtype=wp.float32, device="cpu")
        view.set_mock_contact_force_matrix(mock_matrix)
        result = view.get_contact_force_matrix(0.01)
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, mock_data, rtol=1e-5)

    def test_set_mock_contact_data_partial(self, view):
        """Test setting partial contact data."""
        mock_positions_data = np.random.randn(20, 16, 3).astype(np.float32)
        mock_normals_data = np.random.randn(20, 16, 3).astype(np.float32)
        mock_positions = wp.array(mock_positions_data, dtype=wp.float32, device="cpu")
        mock_normals = wp.array(mock_normals_data, dtype=wp.float32, device="cpu")
        view.set_mock_contact_data(positions=mock_positions, normals=mock_normals)

        positions, normals, _, _, _, _ = view.get_contact_data(0.01)
        positions_np = positions.numpy()
        normals_np = normals.numpy()
        np.testing.assert_allclose(positions_np, mock_positions_data, rtol=1e-5)
        np.testing.assert_allclose(normals_np, mock_normals_data, rtol=1e-5)

    def test_set_mock_contact_data_full(self, view):
        """Test setting full contact data."""
        total_bodies = 20
        max_contacts = 16

        mock_positions_data = np.random.randn(total_bodies, max_contacts, 3).astype(np.float32)
        mock_normals_data = np.random.randn(total_bodies, max_contacts, 3).astype(np.float32)
        mock_impulses_data = np.random.randn(total_bodies, max_contacts, 3).astype(np.float32)
        mock_separations_data = np.random.randn(total_bodies, max_contacts).astype(np.float32)
        mock_num_found_data = np.random.randint(0, max_contacts, (total_bodies,), dtype=np.int32)
        mock_patch_id_data = np.random.randint(0, 10, (total_bodies, max_contacts), dtype=np.int32)

        mock_positions = wp.array(mock_positions_data, dtype=wp.float32, device="cpu")
        mock_normals = wp.array(mock_normals_data, dtype=wp.float32, device="cpu")
        mock_impulses = wp.array(mock_impulses_data, dtype=wp.float32, device="cpu")
        mock_separations = wp.array(mock_separations_data, dtype=wp.float32, device="cpu")
        mock_num_found = wp.array(mock_num_found_data, dtype=wp.int32, device="cpu")
        mock_patch_id = wp.array(mock_patch_id_data, dtype=wp.int32, device="cpu")

        view.set_mock_contact_data(
            positions=mock_positions,
            normals=mock_normals,
            impulses=mock_impulses,
            separations=mock_separations,
            num_found=mock_num_found,
            patch_id=mock_patch_id,
        )

        positions, normals, impulses, separations, num_found, patch_id = view.get_contact_data(0.01)

        np.testing.assert_allclose(positions.numpy(), mock_positions_data, rtol=1e-5)
        np.testing.assert_allclose(normals.numpy(), mock_normals_data, rtol=1e-5)
        np.testing.assert_allclose(impulses.numpy(), mock_impulses_data, rtol=1e-5)
        np.testing.assert_allclose(separations.numpy(), mock_separations_data, rtol=1e-5)
        np.testing.assert_array_equal(num_found.numpy(), mock_num_found_data)
        np.testing.assert_array_equal(patch_id.numpy(), mock_patch_id_data)

    def test_set_mock_friction_data(self, view):
        """Test setting friction data."""
        total_bodies = 20
        max_contacts = 16

        mock_forces_data = np.random.randn(total_bodies, max_contacts, 3).astype(np.float32)
        mock_impulses_data = np.random.randn(total_bodies, max_contacts, 3).astype(np.float32)

        mock_forces = wp.array(mock_forces_data, dtype=wp.float32, device="cpu")
        mock_impulses = wp.array(mock_impulses_data, dtype=wp.float32, device="cpu")

        view.set_mock_friction_data(forces=mock_forces, impulses=mock_impulses)

        forces, impulses, _, _ = view.get_friction_data(0.01)
        np.testing.assert_allclose(forces.numpy(), mock_forces_data, rtol=1e-5)
        np.testing.assert_allclose(impulses.numpy(), mock_impulses_data, rtol=1e-5)


class TestMockRigidContactViewWarpFactory:
    """Tests for create_mock_rigid_contact_view factory with warp backend."""

    def test_factory_basic(self):
        """Test basic factory usage with warp backend."""
        view = create_mock_rigid_contact_view(count=4, num_bodies=5, filter_count=3, backend="warp")
        assert view.count == 4
        assert view.num_bodies == 5
        assert view.filter_count == 3
        assert isinstance(view, MockRigidContactViewWarp)

    def test_factory_custom_max_contacts(self):
        """Test factory with custom max contact data count."""
        view = create_mock_rigid_contact_view(count=2, num_bodies=3, max_contact_data_count=32, backend="warp")
        _, _, _, separations, _, _ = view.get_contact_data(0.01)
        assert separations.shape[1] == 32
        assert isinstance(view, MockRigidContactViewWarp)


class TestMockRigidContactViewWarpZeroFilterCount:
    """Tests for MockRigidContactViewWarp with zero filter count."""

    def test_contact_force_matrix_zero_filters(self):
        """Test contact force matrix with zero filters."""
        view = MockRigidContactViewWarp(count=4, num_bodies=5, filter_count=0)
        matrix = view.get_contact_force_matrix(0.01)
        # Shape: (N*B, F, 3) = (20, 0, 3)
        assert matrix.shape == (20, 0, 3)
