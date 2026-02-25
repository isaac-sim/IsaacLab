# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MockRigidBodyViewWarp."""

import numpy as np
import pytest
import warp as wp
from isaaclab_physx.test.mock_interfaces.factories import create_mock_rigid_body_view
from isaaclab_physx.test.mock_interfaces.views import MockRigidBodyViewWarp


class TestMockRigidBodyViewWarpInit:
    """Tests for MockRigidBodyViewWarp initialization."""

    def test_default_init(self):
        """Test default initialization."""
        view = MockRigidBodyViewWarp()
        assert view.count == 1
        assert len(view.prim_paths) == 1
        assert view._backend == "warp"

    def test_custom_count(self):
        """Test initialization with custom count."""
        view = MockRigidBodyViewWarp(count=4)
        assert view.count == 4
        assert len(view.prim_paths) == 4

    def test_custom_prim_paths(self):
        """Test initialization with custom prim paths."""
        paths = ["/World/Body_A", "/World/Body_B"]
        view = MockRigidBodyViewWarp(count=2, prim_paths=paths)
        assert view.prim_paths == paths


class TestMockRigidBodyViewWarpGetters:
    """Tests for MockRigidBodyViewWarp getter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances."""
        return MockRigidBodyViewWarp(count=4, device="cpu")

    def test_get_transforms_shape(self, view):
        """Test transforms shape - should be (N, 7) with wp.float32 dtype."""
        transforms = view.get_transforms()
        assert transforms.shape == (4, 7)
        assert transforms.dtype == wp.float32

    def test_get_transforms_default_quaternion(self, view):
        """Test that default quaternion is identity (xyzw format)."""
        transforms = view.get_transforms()
        transforms_np = transforms.numpy()
        # (N, 7) float32: [pos(3), quat_xyzw(4)]
        for i in range(4):
            pos = transforms_np[i, :3]
            quat = transforms_np[i, 3:]  # xyzw
            np.testing.assert_allclose(pos, [0.0, 0.0, 0.0])
            np.testing.assert_allclose(quat[:3], [0.0, 0.0, 0.0])  # xyz = 0
            np.testing.assert_allclose(quat[3], 1.0)  # w = 1

    def test_get_velocities_shape(self, view):
        """Test velocities shape - should be (N, 6) with wp.float32 dtype."""
        velocities = view.get_velocities()
        assert velocities.shape == (4, 6)
        assert velocities.dtype == wp.float32

    def test_get_accelerations_shape(self, view):
        """Test accelerations shape - should be (N, 6) with wp.float32 dtype."""
        accelerations = view.get_accelerations()
        assert accelerations.shape == (4, 6)
        assert accelerations.dtype == wp.float32

    def test_get_masses_shape(self, view):
        """Test masses shape."""
        masses = view.get_masses()
        assert masses.shape == (4, 1)
        assert masses.dtype == wp.float32

    def test_get_masses_default_value(self, view):
        """Test that default mass is 1."""
        masses = view.get_masses()
        masses_np = masses.numpy()
        np.testing.assert_allclose(masses_np, np.ones((4, 1)))

    def test_get_coms_shape(self, view):
        """Test centers of mass shape - should be (N, 7) with wp.float32 dtype."""
        coms = view.get_coms()
        assert coms.shape == (4, 7)
        assert coms.dtype == wp.float32

    def test_get_inertias_shape(self, view):
        """Test inertias shape."""
        inertias = view.get_inertias()
        assert inertias.shape == (4, 9)
        assert inertias.dtype == wp.float32

    def test_get_inertias_default_value(self, view):
        """Test that default inertia is identity."""
        inertias = view.get_inertias()
        inertias_np = inertias.numpy()
        expected = np.tile(np.eye(3).flatten(), (4, 1))
        np.testing.assert_allclose(inertias_np, expected)

    def test_getters_return_clones(self, view):
        """Test that getters return clones, not references."""
        transforms1 = view.get_transforms()
        transforms1_np = transforms1.numpy()
        transforms1_np[0, 0] = 999.0
        transforms2 = view.get_transforms()
        transforms2_np = transforms2.numpy()
        assert transforms2_np[0, 0] != 999.0


class TestMockRigidBodyViewWarpSetters:
    """Tests for MockRigidBodyViewWarp setter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances."""
        return MockRigidBodyViewWarp(count=4, device="cpu")

    def test_set_transforms(self, view):
        """Test setting transforms."""
        # Create new transforms
        new_data = np.random.randn(4, 7).astype(np.float32)
        new_transforms = wp.array(new_data, dtype=wp.float32, device="cpu")
        view.set_transforms(new_transforms)
        result = view.get_transforms()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, new_data, rtol=1e-5)

    def test_set_transforms_with_indices(self, view):
        """Test setting transforms with indices."""
        new_data = np.random.randn(2, 7).astype(np.float32)
        new_transforms = wp.array(new_data, dtype=wp.float32, device="cpu")
        indices = wp.array([0, 2], dtype=wp.int32, device="cpu")
        view.set_transforms(new_transforms, indices=indices)
        result = view.get_transforms()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np[0], new_data[0], rtol=1e-5)
        np.testing.assert_allclose(result_np[2], new_data[1], rtol=1e-5)

    def test_set_velocities(self, view):
        """Test setting velocities."""
        new_data = np.random.randn(4, 6).astype(np.float32)
        new_velocities = wp.array(new_data, dtype=wp.float32, device="cpu")
        view.set_velocities(new_velocities)
        result = view.get_velocities()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, new_data, rtol=1e-5)

    def test_set_masses(self, view):
        """Test setting masses."""
        new_masses_np = np.abs(np.random.randn(4, 1)).astype(np.float32)
        new_masses = wp.array(new_masses_np, dtype=wp.float32, device="cpu")
        view.set_masses(new_masses)
        result = view.get_masses()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, new_masses_np, rtol=1e-5)


class TestMockRigidBodyViewWarpMockSetters:
    """Tests for MockRigidBodyViewWarp mock setter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances."""
        return MockRigidBodyViewWarp(count=4, device="cpu")

    def test_set_mock_transforms(self, view):
        """Test mock transform setter."""
        mock_data = np.random.randn(4, 7).astype(np.float32)
        mock_transforms = wp.array(mock_data, dtype=wp.float32, device="cpu")
        view.set_mock_transforms(mock_transforms)
        result = view.get_transforms()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, mock_data, rtol=1e-5)

    def test_set_mock_velocities(self, view):
        """Test mock velocity setter."""
        mock_data = np.random.randn(4, 6).astype(np.float32)
        mock_velocities = wp.array(mock_data, dtype=wp.float32, device="cpu")
        view.set_mock_velocities(mock_velocities)
        result = view.get_velocities()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, mock_data, rtol=1e-5)

    def test_set_mock_accelerations(self, view):
        """Test mock acceleration setter."""
        mock_data = np.random.randn(4, 6).astype(np.float32)
        mock_accelerations = wp.array(mock_data, dtype=wp.float32, device="cpu")
        view.set_mock_accelerations(mock_accelerations)
        result = view.get_accelerations()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, mock_data, rtol=1e-5)

    def test_set_mock_masses(self, view):
        """Test mock mass setter."""
        mock_data = np.abs(np.random.randn(4, 1)).astype(np.float32)
        mock_masses = wp.array(mock_data, dtype=wp.float32, device="cpu")
        view.set_mock_masses(mock_masses)
        result = view.get_masses()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, mock_data, rtol=1e-5)

    def test_set_mock_coms(self, view):
        """Test mock center of mass setter."""
        mock_data = np.random.randn(4, 7).astype(np.float32)
        mock_coms = wp.array(mock_data, dtype=wp.float32, device="cpu")
        view.set_mock_coms(mock_coms)
        result = view.get_coms()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, mock_data, rtol=1e-5)

    def test_set_mock_inertias(self, view):
        """Test mock inertia setter."""
        mock_data = np.random.randn(4, 9).astype(np.float32)
        mock_inertias = wp.array(mock_data, dtype=wp.float32, device="cpu")
        view.set_mock_inertias(mock_inertias)
        result = view.get_inertias()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, mock_data, rtol=1e-5)


class TestMockRigidBodyViewWarpActions:
    """Tests for MockRigidBodyViewWarp action methods."""

    def test_apply_forces_and_torques_at_position_noop(self):
        """Test that apply_forces_and_torques_at_position is a no-op."""
        view = MockRigidBodyViewWarp(count=4)
        # Should not raise
        forces = wp.zeros((4, 3), dtype=wp.float32, device="cpu")
        torques = wp.zeros((4, 3), dtype=wp.float32, device="cpu")
        positions = wp.zeros((4, 3), dtype=wp.float32, device="cpu")
        view.apply_forces_and_torques_at_position(
            forces=forces,
            torques=torques,
            positions=positions,
        )


class TestMockRigidBodyViewWarpFactory:
    """Tests for create_mock_rigid_body_view factory with warp backend."""

    def test_factory_basic(self):
        """Test basic factory usage with warp backend."""
        view = create_mock_rigid_body_view(count=4, backend="warp")
        assert view.count == 4
        assert isinstance(view, MockRigidBodyViewWarp)

    def test_factory_with_prim_paths(self):
        """Test factory with custom prim paths."""
        paths = ["/World/A", "/World/B"]
        view = create_mock_rigid_body_view(count=2, prim_paths=paths, backend="warp")
        assert view.prim_paths == paths
        assert isinstance(view, MockRigidBodyViewWarp)
