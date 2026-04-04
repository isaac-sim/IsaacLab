# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MockArticulationViewWarp."""

import numpy as np
import pytest
import warp as wp
from isaaclab_physx.test.mock_interfaces.factories import (
    create_mock_articulation_view,
    create_mock_humanoid_view,
    create_mock_quadruped_view,
)
from isaaclab_physx.test.mock_interfaces.views import MockArticulationViewWarp


class TestMockArticulationViewWarpInit:
    """Tests for MockArticulationViewWarp initialization."""

    def test_default_init(self):
        """Test default initialization."""
        view = MockArticulationViewWarp()
        assert view.count == 1
        assert view.shared_metatype.dof_count == 1
        assert view.shared_metatype.link_count == 2
        assert view._backend == "warp"

    def test_custom_configuration(self):
        """Test initialization with custom configuration."""
        view = MockArticulationViewWarp(
            count=4,
            num_dofs=12,
            num_links=13,
            fixed_base=True,
        )
        assert view.count == 4
        assert view.shared_metatype.dof_count == 12
        assert view.shared_metatype.link_count == 13
        assert view.shared_metatype.fixed_base is True

    def test_custom_names(self):
        """Test initialization with custom DOF and link names."""
        dof_names = ["joint_0", "joint_1"]
        link_names = ["base", "link_1", "link_2"]
        view = MockArticulationViewWarp(
            num_dofs=2,
            num_links=3,
            dof_names=dof_names,
            link_names=link_names,
        )
        assert view.shared_metatype.dof_names == dof_names
        assert view.shared_metatype.link_names == link_names

    def test_tendon_properties(self):
        """Test tendon properties are zero."""
        view = MockArticulationViewWarp()
        assert view.max_fixed_tendons == 0
        assert view.max_spatial_tendons == 0


class TestMockArticulationViewWarpRootGetters:
    """Tests for MockArticulationViewWarp root getter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockArticulationViewWarp(count=4, num_dofs=12, num_links=13, device="cpu")

    def test_get_root_transforms_shape(self, view):
        """Test root transforms shape - should be (N,) with wp.transformf dtype."""
        transforms = view.get_root_transforms()
        assert transforms.shape == (4,)
        assert transforms.dtype == wp.transformf

    def test_get_root_transforms_default_quaternion(self, view):
        """Test that default quaternion is identity (xyzw format)."""
        transforms = view.get_root_transforms()
        transforms_np = transforms.numpy()
        for i in range(4):
            quat = transforms_np[i, 3:]  # xyzw
            np.testing.assert_allclose(quat[3], 1.0)  # w = 1

    def test_get_root_velocities_shape(self, view):
        """Test root velocities shape - should be (N,) with wp.spatial_vectorf dtype."""
        velocities = view.get_root_velocities()
        assert velocities.shape == (4,)
        assert velocities.dtype == wp.spatial_vectorf


class TestMockArticulationViewWarpLinkGetters:
    """Tests for MockArticulationViewWarp link getter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockArticulationViewWarp(count=4, num_dofs=12, num_links=13, device="cpu")

    def test_get_link_transforms_shape(self, view):
        """Test link transforms shape - should be (N, L) with wp.transformf dtype."""
        transforms = view.get_link_transforms()
        assert transforms.shape == (4, 13)
        assert transforms.dtype == wp.transformf

    def test_get_link_velocities_shape(self, view):
        """Test link velocities shape - should be (N, L) with wp.spatial_vectorf dtype."""
        velocities = view.get_link_velocities()
        assert velocities.shape == (4, 13)
        assert velocities.dtype == wp.spatial_vectorf

    def test_get_link_accelerations_shape(self, view):
        """Test link accelerations shape - should be (N, L) with wp.spatial_vectorf dtype."""
        accelerations = view.get_link_accelerations()
        assert accelerations.shape == (4, 13)
        assert accelerations.dtype == wp.spatial_vectorf

    def test_get_link_incoming_joint_force_shape(self, view):
        """Test link incoming joint force shape - should be (N, L) with wp.spatial_vectorf dtype."""
        forces = view.get_link_incoming_joint_force()
        assert forces.shape == (4, 13)
        assert forces.dtype == wp.spatial_vectorf


class TestMockArticulationViewWarpDOFGetters:
    """Tests for MockArticulationViewWarp DOF getter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockArticulationViewWarp(count=4, num_dofs=12, num_links=13, device="cpu")

    def test_get_dof_positions_shape(self, view):
        """Test DOF positions shape."""
        positions = view.get_dof_positions()
        assert positions.shape == (4, 12)
        assert positions.dtype == wp.float32

    def test_get_dof_velocities_shape(self, view):
        """Test DOF velocities shape."""
        velocities = view.get_dof_velocities()
        assert velocities.shape == (4, 12)
        assert velocities.dtype == wp.float32

    def test_get_dof_projected_joint_forces_shape(self, view):
        """Test projected joint forces shape."""
        forces = view.get_dof_projected_joint_forces()
        assert forces.shape == (4, 12)
        assert forces.dtype == wp.float32

    def test_get_dof_limits_shape(self, view):
        """Test DOF limits shape - should be (N, J) with wp.vec2f dtype."""
        limits = view.get_dof_limits()
        assert limits.shape == (4, 12)
        assert limits.dtype == wp.vec2f

    def test_get_dof_limits_default_values(self, view):
        """Test that default limits are infinite."""
        limits = view.get_dof_limits()
        limits_np = limits.numpy()
        assert np.all(limits_np[:, :, 0] == float("-inf"))  # lower
        assert np.all(limits_np[:, :, 1] == float("inf"))  # upper

    def test_get_dof_stiffnesses_shape(self, view):
        """Test DOF stiffnesses shape."""
        stiffnesses = view.get_dof_stiffnesses()
        assert stiffnesses.shape == (4, 12)
        assert stiffnesses.dtype == wp.float32

    def test_get_dof_dampings_shape(self, view):
        """Test DOF dampings shape."""
        dampings = view.get_dof_dampings()
        assert dampings.shape == (4, 12)
        assert dampings.dtype == wp.float32

    def test_get_dof_max_forces_shape(self, view):
        """Test DOF max forces shape."""
        max_forces = view.get_dof_max_forces()
        assert max_forces.shape == (4, 12)
        assert max_forces.dtype == wp.float32

    def test_get_dof_max_velocities_shape(self, view):
        """Test DOF max velocities shape."""
        max_velocities = view.get_dof_max_velocities()
        assert max_velocities.shape == (4, 12)
        assert max_velocities.dtype == wp.float32

    def test_get_dof_armatures_shape(self, view):
        """Test DOF armatures shape."""
        armatures = view.get_dof_armatures()
        assert armatures.shape == (4, 12)
        assert armatures.dtype == wp.float32

    def test_get_dof_friction_coefficients_shape(self, view):
        """Test DOF friction coefficients shape."""
        friction = view.get_dof_friction_coefficients()
        assert friction.shape == (4, 12)
        assert friction.dtype == wp.float32

    def test_get_dof_friction_properties_shape(self, view):
        """Test DOF friction properties shape."""
        friction_props = view.get_dof_friction_properties()
        assert friction_props.shape == (4, 12, 3)
        assert friction_props.dtype == wp.float32


class TestMockArticulationViewWarpMassGetters:
    """Tests for MockArticulationViewWarp mass property getter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockArticulationViewWarp(count=4, num_dofs=12, num_links=13, device="cpu")

    def test_get_masses_shape(self, view):
        """Test masses shape."""
        masses = view.get_masses()
        assert masses.shape == (4, 13)
        assert masses.dtype == wp.float32

    def test_get_coms_shape(self, view):
        """Test centers of mass shape - should be (N, L) with wp.transformf dtype."""
        coms = view.get_coms()
        assert coms.shape == (4, 13)
        assert coms.dtype == wp.transformf

    def test_get_inertias_shape(self, view):
        """Test inertias shape."""
        inertias = view.get_inertias()
        assert inertias.shape == (4, 13, 9)
        assert inertias.dtype == wp.float32


class TestMockArticulationViewWarpSetters:
    """Tests for MockArticulationViewWarp setter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockArticulationViewWarp(count=4, num_dofs=12, num_links=13, device="cpu")

    def test_set_root_transforms(self, view):
        """Test setting root transforms."""
        new_data = np.random.randn(4, 7).astype(np.float32)
        new_transforms = wp.array(new_data, dtype=wp.float32, device="cpu")
        view.set_root_transforms(new_transforms)
        result = view.get_root_transforms()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, new_data, rtol=1e-5)

    def test_set_dof_positions(self, view):
        """Test setting DOF positions."""
        new_data = np.random.randn(4, 12).astype(np.float32)
        new_positions = wp.array(new_data, dtype=wp.float32, device="cpu")
        view.set_dof_positions(new_positions)
        result = view.get_dof_positions()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, new_data, rtol=1e-5)

    def test_set_dof_positions_with_indices(self, view):
        """Test setting DOF positions with indices."""
        new_data = np.random.randn(2, 12).astype(np.float32)
        new_positions = wp.array(new_data, dtype=wp.float32, device="cpu")
        indices = wp.array([0, 2], dtype=wp.int32, device="cpu")
        view.set_dof_positions(new_positions, indices=indices)
        result = view.get_dof_positions()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np[0], new_data[0], rtol=1e-5)
        np.testing.assert_allclose(result_np[2], new_data[1], rtol=1e-5)

    def test_set_dof_velocities(self, view):
        """Test setting DOF velocities."""
        new_data = np.random.randn(4, 12).astype(np.float32)
        new_velocities = wp.array(new_data, dtype=wp.float32, device="cpu")
        view.set_dof_velocities(new_velocities)
        result = view.get_dof_velocities()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, new_data, rtol=1e-5)

    def test_set_dof_limits(self, view):
        """Test setting DOF limits."""
        new_data = np.random.randn(4, 12, 2).astype(np.float32)
        new_limits = wp.array(new_data, dtype=wp.float32, device="cpu")
        view.set_dof_limits(new_limits)
        result = view.get_dof_limits()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, new_data, rtol=1e-5)

    def test_set_dof_stiffnesses(self, view):
        """Test setting DOF stiffnesses."""
        new_data = np.abs(np.random.randn(4, 12)).astype(np.float32)
        new_stiffnesses = wp.array(new_data, dtype=wp.float32, device="cpu")
        view.set_dof_stiffnesses(new_stiffnesses)
        result = view.get_dof_stiffnesses()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, new_data, rtol=1e-5)


class TestMockArticulationViewWarpNoopSetters:
    """Tests for MockArticulationViewWarp no-op setter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs."""
        return MockArticulationViewWarp(count=4, num_dofs=12, device="cpu")

    def test_set_dof_position_targets_noop(self, view):
        """Test that set_dof_position_targets is a no-op."""
        targets = wp.zeros((4, 12), dtype=wp.float32, device="cpu")
        view.set_dof_position_targets(targets)

    def test_set_dof_velocity_targets_noop(self, view):
        """Test that set_dof_velocity_targets is a no-op."""
        targets = wp.zeros((4, 12), dtype=wp.float32, device="cpu")
        view.set_dof_velocity_targets(targets)

    def test_set_dof_actuation_forces_noop(self, view):
        """Test that set_dof_actuation_forces is a no-op."""
        forces = wp.zeros((4, 12), dtype=wp.float32, device="cpu")
        view.set_dof_actuation_forces(forces)


class TestMockArticulationViewWarpMockSetters:
    """Tests for MockArticulationViewWarp mock setter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockArticulationViewWarp(count=4, num_dofs=12, num_links=13, device="cpu")

    def test_set_mock_root_transforms(self, view):
        """Test mock root transform setter."""
        mock_data = np.random.randn(4, 7).astype(np.float32)
        mock_transforms = wp.array(mock_data, dtype=wp.float32, device="cpu")
        view.set_mock_root_transforms(mock_transforms)
        result = view.get_root_transforms()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, mock_data, rtol=1e-5)

    def test_set_mock_link_transforms(self, view):
        """Test mock link transform setter."""
        mock_data = np.random.randn(4, 13, 7).astype(np.float32)
        mock_transforms = wp.array(mock_data, dtype=wp.float32, device="cpu")
        view.set_mock_link_transforms(mock_transforms)
        result = view.get_link_transforms()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, mock_data, rtol=1e-5)

    def test_set_mock_dof_positions(self, view):
        """Test mock DOF position setter."""
        mock_data = np.random.randn(4, 12).astype(np.float32)
        mock_positions = wp.array(mock_data, dtype=wp.float32, device="cpu")
        view.set_mock_dof_positions(mock_positions)
        result = view.get_dof_positions()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, mock_data, rtol=1e-5)

    def test_set_mock_dof_velocities(self, view):
        """Test mock DOF velocity setter."""
        mock_data = np.random.randn(4, 12).astype(np.float32)
        mock_velocities = wp.array(mock_data, dtype=wp.float32, device="cpu")
        view.set_mock_dof_velocities(mock_velocities)
        result = view.get_dof_velocities()
        result_np = result.numpy()
        np.testing.assert_allclose(result_np, mock_data, rtol=1e-5)


class TestMockArticulationViewWarpFactories:
    """Tests for articulation view factory functions with warp backend."""

    def test_create_mock_articulation_view_basic(self):
        """Test basic factory usage with warp backend."""
        view = create_mock_articulation_view(count=4, num_dofs=12, num_links=13, backend="warp")
        assert view.count == 4
        assert view.shared_metatype.dof_count == 12
        assert view.shared_metatype.link_count == 13
        assert isinstance(view, MockArticulationViewWarp)

    def test_create_mock_quadruped_view(self):
        """Test quadruped factory with warp backend."""
        view = create_mock_quadruped_view(count=4, backend="warp")
        assert view.count == 4
        assert view.shared_metatype.dof_count == 12
        assert view.shared_metatype.link_count == 13
        assert view.shared_metatype.fixed_base is False
        assert "FL_hip_joint" in view.shared_metatype.dof_names
        assert "base" in view.shared_metatype.link_names
        assert isinstance(view, MockArticulationViewWarp)

    def test_create_mock_humanoid_view(self):
        """Test humanoid factory with warp backend."""
        view = create_mock_humanoid_view(count=2, backend="warp")
        assert view.count == 2
        assert view.shared_metatype.dof_count == 21
        assert view.shared_metatype.link_count == 22
        assert view.shared_metatype.fixed_base is False
        assert "left_elbow" in view.shared_metatype.dof_names
        assert "pelvis" in view.shared_metatype.link_names
        assert isinstance(view, MockArticulationViewWarp)
