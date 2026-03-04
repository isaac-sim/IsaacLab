# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MockNewtonArticulationView."""

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.test.mock_interfaces.views import MockNewtonArticulationView


class TestMockNewtonArticulationViewInit:
    """Tests for MockNewtonArticulationView initialization."""

    def test_default_init(self):
        """Test default initialization."""
        view = MockNewtonArticulationView()
        assert view.count == 1
        assert view.joint_dof_count == 1
        assert view.link_count == 2
        assert view.is_fixed_base is False

    def test_custom_configuration(self):
        """Test initialization with custom configuration."""
        view = MockNewtonArticulationView(
            num_instances=4,
            num_joints=12,
            num_bodies=13,
            is_fixed_base=True,
        )
        assert view.count == 4
        assert view.joint_dof_count == 12
        assert view.link_count == 13
        assert view.is_fixed_base is True

    def test_custom_names(self):
        """Test initialization with custom joint and body names."""
        joint_names = ["joint_0", "joint_1"]
        body_names = ["base", "link_1", "link_2"]
        view = MockNewtonArticulationView(
            num_joints=2,
            num_bodies=3,
            joint_names=joint_names,
            body_names=body_names,
        )
        assert view.joint_dof_names == joint_names
        assert view.body_names == body_names

    def test_fixed_base_config(self):
        """Test fixed-base configuration."""
        view = MockNewtonArticulationView(
            num_instances=2,
            num_joints=6,
            num_bodies=7,
            is_fixed_base=True,
        )
        assert view.is_fixed_base is True
        # Root velocities should be None for fixed base
        assert view.get_root_velocities(None) is None
        assert view.get_link_velocities(None) is None


class TestMockNewtonArticulationViewRootGetters:
    """Tests for MockNewtonArticulationView root getter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockNewtonArticulationView(num_instances=4, num_joints=12, num_bodies=13, device="cpu")

    def test_get_root_transforms_shape(self, view):
        """Test root transforms shape - should be (N, 1) with wp.transformf dtype."""
        transforms = view.get_root_transforms(None)
        assert transforms.shape == (4, 1)
        assert transforms.dtype == wp.transformf

    def test_get_root_transforms_default_identity(self, view):
        """Test that default root transforms are zero-initialized."""
        transforms = view.get_root_transforms(None)
        transforms_np = transforms.numpy()
        # Default is zero, meaning all zeros for position and quaternion
        np.testing.assert_allclose(transforms_np, 0.0)

    def test_get_root_velocities_shape(self, view):
        """Test root velocities shape - should be (N, 1) with wp.spatial_vectorf dtype."""
        velocities = view.get_root_velocities(None)
        assert velocities.shape == (4, 1)
        assert velocities.dtype == wp.spatial_vectorf

    def test_fixed_base_root_velocities_none(self):
        """Test that fixed-base root velocities are None."""
        view = MockNewtonArticulationView(
            num_instances=4, num_joints=12, num_bodies=13, is_fixed_base=True, device="cpu"
        )
        assert view.get_root_velocities(None) is None

    def test_fixed_base_root_transforms_shape(self):
        """Test that fixed-base root transforms have shape (N, 1, 1)."""
        view = MockNewtonArticulationView(
            num_instances=4, num_joints=12, num_bodies=13, is_fixed_base=True, device="cpu"
        )
        transforms = view.get_root_transforms(None)
        assert transforms.shape == (4, 1, 1)
        assert transforms.dtype == wp.transformf


class TestMockNewtonArticulationViewLinkGetters:
    """Tests for MockNewtonArticulationView link getter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockNewtonArticulationView(num_instances=4, num_joints=12, num_bodies=13, device="cpu")

    def test_get_link_transforms_shape(self, view):
        """Test link transforms shape - should be (N, 1, L) with wp.transformf dtype."""
        transforms = view.get_link_transforms(None)
        assert transforms.shape == (4, 1, 13)
        assert transforms.dtype == wp.transformf

    def test_get_link_velocities_shape(self, view):
        """Test link velocities shape - should be (N, 1, L) with wp.spatial_vectorf dtype."""
        velocities = view.get_link_velocities(None)
        assert velocities.shape == (4, 1, 13)
        assert velocities.dtype == wp.spatial_vectorf

    def test_fixed_base_link_velocities_none(self):
        """Test that fixed-base link velocities are None."""
        view = MockNewtonArticulationView(
            num_instances=4, num_joints=12, num_bodies=13, is_fixed_base=True, device="cpu"
        )
        assert view.get_link_velocities(None) is None


class TestMockNewtonArticulationViewDOFGetters:
    """Tests for MockNewtonArticulationView DOF getter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockNewtonArticulationView(num_instances=4, num_joints=12, num_bodies=13, device="cpu")

    def test_get_dof_positions_shape(self, view):
        """Test DOF positions shape - should be (N, 1, J)."""
        positions = view.get_dof_positions(None)
        assert positions.shape == (4, 1, 12)
        assert positions.dtype == wp.float32

    def test_get_dof_velocities_shape(self, view):
        """Test DOF velocities shape - should be (N, 1, J)."""
        velocities = view.get_dof_velocities(None)
        assert velocities.shape == (4, 1, 12)
        assert velocities.dtype == wp.float32

    @pytest.mark.parametrize(
        "attr_name",
        [
            "joint_limit_lower",
            "joint_limit_upper",
            "joint_target_ke",
            "joint_target_kd",
            "joint_armature",
            "joint_friction",
            "joint_velocity_limit",
            "joint_effort_limit",
        ],
    )
    def test_get_joint_attribute_shape(self, view, attr_name):
        """Test joint attribute shapes via get_attribute()."""
        attr = view.get_attribute(attr_name, None)
        assert attr.shape == (4, 1, 12)
        assert attr.dtype == wp.float32


class TestMockNewtonArticulationViewMassGetters:
    """Tests for MockNewtonArticulationView mass property getter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockNewtonArticulationView(num_instances=4, num_joints=12, num_bodies=13, device="cpu")

    def test_get_body_mass_shape(self, view):
        """Test body mass shape via get_attribute()."""
        mass = view.get_attribute("body_mass", None)
        assert mass.shape == (4, 1, 13)
        assert mass.dtype == wp.float32

    def test_get_body_com_shape(self, view):
        """Test body COM shape via get_attribute()."""
        com = view.get_attribute("body_com", None)
        assert com.shape == (4, 1, 13)
        assert com.dtype == wp.vec3f

    def test_get_body_inertia_shape(self, view):
        """Test body inertia shape via get_attribute()."""
        inertia = view.get_attribute("body_inertia", None)
        assert inertia.shape == (4, 1, 13)
        assert inertia.dtype == wp.mat33f


class TestMockNewtonArticulationViewSetters:
    """Tests for MockNewtonArticulationView setter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockNewtonArticulationView(num_instances=4, num_joints=12, num_bodies=13, device="cpu")

    def test_set_root_transforms(self, view):
        """Test setting root transforms round-trip."""
        new_transforms = wp.zeros((4, 1), dtype=wp.transformf, device="cpu")
        view.set_root_transforms(None, new_transforms)
        result = view.get_root_transforms(None)
        np.testing.assert_allclose(result.numpy(), new_transforms.numpy())

    def test_set_root_velocities(self, view):
        """Test setting root velocities round-trip."""
        new_velocities = wp.zeros((4, 1), dtype=wp.spatial_vectorf, device="cpu")
        view.set_root_velocities(None, new_velocities)
        result = view.get_root_velocities(None)
        np.testing.assert_allclose(result.numpy(), new_velocities.numpy())

    def test_noop_setters_flag(self, view):
        """Test that _noop_setters disables writes."""
        # Set initial data
        initial_transforms = wp.zeros((4, 1), dtype=wp.transformf, device="cpu")
        view.set_root_transforms(None, initial_transforms)

        # Enable noop
        view._noop_setters = True

        # Try to write different data
        new_transforms = wp.ones((4, 1), dtype=wp.transformf, device="cpu")
        view.set_root_transforms(None, new_transforms)

        # Should still have initial data
        result = view.get_root_transforms(None)
        np.testing.assert_allclose(result.numpy(), initial_transforms.numpy())


class TestMockNewtonArticulationViewMockSetters:
    """Tests for MockNewtonArticulationView mock setter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockNewtonArticulationView(num_instances=4, num_joints=12, num_bodies=13, device="cpu")

    def test_set_mock_root_transforms(self, view):
        """Test mock root transform setter round-trip."""
        mock_data = wp.zeros((4, 1), dtype=wp.transformf, device="cpu")
        view.set_mock_root_transforms(mock_data)
        result = view.get_root_transforms(None)
        np.testing.assert_allclose(result.numpy(), mock_data.numpy())

    def test_set_mock_root_velocities(self, view):
        """Test mock root velocity setter round-trip."""
        mock_data = wp.zeros((4, 1), dtype=wp.spatial_vectorf, device="cpu")
        view.set_mock_root_velocities(mock_data)
        result = view.get_root_velocities(None)
        np.testing.assert_allclose(result.numpy(), mock_data.numpy())

    def test_set_mock_link_transforms(self, view):
        """Test mock link transform setter round-trip."""
        mock_data = wp.zeros((4, 1, 13), dtype=wp.transformf, device="cpu")
        view.set_mock_link_transforms(mock_data)
        result = view.get_link_transforms(None)
        np.testing.assert_allclose(result.numpy(), mock_data.numpy())

    def test_set_mock_link_velocities(self, view):
        """Test mock link velocity setter round-trip."""
        mock_data = wp.zeros((4, 1, 13), dtype=wp.spatial_vectorf, device="cpu")
        view.set_mock_link_velocities(mock_data)
        result = view.get_link_velocities(None)
        np.testing.assert_allclose(result.numpy(), mock_data.numpy())

    def test_set_mock_dof_positions(self, view):
        """Test mock DOF position setter round-trip."""
        mock_data = wp.array(np.random.randn(4, 1, 12).astype(np.float32), dtype=wp.float32, device="cpu")
        view.set_mock_dof_positions(mock_data)
        result = view.get_dof_positions(None)
        np.testing.assert_allclose(result.numpy(), mock_data.numpy(), rtol=1e-5)

    def test_set_mock_dof_velocities(self, view):
        """Test mock DOF velocity setter round-trip."""
        mock_data = wp.array(np.random.randn(4, 1, 12).astype(np.float32), dtype=wp.float32, device="cpu")
        view.set_mock_dof_velocities(mock_data)
        result = view.get_dof_velocities(None)
        np.testing.assert_allclose(result.numpy(), mock_data.numpy(), rtol=1e-5)

    def test_set_mock_masses(self, view):
        """Test mock body mass setter round-trip."""
        mock_data = wp.array((np.random.rand(4, 1, 13) * 10).astype(np.float32), dtype=wp.float32, device="cpu")
        view.set_mock_masses(mock_data)
        result = view.get_attribute("body_mass", None)
        np.testing.assert_allclose(result.numpy(), mock_data.numpy(), rtol=1e-5)

    def test_set_mock_coms(self, view):
        """Test mock body COM setter round-trip."""
        mock_data = wp.zeros((4, 1, 13), dtype=wp.vec3f, device="cpu")
        view.set_mock_coms(mock_data)
        result = view.get_attribute("body_com", None)
        np.testing.assert_allclose(result.numpy(), mock_data.numpy())

    def test_set_mock_inertias(self, view):
        """Test mock body inertia setter round-trip."""
        mock_data = wp.zeros((4, 1, 13), dtype=wp.mat33f, device="cpu")
        view.set_mock_inertias(mock_data)
        result = view.get_attribute("body_inertia", None)
        np.testing.assert_allclose(result.numpy(), mock_data.numpy())


class TestMockNewtonArticulationViewRandomData:
    """Tests for MockNewtonArticulationView random data generation."""

    def test_set_random_mock_data_populates_arrays(self):
        """Test that set_random_mock_data populates non-None arrays."""
        view = MockNewtonArticulationView(num_instances=4, num_joints=12, num_bodies=13, device="cpu")
        view.set_random_mock_data()

        # Root state should be populated
        transforms = view.get_root_transforms(None)
        assert transforms is not None
        assert transforms.shape == (4, 1)

        velocities = view.get_root_velocities(None)
        assert velocities is not None
        assert velocities.shape == (4, 1)

        # Link state should be populated
        link_transforms = view.get_link_transforms(None)
        assert link_transforms is not None
        assert link_transforms.shape == (4, 1, 13)

        link_velocities = view.get_link_velocities(None)
        assert link_velocities is not None
        assert link_velocities.shape == (4, 1, 13)

        # DOF state should be populated
        positions = view.get_dof_positions(None)
        assert positions is not None
        assert positions.shape == (4, 1, 12)

        velocities = view.get_dof_velocities(None)
        assert velocities is not None
        assert velocities.shape == (4, 1, 12)

        # Attributes should be populated
        mass = view.get_attribute("body_mass", None)
        assert mass is not None
        assert mass.shape == (4, 1, 13)

    def test_set_random_mock_data_fixed_base(self):
        """Test random data with fixed base (no velocities)."""
        view = MockNewtonArticulationView(
            num_instances=4, num_joints=12, num_bodies=13, is_fixed_base=True, device="cpu"
        )
        view.set_random_mock_data()

        # Root transforms should have fixed-base shape
        transforms = view.get_root_transforms(None)
        assert transforms.shape == (4, 1, 1)

        # Velocities should still be None for fixed base
        assert view.get_root_velocities(None) is None
        assert view.get_link_velocities(None) is None

    def test_set_random_mock_data_has_nonzero_values(self):
        """Test that random data has non-zero values."""
        view = MockNewtonArticulationView(num_instances=4, num_joints=12, num_bodies=13, device="cpu")
        view.set_random_mock_data()

        positions = view.get_dof_positions(None)
        assert not np.allclose(positions.numpy(), 0.0)

    def test_get_attribute_unknown_raises(self):
        """Test that requesting an unknown attribute raises KeyError."""
        view = MockNewtonArticulationView()
        with pytest.raises(KeyError):
            view.get_attribute("nonexistent_attribute", None)
