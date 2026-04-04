# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MockArticulationView."""

import pytest
import torch
from isaaclab_physx.test.mock_interfaces.factories import (
    create_mock_articulation_view,
    create_mock_humanoid_view,
    create_mock_quadruped_view,
)
from isaaclab_physx.test.mock_interfaces.views import MockArticulationView


class TestMockArticulationViewInit:
    """Tests for MockArticulationView initialization."""

    def test_default_init(self):
        """Test default initialization."""
        view = MockArticulationView()
        assert view.count == 1
        assert view.shared_metatype.dof_count == 1
        assert view.shared_metatype.link_count == 2

    def test_custom_configuration(self):
        """Test initialization with custom configuration."""
        view = MockArticulationView(
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
        view = MockArticulationView(
            num_dofs=2,
            num_links=3,
            dof_names=dof_names,
            link_names=link_names,
        )
        assert view.shared_metatype.dof_names == dof_names
        assert view.shared_metatype.link_names == link_names

    def test_tendon_properties(self):
        """Test tendon properties are zero."""
        view = MockArticulationView()
        assert view.max_fixed_tendons == 0
        assert view.max_spatial_tendons == 0


class TestMockArticulationViewRootGetters:
    """Tests for MockArticulationView root getter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockArticulationView(count=4, num_dofs=12, num_links=13, device="cpu")

    def test_get_root_transforms_shape(self, view):
        """Test root transforms shape."""
        transforms = view.get_root_transforms()
        assert transforms.shape == (4, 7)

    def test_get_root_transforms_default_quaternion(self, view):
        """Test that default quaternion is identity (xyzw format)."""
        transforms = view.get_root_transforms()
        assert torch.allclose(transforms[:, 6], torch.ones(4))  # w = 1

    def test_get_root_velocities_shape(self, view):
        """Test root velocities shape."""
        velocities = view.get_root_velocities()
        assert velocities.shape == (4, 6)


class TestMockArticulationViewLinkGetters:
    """Tests for MockArticulationView link getter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockArticulationView(count=4, num_dofs=12, num_links=13, device="cpu")

    def test_get_link_transforms_shape(self, view):
        """Test link transforms shape."""
        transforms = view.get_link_transforms()
        assert transforms.shape == (4, 13, 7)

    def test_get_link_velocities_shape(self, view):
        """Test link velocities shape."""
        velocities = view.get_link_velocities()
        assert velocities.shape == (4, 13, 6)


class TestMockArticulationViewDOFGetters:
    """Tests for MockArticulationView DOF getter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockArticulationView(count=4, num_dofs=12, num_links=13, device="cpu")

    def test_get_dof_positions_shape(self, view):
        """Test DOF positions shape."""
        positions = view.get_dof_positions()
        assert positions.shape == (4, 12)

    def test_get_dof_velocities_shape(self, view):
        """Test DOF velocities shape."""
        velocities = view.get_dof_velocities()
        assert velocities.shape == (4, 12)

    def test_get_dof_projected_joint_forces_shape(self, view):
        """Test projected joint forces shape."""
        forces = view.get_dof_projected_joint_forces()
        assert forces.shape == (4, 12)

    def test_get_dof_limits_shape(self, view):
        """Test DOF limits shape."""
        limits = view.get_dof_limits()
        assert limits.shape == (4, 12, 2)

    def test_get_dof_limits_default_values(self, view):
        """Test that default limits are infinite."""
        limits = view.get_dof_limits()
        assert torch.all(limits[:, :, 0] == float("-inf"))  # lower
        assert torch.all(limits[:, :, 1] == float("inf"))  # upper

    def test_get_dof_stiffnesses_shape(self, view):
        """Test DOF stiffnesses shape."""
        stiffnesses = view.get_dof_stiffnesses()
        assert stiffnesses.shape == (4, 12)

    def test_get_dof_dampings_shape(self, view):
        """Test DOF dampings shape."""
        dampings = view.get_dof_dampings()
        assert dampings.shape == (4, 12)

    def test_get_dof_max_forces_shape(self, view):
        """Test DOF max forces shape."""
        max_forces = view.get_dof_max_forces()
        assert max_forces.shape == (4, 12)

    def test_get_dof_max_velocities_shape(self, view):
        """Test DOF max velocities shape."""
        max_velocities = view.get_dof_max_velocities()
        assert max_velocities.shape == (4, 12)

    def test_get_dof_armatures_shape(self, view):
        """Test DOF armatures shape."""
        armatures = view.get_dof_armatures()
        assert armatures.shape == (4, 12)

    def test_get_dof_friction_coefficients_shape(self, view):
        """Test DOF friction coefficients shape."""
        friction = view.get_dof_friction_coefficients()
        assert friction.shape == (4, 12)


class TestMockArticulationViewMassGetters:
    """Tests for MockArticulationView mass property getter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockArticulationView(count=4, num_dofs=12, num_links=13, device="cpu")

    def test_get_masses_shape(self, view):
        """Test masses shape."""
        masses = view.get_masses()
        assert masses.shape == (4, 13)

    def test_get_coms_shape(self, view):
        """Test centers of mass shape."""
        coms = view.get_coms()
        assert coms.shape == (4, 13, 7)

    def test_get_inertias_shape(self, view):
        """Test inertias shape."""
        inertias = view.get_inertias()
        assert inertias.shape == (4, 13, 9)


class TestMockArticulationViewSetters:
    """Tests for MockArticulationView setter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockArticulationView(count=4, num_dofs=12, num_links=13, device="cpu")

    def test_set_root_transforms(self, view):
        """Test setting root transforms."""
        new_data = torch.randn(4, 7)
        view.set_root_transforms(new_data)
        result = view.get_root_transforms()
        assert torch.allclose(result, new_data)

    def test_set_dof_positions(self, view):
        """Test setting DOF positions."""
        new_data = torch.randn(4, 12)
        view.set_dof_positions(new_data)
        result = view.get_dof_positions()
        assert torch.allclose(result, new_data)

    def test_set_dof_positions_with_indices(self, view):
        """Test setting DOF positions with indices."""
        new_data = torch.randn(2, 12)
        indices = torch.tensor([0, 2])
        view.set_dof_positions(new_data, indices=indices)
        result = view.get_dof_positions()
        assert torch.allclose(result[0], new_data[0])
        assert torch.allclose(result[2], new_data[1])

    def test_set_dof_velocities(self, view):
        """Test setting DOF velocities."""
        new_data = torch.randn(4, 12)
        view.set_dof_velocities(new_data)
        result = view.get_dof_velocities()
        assert torch.allclose(result, new_data)

    def test_set_dof_limits(self, view):
        """Test setting DOF limits."""
        new_data = torch.randn(4, 12, 2)
        view.set_dof_limits(new_data)
        result = view.get_dof_limits()
        assert torch.allclose(result, new_data)

    def test_set_dof_stiffnesses(self, view):
        """Test setting DOF stiffnesses."""
        new_data = torch.randn(4, 12).abs()
        view.set_dof_stiffnesses(new_data)
        result = view.get_dof_stiffnesses()
        assert torch.allclose(result, new_data)


class TestMockArticulationViewNoopSetters:
    """Tests for MockArticulationView no-op setter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs."""
        return MockArticulationView(count=4, num_dofs=12, device="cpu")

    def test_set_dof_position_targets_noop(self, view):
        """Test that set_dof_position_targets is a no-op."""
        view.set_dof_position_targets(torch.randn(4, 12))

    def test_set_dof_velocity_targets_noop(self, view):
        """Test that set_dof_velocity_targets is a no-op."""
        view.set_dof_velocity_targets(torch.randn(4, 12))

    def test_set_dof_actuation_forces_noop(self, view):
        """Test that set_dof_actuation_forces is a no-op."""
        view.set_dof_actuation_forces(torch.randn(4, 12))


class TestMockArticulationViewMockSetters:
    """Tests for MockArticulationView mock setter methods."""

    @pytest.fixture
    def view(self):
        """Create a view with 4 instances, 12 DOFs, 13 links."""
        return MockArticulationView(count=4, num_dofs=12, num_links=13, device="cpu")

    def test_set_mock_root_transforms(self, view):
        """Test mock root transform setter."""
        mock_data = torch.randn(4, 7)
        view.set_mock_root_transforms(mock_data)
        result = view.get_root_transforms()
        assert torch.allclose(result, mock_data)

    def test_set_mock_link_transforms(self, view):
        """Test mock link transform setter."""
        mock_data = torch.randn(4, 13, 7)
        view.set_mock_link_transforms(mock_data)
        result = view.get_link_transforms()
        assert torch.allclose(result, mock_data)

    def test_set_mock_dof_positions(self, view):
        """Test mock DOF position setter."""
        mock_data = torch.randn(4, 12)
        view.set_mock_dof_positions(mock_data)
        result = view.get_dof_positions()
        assert torch.allclose(result, mock_data)

    def test_set_mock_dof_velocities(self, view):
        """Test mock DOF velocity setter."""
        mock_data = torch.randn(4, 12)
        view.set_mock_dof_velocities(mock_data)
        result = view.get_dof_velocities()
        assert torch.allclose(result, mock_data)


class TestMockArticulationViewFactories:
    """Tests for articulation view factory functions."""

    def test_create_mock_articulation_view_basic(self):
        """Test basic factory usage."""
        view = create_mock_articulation_view(count=4, num_dofs=12, num_links=13)
        assert view.count == 4
        assert view.shared_metatype.dof_count == 12
        assert view.shared_metatype.link_count == 13

    def test_create_mock_quadruped_view(self):
        """Test quadruped factory."""
        view = create_mock_quadruped_view(count=4)
        assert view.count == 4
        assert view.shared_metatype.dof_count == 12
        assert view.shared_metatype.link_count == 13
        assert view.shared_metatype.fixed_base is False
        assert "FL_hip_joint" in view.shared_metatype.dof_names
        assert "base" in view.shared_metatype.link_names

    def test_create_mock_humanoid_view(self):
        """Test humanoid factory."""
        view = create_mock_humanoid_view(count=2)
        assert view.count == 2
        assert view.shared_metatype.dof_count == 21
        assert view.shared_metatype.link_count == 22
        assert view.shared_metatype.fixed_base is False
        assert "left_elbow" in view.shared_metatype.dof_names
        assert "pelvis" in view.shared_metatype.link_names
