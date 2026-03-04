# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test cases for PinkKinematicsConfiguration class."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest
from pink.exceptions import FrameNotFound

from isaaclab.controllers.pink_ik.pink_kinematics_configuration import PinkKinematicsConfiguration


class TestPinkKinematicsConfiguration:
    """Test suite for PinkKinematicsConfiguration class."""

    @pytest.fixture
    def urdf_path(self):
        """Path to test URDF file."""
        return Path(__file__).parent / "urdfs/test_urdf_two_link_robot.urdf"

    @pytest.fixture
    def mesh_path(self):
        """Path to mesh directory (empty for simple test)."""
        return ""

    @pytest.fixture
    def controlled_joint_names(self):
        """List of controlled joint names for testing."""
        return ["joint_1", "joint_2"]

    @pytest.fixture
    def pink_config(self, urdf_path, mesh_path, controlled_joint_names):
        """Create a PinkKinematicsConfiguration instance for testing."""
        return PinkKinematicsConfiguration(
            urdf_path=str(urdf_path),
            mesh_path=mesh_path,
            controlled_joint_names=controlled_joint_names,
            copy_data=True,
            forward_kinematics=True,
        )

    def test_initialization(self, pink_config, controlled_joint_names):
        """Test proper initialization of PinkKinematicsConfiguration."""
        # Check that controlled joint names are stored correctly
        assert pink_config._controlled_joint_names == controlled_joint_names

        # Check that both full and controlled models are created
        assert pink_config.full_model is not None
        assert pink_config.controlled_model is not None
        assert pink_config.full_data is not None
        assert pink_config.controlled_data is not None

        # Check that configuration vectors are initialized
        assert pink_config.full_q is not None
        assert pink_config.controlled_q is not None

        # Check that the controlled model has the same number or fewer joints than the full model
        assert pink_config.controlled_model.nq == pink_config.full_model.nq

    def test_joint_names_properties(self, pink_config):
        """Test joint name properties."""
        # Test controlled joint names in pinocchio order
        controlled_names = pink_config.controlled_joint_names_pinocchio_order
        assert isinstance(controlled_names, list)
        assert len(controlled_names) == len(pink_config._controlled_joint_names)
        assert "joint_1" in controlled_names
        assert "joint_2" in controlled_names

        # Test all joint names in pinocchio order
        all_names = pink_config.all_joint_names_pinocchio_order
        assert isinstance(all_names, list)
        assert len(all_names) == len(controlled_names)
        assert "joint_1" in all_names
        assert "joint_2" in all_names

    def test_update_with_valid_configuration(self, pink_config):
        """Test updating configuration with valid joint values."""
        # Get initial configuration
        initial_q = pink_config.full_q.copy()

        # Create a new configuration with different joint values
        new_q = initial_q.copy()
        new_q[1] = 0.5  # Change first revolute joint value (index 1, since 0 is fixed joint)

        # Update configuration
        pink_config.update(new_q)

        # Check that the configuration was updated
        print(pink_config.full_q)
        assert not np.allclose(pink_config.full_q, initial_q)
        assert np.allclose(pink_config.full_q, new_q)

    def test_update_with_none(self, pink_config):
        """Test updating configuration with None (should use current configuration)."""
        # Get initial configuration
        initial_q = pink_config.full_q.copy()

        # Update with None
        pink_config.update(None)

        # Configuration should remain the same
        assert np.allclose(pink_config.full_q, initial_q)

    def test_update_with_wrong_dimensions(self, pink_config):
        """Test that update raises ValueError with wrong configuration dimensions."""
        # Create configuration with wrong number of joints
        wrong_q = np.array([0.1, 0.2, 0.3])  # Wrong number of joints

        with pytest.raises(ValueError, match="q must have the same length as the number of joints"):
            pink_config.update(wrong_q)

    def test_get_frame_jacobian_existing_frame(self, pink_config):
        """Test getting Jacobian for an existing frame."""
        # Get Jacobian for link_1 frame
        jacobian = pink_config.get_frame_jacobian("link_1")

        # Check that Jacobian has correct shape
        # Should be 6 rows (linear + angular velocity) and columns equal to controlled joints
        expected_rows = 6
        expected_cols = len(pink_config._controlled_joint_names)
        assert jacobian.shape == (expected_rows, expected_cols)

        # Check that Jacobian is not all zeros (should have some non-zero values)
        assert not np.allclose(jacobian, 0.0)

    def test_get_frame_jacobian_nonexistent_frame(self, pink_config):
        """Test that get_frame_jacobian raises FrameNotFound for non-existent frame."""
        with pytest.raises(FrameNotFound):
            pink_config.get_frame_jacobian("nonexistent_frame")

    def test_get_transform_frame_to_world_existing_frame(self, pink_config):
        """Test getting transform for an existing frame."""
        # Get transform for link_1 frame
        transform = pink_config.get_transform_frame_to_world("link_1")

        # Check that transform is a pinocchio SE3 object
        assert isinstance(transform, pin.SE3)

        # Check that transform has reasonable values (not identity for non-zero joint angles)
        assert not np.allclose(transform.homogeneous, np.eye(4))

    def test_get_transform_frame_to_world_nonexistent_frame(self, pink_config):
        """Test that get_transform_frame_to_world raises FrameNotFound for non-existent frame."""
        with pytest.raises(FrameNotFound):
            pink_config.get_transform_frame_to_world("nonexistent_frame")

    def test_multiple_controlled_joints(self, urdf_path, mesh_path):
        """Test configuration with multiple controlled joints."""
        # Create configuration with all available joints as controlled
        controlled_joint_names = ["joint_1", "joint_2"]  # Both revolute joints

        pink_config = PinkKinematicsConfiguration(
            urdf_path=str(urdf_path),
            mesh_path=mesh_path,
            controlled_joint_names=controlled_joint_names,
        )

        # Check that controlled model has correct number of joints
        assert pink_config.controlled_model.nq == len(controlled_joint_names)

    def test_no_controlled_joints(self, urdf_path, mesh_path):
        """Test configuration with no controlled joints."""
        controlled_joint_names = []

        pink_config = PinkKinematicsConfiguration(
            urdf_path=str(urdf_path),
            mesh_path=mesh_path,
            controlled_joint_names=controlled_joint_names,
        )

        # Check that controlled model has 0 joints
        assert pink_config.controlled_model.nq == 0
        assert len(pink_config.controlled_q) == 0

    def test_jacobian_consistency(self, pink_config):
        """Test that Jacobian computation is consistent across updates."""
        # Get Jacobian at initial configuration
        jacobian_1 = pink_config.get_frame_jacobian("link_2")

        # Update configuration
        new_q = pink_config.full_q.copy()
        new_q[1] = 0.3  # Change first revolute joint (index 1, since 0 is fixed joint)
        pink_config.update(new_q)

        # Get Jacobian at new configuration
        jacobian_2 = pink_config.get_frame_jacobian("link_2")

        # Jacobians should be different (not all close)
        assert not np.allclose(jacobian_1, jacobian_2)

    def test_transform_consistency(self, pink_config):
        """Test that transform computation is consistent across updates."""
        # Get transform at initial configuration
        transform_1 = pink_config.get_transform_frame_to_world("link_2")

        # Update configuration
        new_q = pink_config.full_q.copy()
        new_q[1] = 0.5  # Change first revolute joint (index 1, since 0 is fixed joint)
        pink_config.update(new_q)

        # Get transform at new configuration
        transform_2 = pink_config.get_transform_frame_to_world("link_2")

        # Transforms should be different
        assert not np.allclose(transform_1.homogeneous, transform_2.homogeneous)

    def test_inheritance_from_configuration(self, pink_config):
        """Test that PinkKinematicsConfiguration properly inherits from Pink Configuration."""
        from pink.configuration import Configuration

        # Check inheritance
        assert isinstance(pink_config, Configuration)

        # Check that we can call parent class methods
        assert hasattr(pink_config, "update")
        assert hasattr(pink_config, "get_transform_frame_to_world")

    def test_controlled_joint_indices_calculation(self, pink_config):
        """Test that controlled joint indices are calculated correctly."""
        # Check that controlled joint indices are valid
        assert len(pink_config._controlled_joint_indices) == len(pink_config._controlled_joint_names)

        # Check that all indices are within bounds
        for idx in pink_config._controlled_joint_indices:
            assert 0 <= idx < len(pink_config._all_joint_names)

        # Check that indices correspond to controlled joint names
        for i, idx in enumerate(pink_config._controlled_joint_indices):
            joint_name = pink_config._all_joint_names[idx]
            assert joint_name in pink_config._controlled_joint_names

    def test_full_model_integrity(self, pink_config):
        """Test that the full model maintains integrity."""
        # Check that full model has all joints
        assert pink_config.full_model.nq > 0
        assert len(pink_config.full_model.names) > 1  # More than just "universe"

    def test_controlled_model_integrity(self, pink_config):
        """Test that the controlled model maintains integrity."""
        # Check that controlled model has correct number of joints
        assert pink_config.controlled_model.nq == len(pink_config._controlled_joint_names)

    def test_configuration_vector_consistency(self, pink_config):
        """Test that configuration vectors are consistent between full and controlled models."""
        # Check that controlled_q is a subset of full_q
        controlled_indices = pink_config._controlled_joint_indices
        for i, idx in enumerate(controlled_indices):
            assert np.isclose(pink_config.controlled_q[i], pink_config.full_q[idx])

    def test_error_handling_invalid_urdf(self, mesh_path, controlled_joint_names):
        """Test error handling with invalid URDF path."""
        with pytest.raises(Exception):  # Should raise some exception for invalid URDF
            PinkKinematicsConfiguration(
                urdf_path="nonexistent.urdf",
                mesh_path=mesh_path,
                controlled_joint_names=controlled_joint_names,
            )

    def test_error_handling_invalid_joint_names(self, urdf_path, mesh_path):
        """Test error handling with invalid joint names."""
        invalid_joint_names = ["nonexistent_joint"]

        # This should not raise an error, but the controlled model should have 0 joints
        pink_config = PinkKinematicsConfiguration(
            urdf_path=str(urdf_path),
            mesh_path=mesh_path,
            controlled_joint_names=invalid_joint_names,
        )

        assert pink_config.controlled_model.nq == 0
        assert len(pink_config.controlled_q) == 0

    def test_undercontrolled_kinematics_model(self, urdf_path, mesh_path):
        """Test that the fixed joint to world is properly handled."""

        test_model = PinkKinematicsConfiguration(
            urdf_path=str(urdf_path),
            mesh_path=mesh_path,
            controlled_joint_names=["joint_1"],
            copy_data=True,
            forward_kinematics=True,
        )
        # Check that the controlled model only includes the revolute joints
        assert "joint_1" in test_model.controlled_joint_names_pinocchio_order
        assert "joint_2" not in test_model.controlled_joint_names_pinocchio_order
        assert len(test_model.controlled_joint_names_pinocchio_order) == 1  # Only the two revolute joints

        # Check that the full configuration has more elements than controlled
        assert len(test_model.full_q) > len(test_model.controlled_q)
        assert len(test_model.full_q) == len(test_model.all_joint_names_pinocchio_order)
        assert len(test_model.controlled_q) == len(test_model.controlled_joint_names_pinocchio_order)
