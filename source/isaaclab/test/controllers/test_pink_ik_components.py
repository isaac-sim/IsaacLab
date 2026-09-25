# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test cases for PinkKinematicsConfiguration class."""

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest
from pink.exceptions import FrameNotFound

from isaaclab.controllers.pink_ik.pink_kinematics_configuration import PinkKinematicsConfiguration

pytestmark = pytest.mark.integration


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

    @pytest.mark.parametrize("controlled_joint_names", [[], ["nonexistent_joint"]])
    def test_no_controlled_joints(self, urdf_path, mesh_path, controlled_joint_names):
        """Empty or unknown controlled joint names lock every joint."""
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

    def test_error_handling_invalid_urdf(self, mesh_path, controlled_joint_names):
        """Test error handling with invalid URDF path."""
        with pytest.raises(Exception):  # Should raise some exception for invalid URDF
            PinkKinematicsConfiguration(
                urdf_path="nonexistent.urdf",
                mesh_path=mesh_path,
                controlled_joint_names=controlled_joint_names,
            )

    @pytest.mark.parametrize("controlled, locked", [("joint_1", "joint_2"), ("joint_2", "joint_1")])
    def test_undercontrolled_kinematics_model(self, urdf_path, mesh_path, controlled, locked):
        """Test that the fixed joint to world is properly handled."""

        test_model = PinkKinematicsConfiguration(
            urdf_path=str(urdf_path),
            mesh_path=mesh_path,
            controlled_joint_names=[controlled],
            copy_data=True,
            forward_kinematics=True,
        )
        # Check that the controlled model only includes the revolute joints
        assert controlled in test_model.controlled_joint_names_pinocchio_order
        assert locked not in test_model.controlled_joint_names_pinocchio_order
        assert len(test_model.controlled_joint_names_pinocchio_order) == 1  # Only the two revolute joints

        # Check that the full configuration has more elements than controlled
        assert len(test_model.full_q) > len(test_model.controlled_q)
        assert len(test_model.full_q) == len(test_model.all_joint_names_pinocchio_order)
        assert len(test_model.controlled_q) == len(test_model.controlled_joint_names_pinocchio_order)

        # A full configuration update forwards the controlled joint's own value to the reduced model
        all_names = test_model.all_joint_names_pinocchio_order
        full_q = 0.1 * np.arange(1, len(all_names) + 1)
        test_model.update(full_q)
        np.testing.assert_allclose(test_model.q, full_q[[all_names.index(controlled)]])


# The robot config only supplies ``disable_gravity``, which each row overrides, so one robot covers the branches.
@pytest.mark.parametrize(
    "fixed_base, disable_gravity, robot_name",
    [(False, False, "GR1T2_HIGH_PD_CFG"), (True, False, "GR1T2_HIGH_PD_CFG"), (False, True, "GR1T2_HIGH_PD_CFG")],
)
def test_action_gravity_compensation_with_migrated_robot_configs(fixed_base, disable_gravity, robot_name):
    """The Pink robot configs retain direct gravity access and the action's effort targets."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    import torch

    from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction

    import isaaclab_assets

    robot_cfg = getattr(isaaclab_assets, robot_name).copy()
    robot_cfg.spawn.rigid_props.disable_gravity = disable_gravity
    num_base_dofs = 0 if fixed_base else 6
    forces = torch.arange(2 * (3 + num_base_dofs), dtype=torch.float32).reshape(2, -1)
    asset = SimpleNamespace(
        cfg=robot_cfg,
        data=SimpleNamespace(gravity_compensation_forces=SimpleNamespace(torch=forces)),
        num_base_dofs=num_base_dofs,
        is_fixed_base=fixed_base,
        set_joint_effort_target_index=Mock(),
    )
    action = SimpleNamespace(
        _asset=asset, _controlled_joint_ids=[0, 2], _controlled_joint_ids_tensor=torch.tensor([0, 2])
    )
    PinkInverseKinematicsAction._apply_gravity_compensation(action)

    if disable_gravity:
        asset.set_joint_effort_target_index.assert_not_called()
    else:
        asset.set_joint_effort_target_index.assert_called_once()
        kwargs = asset.set_joint_effort_target_index.call_args.kwargs
        assert kwargs["joint_ids"] == [0, 2]
        expected = torch.zeros(2, 2) if fixed_base else forces[:, [6, 8]]
        torch.testing.assert_close(kwargs["target"], expected)
