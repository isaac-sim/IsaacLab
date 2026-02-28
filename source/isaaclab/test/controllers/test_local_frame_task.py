# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test cases for LocalFrameTask class."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask
from isaaclab.controllers.pink_ik.pink_kinematics_configuration import PinkKinematicsConfiguration

# class TestLocalFrameTask:
#     """Test suite for LocalFrameTask class."""


@pytest.fixture
def urdf_path():
    """Path to test URDF file."""
    return Path(__file__).parent / "urdfs" / "test_urdf_two_link_robot.urdf"


@pytest.fixture
def controlled_joint_names():
    """List of controlled joint names for testing."""
    return ["joint_1", "joint_2"]


@pytest.fixture
def pink_config(urdf_path, controlled_joint_names):
    """Create a PinkKinematicsConfiguration instance for testing."""
    return PinkKinematicsConfiguration(
        urdf_path=str(urdf_path),
        controlled_joint_names=controlled_joint_names,
        # copy_data=True,
        # forward_kinematics=True,
    )


@pytest.fixture
def local_frame_task():
    """Create a LocalFrameTask instance for testing."""
    return LocalFrameTask(
        frame="link_2",
        base_link_frame_name="base_link",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=0.0,
        gain=1.0,
    )


def test_initialization(local_frame_task):
    """Test proper initialization of LocalFrameTask."""
    # Check that the task is properly initialized
    assert local_frame_task.frame == "link_2"
    assert local_frame_task.base_link_frame_name == "base_link"
    assert np.allclose(local_frame_task.cost[:3], [1.0, 1.0, 1.0])
    assert np.allclose(local_frame_task.cost[3:], [1.0, 1.0, 1.0])
    assert local_frame_task.lm_damping == 0.0
    assert local_frame_task.gain == 1.0

    # Check that target is initially None
    assert local_frame_task.transform_target_to_base is None


def test_initialization_with_sequence_costs():
    """Test initialization with sequence costs."""
    task = LocalFrameTask(
        frame="link_1",
        base_link_frame_name="base_link",
        position_cost=[1.0, 1.0, 1.0],
        orientation_cost=[1.0, 1.0, 1.0],
        lm_damping=0.1,
        gain=2.0,
    )

    assert task.frame == "link_1"
    assert task.base_link_frame_name == "base_link"
    assert np.allclose(task.cost[:3], [1.0, 1.0, 1.0])
    assert np.allclose(task.cost[3:], [1.0, 1.0, 1.0])
    assert task.lm_damping == 0.1
    assert task.gain == 2.0


def test_inheritance_from_frame_task(local_frame_task):
    """Test that LocalFrameTask properly inherits from FrameTask."""
    from pink.tasks.frame_task import FrameTask

    # Check inheritance
    assert isinstance(local_frame_task, FrameTask)

    # Check that we can call parent class methods
    assert hasattr(local_frame_task, "compute_error")
    assert hasattr(local_frame_task, "compute_jacobian")


def test_set_target(local_frame_task):
    """Test setting target with a transform."""
    # Create a test transform
    target_transform = pin.SE3.Identity()
    target_transform.translation = np.array([0.1, 0.2, 0.3])
    target_transform.rotation = pin.exp3(np.array([0.1, 0.0, 0.0]))

    # Set the target
    local_frame_task.set_target(target_transform)

    # Check that target was set correctly
    assert local_frame_task.transform_target_to_base is not None
    assert isinstance(local_frame_task.transform_target_to_base, pin.SE3)

    # Check that it's a copy (not the same object)
    assert local_frame_task.transform_target_to_base is not target_transform

    # Check that values match
    assert np.allclose(local_frame_task.transform_target_to_base.translation, target_transform.translation)
    assert np.allclose(local_frame_task.transform_target_to_base.rotation, target_transform.rotation)


def test_set_target_from_configuration(local_frame_task, pink_config):
    """Test setting target from a robot configuration."""
    # Set target from configuration
    local_frame_task.set_target_from_configuration(pink_config)

    # Check that target was set
    assert local_frame_task.transform_target_to_base is not None
    assert isinstance(local_frame_task.transform_target_to_base, pin.SE3)


def test_set_target_from_configuration_wrong_type(local_frame_task):
    """Test that set_target_from_configuration raises error with wrong type."""
    with pytest.raises(ValueError, match="configuration must be a PinkKinematicsConfiguration"):
        local_frame_task.set_target_from_configuration("not_a_configuration")


def test_compute_error_with_target_set(local_frame_task, pink_config):
    """Test computing error when target is set."""
    # Set a target
    target_transform = pin.SE3.Identity()
    target_transform.translation = np.array([0.1, 0.2, 0.3])
    local_frame_task.set_target(target_transform)

    # Compute error
    error = local_frame_task.compute_error(pink_config)

    # Check that error is computed correctly
    assert isinstance(error, np.ndarray)
    assert error.shape == (6,)  # 6D error (3 position + 3 orientation)

    # Error should not be all zeros (unless target exactly matches current pose)
    # This is a reasonable assumption for a random target


def test_compute_error_without_target(local_frame_task, pink_config):
    """Test that compute_error raises error when no target is set."""
    with pytest.raises(ValueError, match="no target set for frame 'link_2'"):
        local_frame_task.compute_error(pink_config)


def test_compute_error_wrong_configuration_type(local_frame_task):
    """Test that compute_error raises error with wrong configuration type."""
    # Set a target first
    target_transform = pin.SE3.Identity()
    local_frame_task.set_target(target_transform)

    with pytest.raises(ValueError, match="configuration must be a PinkKinematicsConfiguration"):
        local_frame_task.compute_error("not_a_configuration")


def test_compute_jacobian_with_target_set(local_frame_task, pink_config):
    """Test computing Jacobian when target is set."""
    # Set a target
    target_transform = pin.SE3.Identity()
    target_transform.translation = np.array([0.1, 0.2, 0.3])
    local_frame_task.set_target(target_transform)

    # Compute Jacobian
    jacobian = local_frame_task.compute_jacobian(pink_config)

    # Check that Jacobian is computed correctly
    assert isinstance(jacobian, np.ndarray)
    assert jacobian.shape == (6, 2)  # 6 rows (error), 2 columns (controlled joints)

    # Jacobian should not be all zeros
    assert not np.allclose(jacobian, 0.0)


def test_compute_jacobian_without_target(local_frame_task, pink_config):
    """Test that compute_jacobian raises error when no target is set."""
    with pytest.raises(Exception, match="no target set for frame 'link_2'"):
        local_frame_task.compute_jacobian(pink_config)


def test_error_consistency_across_configurations(local_frame_task, pink_config):
    """Test that error computation is consistent across different configurations."""
    # Set a target
    target_transform = pin.SE3.Identity()
    target_transform.translation = np.array([0.1, 0.2, 0.3])
    local_frame_task.set_target(target_transform)

    # Compute error at initial configuration
    error_1 = local_frame_task.compute_error(pink_config)

    # Update configuration
    new_q = pink_config.full_q.copy()
    new_q[1] = 0.5  # Change first revolute joint
    pink_config.update(new_q)

    # Compute error at new configuration
    error_2 = local_frame_task.compute_error(pink_config)

    # Errors should be different (not all close)
    assert not np.allclose(error_1, error_2)


def test_jacobian_consistency_across_configurations(local_frame_task, pink_config):
    """Test that Jacobian computation is consistent across different configurations."""
    # Set a target
    target_transform = pin.SE3.Identity()
    target_transform.translation = np.array([0.1, 0.2, 0.3])
    local_frame_task.set_target(target_transform)

    # Compute Jacobian at initial configuration
    jacobian_1 = local_frame_task.compute_jacobian(pink_config)

    # Update configuration
    new_q = pink_config.full_q.copy()
    new_q[1] = 0.3  # Change first revolute joint
    pink_config.update(new_q)

    # Compute Jacobian at new configuration
    jacobian_2 = local_frame_task.compute_jacobian(pink_config)

    # Jacobians should be different (not all close)
    assert not np.allclose(jacobian_1, jacobian_2)


def test_error_zero_at_target_pose(local_frame_task, pink_config):
    """Test that error is zero when current pose matches target pose."""
    # Get current transform of the frame
    current_transform = pink_config.get_transform_frame_to_world("link_2")

    # Set target to current pose
    local_frame_task.set_target(current_transform)

    # Compute error
    error = local_frame_task.compute_error(pink_config)

    # Error should be very close to zero
    assert np.allclose(error, 0.0, atol=1e-10)


def test_different_frames(pink_config):
    """Test LocalFrameTask with different frame names."""
    # Test with link_1 frame
    task_link1 = LocalFrameTask(
        frame="link_1",
        base_link_frame_name="base_link",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    # Set target and compute error
    target_transform = pin.SE3.Identity()
    target_transform.translation = np.array([0.1, 0.0, 0.0])
    task_link1.set_target(target_transform)

    error_link1 = task_link1.compute_error(pink_config)
    assert error_link1.shape == (6,)

    # Test with base_link frame
    task_base = LocalFrameTask(
        frame="base_link",
        base_link_frame_name="base_link",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    task_base.set_target(target_transform)
    error_base = task_base.compute_error(pink_config)
    assert error_base.shape == (6,)


def test_different_base_frames(pink_config):
    """Test LocalFrameTask with different base frame names."""
    # Test with base_link as base frame
    task_base_base = LocalFrameTask(
        frame="link_2",
        base_link_frame_name="base_link",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    target_transform = pin.SE3.Identity()
    task_base_base.set_target(target_transform)
    error_base_base = task_base_base.compute_error(pink_config)
    assert error_base_base.shape == (6,)

    # Test with link_1 as base frame
    task_link1_base = LocalFrameTask(
        frame="link_2",
        base_link_frame_name="link_1",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    task_link1_base.set_target(target_transform)
    error_link1_base = task_link1_base.compute_error(pink_config)
    assert error_link1_base.shape == (6,)


def test_sequence_cost_parameters():
    """Test LocalFrameTask with sequence cost parameters."""
    task = LocalFrameTask(
        frame="link_2",
        base_link_frame_name="base_link",
        position_cost=[1.0, 2.0, 3.0],
        orientation_cost=[0.5, 1.0, 1.5],
        lm_damping=0.1,
        gain=2.0,
    )

    assert np.allclose(task.cost[:3], [1.0, 2.0, 3.0])  # Position costs
    assert np.allclose(task.cost[3:], [0.5, 1.0, 1.5])  # Orientation costs
    assert task.lm_damping == 0.1
    assert task.gain == 2.0


def test_error_magnitude_consistency(local_frame_task, pink_config):
    """Test that error computation produces reasonable results."""
    # Set a small target offset
    small_target = pin.SE3.Identity()
    small_target.translation = np.array([0.01, 0.01, 0.01])
    local_frame_task.set_target(small_target)

    error_small = local_frame_task.compute_error(pink_config)

    # Set a large target offset
    large_target = pin.SE3.Identity()
    large_target.translation = np.array([0.5, 0.5, 0.5])
    local_frame_task.set_target(large_target)

    error_large = local_frame_task.compute_error(pink_config)

    # Both errors should be finite and reasonable
    assert np.all(np.isfinite(error_small))
    assert np.all(np.isfinite(error_large))
    assert not np.allclose(error_small, error_large)  # Different targets should produce different errors


def test_jacobian_structure(local_frame_task, pink_config):
    """Test that Jacobian has the correct structure."""
    # Set a target
    target_transform = pin.SE3.Identity()
    target_transform.translation = np.array([0.1, 0.2, 0.3])
    local_frame_task.set_target(target_transform)

    # Compute Jacobian
    jacobian = local_frame_task.compute_jacobian(pink_config)

    # Check structure
    assert jacobian.shape == (6, 2)  # 6 error dimensions, 2 controlled joints

    # Check that Jacobian is not all zeros (basic functionality check)
    assert not np.allclose(jacobian, 0.0)


def test_multiple_target_updates(local_frame_task, pink_config):
    """Test that multiple target updates work correctly."""
    # Set first target
    target1 = pin.SE3.Identity()
    target1.translation = np.array([0.1, 0.0, 0.0])
    local_frame_task.set_target(target1)

    error1 = local_frame_task.compute_error(pink_config)

    # Set second target
    target2 = pin.SE3.Identity()
    target2.translation = np.array([0.0, 0.1, 0.0])
    local_frame_task.set_target(target2)

    error2 = local_frame_task.compute_error(pink_config)

    # Errors should be different
    assert not np.allclose(error1, error2)


def test_inheritance_behavior(local_frame_task):
    """Test that LocalFrameTask properly overrides parent class methods."""
    # Check that the class has the expected methods
    assert hasattr(local_frame_task, "set_target")
    assert hasattr(local_frame_task, "set_target_from_configuration")
    assert hasattr(local_frame_task, "compute_error")
    assert hasattr(local_frame_task, "compute_jacobian")

    # Check that these are the overridden methods, not the parent ones
    assert local_frame_task.set_target.__qualname__ == "LocalFrameTask.set_target"
    assert local_frame_task.compute_error.__qualname__ == "LocalFrameTask.compute_error"
    assert local_frame_task.compute_jacobian.__qualname__ == "LocalFrameTask.compute_jacobian"


def test_target_copying_behavior(local_frame_task):
    """Test that target transforms are properly copied."""
    # Create a target transform
    original_target = pin.SE3.Identity()
    original_target.translation = np.array([0.1, 0.2, 0.3])
    original_rotation = original_target.rotation.copy()

    # Set the target
    local_frame_task.set_target(original_target)

    # Modify the original target
    original_target.translation = np.array([0.5, 0.5, 0.5])
    original_target.rotation = pin.exp3(np.array([0.5, 0.0, 0.0]))

    # Check that the stored target is unchanged
    assert np.allclose(local_frame_task.transform_target_to_base.translation, np.array([0.1, 0.2, 0.3]))
    assert np.allclose(local_frame_task.transform_target_to_base.rotation, original_rotation)


def test_error_computation_with_orientation_difference(local_frame_task, pink_config):
    """Test error computation when there's an orientation difference."""
    # Set a target with orientation difference
    target_transform = pin.SE3.Identity()
    target_transform.rotation = pin.exp3(np.array([0.2, 0.0, 0.0]))  # Rotation around X-axis
    local_frame_task.set_target(target_transform)

    # Compute error
    error = local_frame_task.compute_error(pink_config)

    # Check that error is computed correctly
    assert isinstance(error, np.ndarray)
    assert error.shape == (6,)

    # Error should not be all zeros
    assert not np.allclose(error, 0.0)


def test_jacobian_rank_consistency(local_frame_task, pink_config):
    """Test that Jacobian maintains consistent shape across configurations."""
    # Set a target that we know can be reached by the test robot.
    target_transform = pin.SE3.Identity()
    target_transform.translation = np.array([0.0, 0.0, 0.45])
    # 90 degrees around x axis = pi/2 radians
    target_transform.rotation = pin.exp3(np.array([np.pi / 2, 0.0, 0.0]))
    local_frame_task.set_target(target_transform)

    # Compute Jacobian at multiple configurations
    jacobians = []
    for i in range(5):
        # Update configuration
        new_q = pink_config.full_q.copy()
        new_q[1] = 0.1 * i  # Vary first joint
        pink_config.update(new_q)

        # Compute Jacobian
        jacobian = local_frame_task.compute_jacobian(pink_config)
        jacobians.append(jacobian)

    # All Jacobians should have the same shape
    for jacobian in jacobians:
        assert jacobian.shape == (6, 2)

    # All Jacobians should have rank 2 (full rank for 2-DOF planar arm)
    for jacobian in jacobians:
        assert np.linalg.matrix_rank(jacobian) == 2
