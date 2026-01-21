# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from __future__ import annotations

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

# Rest everything follows.

import numpy as np
import pytest

from isaaclab.devices.openxr.osc_receiver import (
    BODY_JOINT_NAMES,
    DOF_PER_JOINT,
    NUM_BODY_JOINTS,
    BodyOscReceiver,
    _normalize,
    _quat_from_forward_up,
    _rotation_matrix_to_quat,
)


class TestUtilityFunctions:
    """Tests for utility functions in osc_receiver module."""

    def test_normalize_nonzero(self):
        """Test normalization of a non-zero vector."""
        v = np.array([3.0, 4.0, 0.0])
        result = _normalize(v)
        expected = np.array([0.6, 0.8, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_zero(self):
        """Test normalization of a zero vector returns unchanged."""
        v = np.array([0.0, 0.0, 0.0])
        result = _normalize(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_normalize_small(self):
        """Test normalization of a very small vector returns unchanged."""
        v = np.array([1e-8, 1e-8, 1e-8])
        result = _normalize(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_rotation_matrix_to_quat_identity(self):
        """Test identity rotation matrix gives identity quaternion."""
        R = np.eye(3)
        quat = _rotation_matrix_to_quat(R)
        # Identity quaternion is [0, 0, 0, 1]
        np.testing.assert_array_almost_equal(quat, [0.0, 0.0, 0.0, 1.0], decimal=5)

    def test_rotation_matrix_to_quat_90_z(self):
        """Test 90 degree rotation around Z axis."""
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        quat = _rotation_matrix_to_quat(R)
        # 90 degree rotation around Z: [0, 0, sin(45), cos(45)] = [0, 0, 0.707, 0.707]
        expected = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
        np.testing.assert_array_almost_equal(np.abs(quat), np.abs(expected), decimal=3)

    def test_quat_from_forward_up_x_forward(self):
        """Test quaternion from forward pointing in +X direction."""
        forward = np.array([1.0, 0.0, 0.0])
        quat = _quat_from_forward_up(forward)
        # With +X forward and +Z up, this should be close to identity
        # Check that the quaternion is normalized
        np.testing.assert_almost_equal(np.linalg.norm(quat), 1.0)

    def test_quat_from_forward_up_zero_forward(self):
        """Test quaternion from zero forward returns identity."""
        forward = np.array([0.0, 0.0, 0.0])
        quat = _quat_from_forward_up(forward)
        np.testing.assert_array_almost_equal(quat, [0.0, 0.0, 0.0, 1.0])


class TestBodyOscReceiverConstants:
    """Tests for OSC receiver constants."""

    def test_body_joint_names_count(self):
        """Test that NUM_BODY_JOINTS matches BODY_JOINT_NAMES length."""
        assert len(BODY_JOINT_NAMES) == NUM_BODY_JOINTS

    def test_dof_per_joint(self):
        """Test DOF per joint is 7 (pos + quat)."""
        assert DOF_PER_JOINT == 7

    def test_expected_joints(self):
        """Test expected joints are present."""
        expected = {
            "head",
            "hip",
            "chest",
            "left_foot",
            "right_foot",
            "left_knee",
            "right_knee",
            "left_elbow",
            "right_elbow",
        }
        assert set(BODY_JOINT_NAMES) == expected


class TestBodyOscReceiver:
    """Tests for BodyOscReceiver class."""

    @pytest.fixture
    def receiver(self):
        """Create a BodyOscReceiver instance for testing.

        Uses a non-standard port to avoid conflicts.
        """
        receiver = BodyOscReceiver(ip="127.0.0.1", port=19000)
        yield receiver
        # Cleanup: stop the server
        receiver.shutdown()

    def test_initialization(self, receiver):
        """Test receiver initializes with correct data shape."""
        data = receiver.get_matrix()
        assert data.shape == (NUM_BODY_JOINTS, DOF_PER_JOINT)

    def test_initial_positions_zero(self, receiver):
        """Test initial positions are zero."""
        data = receiver.get_matrix()
        positions = data[:, :3]
        np.testing.assert_array_equal(positions, np.zeros((NUM_BODY_JOINTS, 3)))

    def test_initial_quaternions_identity(self, receiver):
        """Test initial quaternions are identity."""
        data = receiver.get_matrix()
        quats = data[:, 3:]
        expected = np.tile([0.0, 0.0, 0.0, 1.0], (NUM_BODY_JOINTS, 1))
        np.testing.assert_array_equal(quats, expected)

    def test_get_flat_shape(self, receiver):
        """Test get_flat returns correct shape."""
        flat = receiver.get_flat()
        assert flat.shape == (NUM_BODY_JOINTS * DOF_PER_JOINT,)

    def test_get_position_valid_joint(self, receiver):
        """Test get_position for valid joint returns correct shape."""
        pos = receiver.get_position("head")
        assert pos.shape == (3,)

    def test_get_position_invalid_joint(self, receiver):
        """Test get_position raises for invalid joint."""
        with pytest.raises(ValueError, match="Unknown joint name"):
            receiver.get_position("invalid_joint")

    def test_get_pose_valid_joint(self, receiver):
        """Test get_pose for valid joint returns correct shape."""
        pose = receiver.get_pose("hip")
        assert pose.shape == (7,)

    def test_get_pose_invalid_joint(self, receiver):
        """Test get_pose raises for invalid joint."""
        with pytest.raises(ValueError, match="Unknown joint name"):
            receiver.get_pose("invalid_joint")

    def test_on_position_updates_data(self, receiver):
        """Test _on_position updates internal data correctly."""
        # Simulate receiving position data for head (index 0)
        receiver._on_position("/tracking/trackers/head/position", 1.0, 2.0, 3.0)

        # Note: coordinate swizzle (x, z, y) -> (x, y, z)
        # So input (1.0, 2.0, 3.0) becomes (1.0, 3.0, 2.0)
        pos = receiver.get_position("head")
        np.testing.assert_array_almost_equal(pos, [1.0, 3.0, 2.0])

    def test_on_position_by_index(self, receiver):
        """Test _on_position works with numeric indices."""
        # Index 1 is "hip"
        receiver._on_position("/tracking/trackers/1/position", 5.0, 6.0, 7.0)

        pos = receiver.get_position("hip")
        np.testing.assert_array_almost_equal(pos, [5.0, 7.0, 6.0])

    def test_on_position_invalid_index_ignored(self, receiver):
        """Test _on_position ignores invalid indices."""
        initial_data = receiver.get_matrix().copy()
        receiver._on_position("/tracking/trackers/999/position", 1.0, 2.0, 3.0)
        np.testing.assert_array_equal(receiver.get_matrix(), initial_data)

    def test_on_position_insufficient_args(self, receiver):
        """Test _on_position ignores messages with insufficient args."""
        initial_data = receiver.get_matrix().copy()
        receiver._on_position("/tracking/trackers/head/position", 1.0, 2.0)  # Only 2 args
        np.testing.assert_array_equal(receiver.get_matrix(), initial_data)

    def test_recompute_rotations(self, receiver):
        """Test recompute_rotations updates quaternions."""
        # Set some positions
        receiver._on_position("/tracking/trackers/hip/position", 0.0, 0.0, 0.0)
        receiver._on_position("/tracking/trackers/chest/position", 0.0, 1.0, 0.0)

        # Recompute rotations
        receiver.recompute_rotations()

        # Quaternion should be updated and normalized
        new_quat = receiver.get_pose("hip")[3:]

        # Since chest is above hip (forward direction), quaternion should be computed
        # The exact value depends on the heuristic, but it should be normalized
        assert new_quat.shape == (4,)
        np.testing.assert_almost_equal(np.linalg.norm(new_quat), 1.0)

    def test_thread_safety_get_matrix(self, receiver):
        """Test get_matrix returns a copy (thread safe)."""
        data1 = receiver.get_matrix()
        data1[0, 0] = 999.0
        data2 = receiver.get_matrix()
        assert data2[0, 0] != 999.0

    def test_thread_safety_get_flat(self, receiver):
        """Test get_flat returns a copy (thread safe)."""
        flat1 = receiver.get_flat()
        flat1[0] = 999.0
        flat2 = receiver.get_flat()
        assert flat2[0] != 999.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
