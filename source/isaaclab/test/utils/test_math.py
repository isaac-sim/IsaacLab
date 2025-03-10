# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

"""Launch Isaac Sim Simulator first.

This is only needed because of warp dependency.
"""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app in headless mode
if not AppLauncher.instance():
    simulation_app = AppLauncher(headless=True).app


"""Rest everything follows."""

import math
import numpy as np
import scipy.spatial.transform as scipy_tf
import torch
import torch.utils.benchmark as benchmark
from math import pi as PI

import isaaclab.utils.math as math_utils

DECIMAL_PRECISION = 5
"""Precision of the test.

This value is used since float operations are inexact. For reference:
https://github.com/pytorch/pytorch/issues/17678
"""


def test_is_identity_pose():
        """Test is_identity_pose method."""
        identity_pos_one_row = torch.zeros(3)
        identity_rot_one_row = torch.tensor((1.0, 0.0, 0.0, 0.0))

    assert math_utils.is_identity_pose(identity_pos_one_row, identity_rot_one_row)

        identity_pos_one_row[0] = 1.0
        identity_rot_one_row[1] = 1.0

    assert not math_utils.is_identity_pose(identity_pos_one_row, identity_rot_one_row)

        identity_pos_multi_row = torch.zeros(3, 3)
        identity_rot_multi_row = torch.zeros(3, 4)
        identity_rot_multi_row[:, 0] = 1.0

    assert math_utils.is_identity_pose(identity_pos_multi_row, identity_rot_multi_row)

        identity_pos_multi_row[0, 0] = 1.0
        identity_rot_multi_row[0, 1] = 1.0

    assert not math_utils.is_identity_pose(identity_pos_multi_row, identity_rot_multi_row)


def test_axis_angle_from_quat():
        """Test axis_angle_from_quat method."""
        # Quaternions of the form (2,4) and (2,2,4)
        quats = [
            torch.Tensor([[1.0, 0.0, 0.0, 0.0], [0.8418536, 0.142006, 0.0, 0.5206887]]),
            torch.Tensor([
                [[1.0, 0.0, 0.0, 0.0], [0.8418536, 0.142006, 0.0, 0.5206887]],
                [[1.0, 0.0, 0.0, 0.0], [0.9850375, 0.0995007, 0.0995007, 0.0995007]],
            ]),
        ]

        # Angles of the form (2,3) and (2,2,3)
        angles = [
            torch.Tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 1.1]]),
            torch.Tensor([[[0.0, 0.0, 0.0], [0.3, 0.0, 1.1]], [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]]]),
        ]

        for quat, angle in zip(quats, angles):
                torch.testing.assert_close(math_utils.axis_angle_from_quat(quat), angle)


def test_axis_angle_from_quat_approximation():
        """Test the Taylor approximation from axis_angle_from_quat method.

        This test checks for unstable conversions where theta is very small.
        """
        # Generate a small rotation quaternion
        # Small angle
        theta = torch.Tensor([0.0000001])
        # Arbitrary normalized axis of rotation in rads, (x,y,z)
        axis = [-0.302286, 0.205494, -0.930803]
        # Generate quaternion
        qw = torch.cos(theta / 2)
        quat_vect = [qw] + [d * torch.sin(theta / 2) for d in axis]
        quaternion = torch.tensor(quat_vect, dtype=torch.float32)

        # Convert quaternion to axis-angle
        axis_angle_computed = math_utils.axis_angle_from_quat(quaternion)

        # Expected axis-angle representation
        axis_angle_expected = torch.tensor([theta * d for d in axis], dtype=torch.float32)

        # Assert that the computed values are close to the expected values
        torch.testing.assert_close(axis_angle_computed, axis_angle_expected)


def test_quat_error_magnitude():
        """Test quat_error_magnitude method."""
        # Define test cases
        # Each tuple contains: q1, q2, expected error
        test_cases = [
            # No rotation
            (torch.Tensor([1, 0, 0, 0]), torch.Tensor([1, 0, 0, 0]), torch.Tensor([0.0])),
            # PI/2 rotation
            (torch.Tensor([1.0, 0, 0.0, 0]), torch.Tensor([0.7071068, 0.7071068, 0, 0]), torch.Tensor([PI / 2])),
            # PI rotation
            (torch.Tensor([1.0, 0, 0.0, 0]), torch.Tensor([0.0, 0.0, 1.0, 0]), torch.Tensor([PI])),
        ]
        # Test higher dimension (batched) inputs
        q1_list = torch.stack([t[0] for t in test_cases], dim=0)
        q2_list = torch.stack([t[1] for t in test_cases], dim=0)
        expected_diff_list = torch.stack([t[2] for t in test_cases], dim=0).flatten()
        test_cases += [(q1_list, q2_list, expected_diff_list)]

        # Iterate over test cases
        for q1, q2, expected_diff in test_cases:
                # Compute the error
                q12_diff = math_utils.quat_error_magnitude(q1, q2)

                # Check that the error is close to the expected value
                if len(q1.shape) > 1:
                    torch.testing.assert_close(q12_diff, expected_diff)
                else:
            assert abs(q12_diff.item() - expected_diff.item()) < 1e-5


def test_quat_unique():
        """Test quat_unique method."""
        # Define test cases
        quats = math_utils.random_orientation(num=1024, device="cpu")

        # Test positive real quaternion
        pos_real_quats = math_utils.quat_unique(quats)

        # Test that the real part is positive
    assert torch.all(pos_real_quats[:, 0] > 0).item()

        non_pos_indices = quats[:, 0] < 0
        # Check imaginary part have sign flipped if real part is negative
        torch.testing.assert_close(pos_real_quats[non_pos_indices], -quats[non_pos_indices])
        torch.testing.assert_close(pos_real_quats[~non_pos_indices], quats[~non_pos_indices])


def test_quat_mul_with_quat_unique():
        """Test quat_mul method with different quaternions.

        This test checks that the quaternion multiplication is consistent when using positive real quaternions
        and regular quaternions. It makes sure that the result is the same regardless of the input quaternion sign
        (i.e. q and -q are same quaternion in the context of rotations).
        """

        quats_1 = math_utils.random_orientation(num=1024, device="cpu")
        quats_2 = math_utils.random_orientation(num=1024, device="cpu")
        # Make quats positive real
        quats_1_pos_real = math_utils.quat_unique(quats_1)
        quats_2_pos_real = math_utils.quat_unique(quats_2)

        # Option 1: Direct computation on quaternions
        quat_result_1 = math_utils.quat_mul(quats_1, math_utils.quat_conjugate(quats_2))
        quat_result_1 = math_utils.quat_unique(quat_result_1)

        # Option 2: Computation on positive real quaternions
        quat_result_2 = math_utils.quat_mul(quats_1_pos_real, math_utils.quat_conjugate(quats_2_pos_real))
        quat_result_2 = math_utils.quat_unique(quat_result_2)

        # Option 3: Mixed computation
        quat_result_3 = math_utils.quat_mul(quats_1, math_utils.quat_conjugate(quats_2_pos_real))
        quat_result_3 = math_utils.quat_unique(quat_result_3)

        # Check that the result is close to the expected value
        torch.testing.assert_close(quat_result_1, quat_result_2)
        torch.testing.assert_close(quat_result_2, quat_result_3)
        torch.testing.assert_close(quat_result_3, quat_result_1)


def test_quat_error_mag_with_quat_unique():
        """Test quat_error_magnitude method with positive real quaternions."""

        quats_1 = math_utils.random_orientation(num=1024, device="cpu")
        quats_2 = math_utils.random_orientation(num=1024, device="cpu")
        # Make quats positive real
        quats_1_pos_real = math_utils.quat_unique(quats_1)
        quats_2_pos_real = math_utils.quat_unique(quats_2)

        # Compute the error
        error_1 = math_utils.quat_error_magnitude(quats_1, quats_2)
        error_2 = math_utils.quat_error_magnitude(quats_1_pos_real, quats_2_pos_real)
        error_3 = math_utils.quat_error_magnitude(quats_1, quats_2_pos_real)
        error_4 = math_utils.quat_error_magnitude(quats_1_pos_real, quats_2)

        # Check that the error is close to the expected value
        torch.testing.assert_close(error_1, error_2)
        torch.testing.assert_close(error_2, error_3)
        torch.testing.assert_close(error_3, error_4)
        torch.testing.assert_close(error_4, error_1)


def test_convention_converter():
        """Test convert_camera_frame_orientation_convention to and from ros, opengl, and world conventions."""
        quat_ros = torch.tensor([[-0.17591989, 0.33985114, 0.82047325, -0.42470819]])
        quat_opengl = torch.tensor([[0.33985113, 0.17591988, 0.42470818, 0.82047324]])
        quat_world = torch.tensor([[-0.3647052, -0.27984815, -0.1159169, 0.88047623]])

        # from ROS
        torch.testing.assert_close(
            math_utils.convert_camera_frame_orientation_convention(quat_ros, "ros", "opengl"), quat_opengl
        )
        torch.testing.assert_close(
            math_utils.convert_camera_frame_orientation_convention(quat_ros, "ros", "world"), quat_world
        )
        torch.testing.assert_close(
            math_utils.convert_camera_frame_orientation_convention(quat_ros, "ros", "ros"), quat_ros
        )
        # from OpenGL
        torch.testing.assert_close(
            math_utils.convert_camera_frame_orientation_convention(quat_opengl, "opengl", "ros"), quat_ros
        )
        torch.testing.assert_close(
            math_utils.convert_camera_frame_orientation_convention(quat_opengl, "opengl", "world"), quat_world
        )
        torch.testing.assert_close(
            math_utils.convert_camera_frame_orientation_convention(quat_opengl, "opengl", "opengl"), quat_opengl
        )
        # from World
        torch.testing.assert_close(
            math_utils.convert_camera_frame_orientation_convention(quat_world, "world", "ros"), quat_ros
        )
        torch.testing.assert_close(
            math_utils.convert_camera_frame_orientation_convention(quat_world, "world", "opengl"), quat_opengl
        )
        torch.testing.assert_close(
            math_utils.convert_camera_frame_orientation_convention(quat_world, "world", "world"), quat_world
        )


def test_wrap_to_pi():
        """Test wrap_to_pi method."""
        # Define test cases
        # Each tuple contains: angle, expected wrapped angle
        test_cases = [
            # No wrapping needed
            (torch.Tensor([0.0]), torch.Tensor([0.0])),
        # Wrapping needed
        (torch.Tensor([2.0 * PI]), torch.Tensor([0.0])),
        (torch.Tensor([-2.0 * PI]), torch.Tensor([0.0])),
            (torch.Tensor([PI]), torch.Tensor([PI])),
            (torch.Tensor([-PI]), torch.Tensor([-PI])),
        (torch.Tensor([PI + 0.1]), torch.Tensor([-PI + 0.1])),
        (torch.Tensor([-PI - 0.1]), torch.Tensor([PI - 0.1])),
    ]

    # Test higher dimension (batched) inputs
    angles_list = torch.stack([t[0] for t in test_cases], dim=0)
    expected_angles_list = torch.stack([t[1] for t in test_cases], dim=0)
    test_cases += [(angles_list, expected_angles_list)]

        # Iterate over test cases
            for angle, expected_angle in test_cases:
                    # Compute the wrapped angle
                    wrapped_angle = math_utils.wrap_to_pi(angle)

                    # Check that the wrapped angle is close to the expected value
        if len(angle.shape) > 1:
                    torch.testing.assert_close(wrapped_angle, expected_angle)
        else:
            assert abs(wrapped_angle.item() - expected_angle.item()) < 1e-5


def test_quat_rotate_and_quat_rotate_inverse():
    """Test quat_rotate and quat_rotate_inverse methods."""
    # Define test cases
    # Each tuple contains: q, v, expected rotated vector
    test_cases = [
        # No rotation
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([1, 0, 0]), torch.Tensor([1, 0, 0])),
        # PI/2 rotation around z-axis
        (torch.Tensor([0.7071068, 0, 0, 0.7071068]), torch.Tensor([1, 0, 0]), torch.Tensor([0, 1, 0])),
        # PI rotation around z-axis
        (torch.Tensor([0, 0, 0, 1]), torch.Tensor([1, 0, 0]), torch.Tensor([-1, 0, 0])),
    ]

    # Test higher dimension (batched) inputs
    q_list = torch.stack([t[0] for t in test_cases], dim=0)
    v_list = torch.stack([t[1] for t in test_cases], dim=0)
    expected_v_list = torch.stack([t[2] for t in test_cases], dim=0)
    test_cases += [(q_list, v_list, expected_v_list)]

    # Iterate over test cases
    for q, v, expected_v in test_cases:
        # Compute the rotated vector
        rotated_v = math_utils.quat_rotate(q, v)
        # Compute the inverse rotation
        inverse_rotated_v = math_utils.quat_rotate_inverse(q, rotated_v)

        # Check that the rotated vector is close to the expected value
        if len(q.shape) > 1:
            torch.testing.assert_close(rotated_v, expected_v)
            torch.testing.assert_close(inverse_rotated_v, v)
        else:
            assert torch.allclose(rotated_v, expected_v)
            assert torch.allclose(inverse_rotated_v, v)


def test_orthogonalize_perspective_depth():
    """Test orthogonalize_perspective_depth method."""
    # Define test cases
    # Each tuple contains: depth, expected orthogonalized depth
    test_cases = [
        # No orthogonalization needed
        (torch.Tensor([1.0]), torch.Tensor([1.0])),
        # Orthogonalization needed
        (torch.Tensor([0.0]), torch.Tensor([0.0])),
        (torch.Tensor([-1.0]), torch.Tensor([0.0])),
    ]

    # Test higher dimension (batched) inputs
    depth_list = torch.stack([t[0] for t in test_cases], dim=0)
    expected_depth_list = torch.stack([t[1] for t in test_cases], dim=0)
    test_cases += [(depth_list, expected_depth_list)]

    # Iterate over test cases
    for depth, expected_depth in test_cases:
        # Compute the orthogonalized depth
        orthogonalized_depth = math_utils.orthogonalize_perspective_depth(depth)

        # Check that the orthogonalized depth is close to the expected value
        if len(depth.shape) > 1:
            torch.testing.assert_close(orthogonalized_depth, expected_depth)
        else:
            assert abs(orthogonalized_depth.item() - expected_depth.item()) < 1e-5


def test_combine_frame_transform():
    """Test combine_frame_transform method."""
    # Define test cases
    # Each tuple contains: q1, v1, q2, v2, expected combined transform
    test_cases = [
        # No transform
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([0, 0, 0]), torch.Tensor([1, 0, 0, 0]), torch.Tensor([0, 0, 0])),
        # Translation only
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([1, 0, 0]), torch.Tensor([1, 0, 0, 0]), torch.Tensor([0, 0, 0])),
        # Rotation only
        (torch.Tensor([0.7071068, 0, 0, 0.7071068]), torch.Tensor([0, 0, 0]), torch.Tensor([1, 0, 0, 0]), torch.Tensor([0, 0, 0])),
        # Translation and rotation
        (torch.Tensor([0.7071068, 0, 0, 0.7071068]), torch.Tensor([1, 0, 0]), torch.Tensor([1, 0, 0, 0]), torch.Tensor([0, 0, 0])),
    ]

    # Test higher dimension (batched) inputs
    q1_list = torch.stack([t[0] for t in test_cases], dim=0)
    v1_list = torch.stack([t[1] for t in test_cases], dim=0)
    q2_list = torch.stack([t[2] for t in test_cases], dim=0)
    v2_list = torch.stack([t[3] for t in test_cases], dim=0)
    test_cases += [(q1_list, v1_list, q2_list, v2_list)]

    # Iterate over test cases
    for q1, v1, q2, v2 in test_cases:
        # Compute the combined transform
        combined_q, combined_v = math_utils.combine_frame_transform(q1, v1, q2, v2)

        # Check that the combined transform is valid
        if len(q1.shape) > 1:
            assert torch.all(torch.abs(torch.norm(combined_q, dim=1) - 1.0) < 1e-5)
        else:
            assert abs(torch.norm(combined_q) - 1.0) < 1e-5


def test_pose_inv():
    """Test pose_inv method."""
    # Define test cases
    # Each tuple contains: q, v, expected inverse pose
    test_cases = [
        # No transform
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([0, 0, 0])),
        # Translation only
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([1, 0, 0])),
        # Rotation only
        (torch.Tensor([0.7071068, 0, 0, 0.7071068]), torch.Tensor([0, 0, 0])),
        # Translation and rotation
        (torch.Tensor([0.7071068, 0, 0, 0.7071068]), torch.Tensor([1, 0, 0])),
    ]

    # Test higher dimension (batched) inputs
    q_list = torch.stack([t[0] for t in test_cases], dim=0)
    v_list = torch.stack([t[1] for t in test_cases], dim=0)
    test_cases += [(q_list, v_list)]

    # Iterate over test cases
    for q, v in test_cases:
        # Compute the inverse pose
        inv_q, inv_v = math_utils.pose_inv(q, v)

        # Check that the inverse pose is valid
        if len(q.shape) > 1:
            assert torch.all(torch.abs(torch.norm(inv_q, dim=1) - 1.0) < 1e-5)
        else:
            assert abs(torch.norm(inv_q) - 1.0) < 1e-5


def test_quat_slerp():
    """Test quat_slerp method."""
    # Define test cases
    # Each tuple contains: q1, q2, t, expected interpolated quaternion
    test_cases = [
        # No interpolation
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([1, 0, 0, 0]), torch.Tensor([0.0]), torch.Tensor([1, 0, 0, 0])),
        # Full interpolation
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([1, 0, 0, 0]), torch.Tensor([1.0]), torch.Tensor([1, 0, 0, 0])),
        # Half interpolation
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([0.7071068, 0, 0, 0.7071068]), torch.Tensor([0.5]), torch.Tensor([0.9238795, 0, 0, 0.3826834])),
    ]

    # Test higher dimension (batched) inputs
    q1_list = torch.stack([t[0] for t in test_cases], dim=0)
    q2_list = torch.stack([t[1] for t in test_cases], dim=0)
    t_list = torch.stack([t[2] for t in test_cases], dim=0)
    expected_q_list = torch.stack([t[3] for t in test_cases], dim=0)
    test_cases += [(q1_list, q2_list, t_list, expected_q_list)]

    # Iterate over test cases
    for q1, q2, t, expected_q in test_cases:
        # Compute the interpolated quaternion
        interpolated_q = math_utils.quat_slerp(q1, q2, t)

        # Check that the interpolated quaternion is close to the expected value
        if len(q1.shape) > 1:
            torch.testing.assert_close(interpolated_q, expected_q)
        else:
            assert torch.allclose(interpolated_q, expected_q)


def test_interpolate_rotations():
    """Test interpolate_rotations method."""
    # Define test cases
    # Each tuple contains: q1, q2, t, expected interpolated quaternion
    test_cases = [
        # No interpolation
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([1, 0, 0, 0]), torch.Tensor([0.0]), torch.Tensor([1, 0, 0, 0])),
        # Full interpolation
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([1, 0, 0, 0]), torch.Tensor([1.0]), torch.Tensor([1, 0, 0, 0])),
        # Half interpolation
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([0.7071068, 0, 0, 0.7071068]), torch.Tensor([0.5]), torch.Tensor([0.9238795, 0, 0, 0.3826834])),
    ]

    # Test higher dimension (batched) inputs
    q1_list = torch.stack([t[0] for t in test_cases], dim=0)
    q2_list = torch.stack([t[1] for t in test_cases], dim=0)
    t_list = torch.stack([t[2] for t in test_cases], dim=0)
    expected_q_list = torch.stack([t[3] for t in test_cases], dim=0)
    test_cases += [(q1_list, q2_list, t_list, expected_q_list)]

    # Iterate over test cases
    for q1, q2, t, expected_q in test_cases:
        # Compute the interpolated quaternion
        interpolated_q = math_utils.interpolate_rotations(q1, q2, t)

        # Check that the interpolated quaternion is close to the expected value
        if len(q1.shape) > 1:
            torch.testing.assert_close(interpolated_q, expected_q)
        else:
            assert torch.allclose(interpolated_q, expected_q)


def test_interpolate_poses():
    """Test interpolate_poses method."""
    # Define test cases
    # Each tuple contains: q1, v1, q2, v2, t, expected interpolated pose
    test_cases = [
        # No interpolation
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([0, 0, 0]), torch.Tensor([1, 0, 0, 0]), torch.Tensor([0, 0, 0]), torch.Tensor([0.0])),
        # Full interpolation
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([0, 0, 0]), torch.Tensor([1, 0, 0, 0]), torch.Tensor([0, 0, 0]), torch.Tensor([1.0])),
        # Half interpolation
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([0, 0, 0]), torch.Tensor([0.7071068, 0, 0, 0.7071068]), torch.Tensor([1, 0, 0]), torch.Tensor([0.5])),
    ]

    # Test higher dimension (batched) inputs
    q1_list = torch.stack([t[0] for t in test_cases], dim=0)
    v1_list = torch.stack([t[1] for t in test_cases], dim=0)
    q2_list = torch.stack([t[2] for t in test_cases], dim=0)
    v2_list = torch.stack([t[3] for t in test_cases], dim=0)
    t_list = torch.stack([t[4] for t in test_cases], dim=0)
    test_cases += [(q1_list, v1_list, q2_list, v2_list, t_list)]

    # Iterate over test cases
    for q1, v1, q2, v2, t in test_cases:
        # Compute the interpolated pose
        interpolated_q, interpolated_v = math_utils.interpolate_poses(q1, v1, q2, v2, t)

        # Check that the interpolated pose is valid
        if len(q1.shape) > 1:
            assert torch.all(torch.abs(torch.norm(interpolated_q, dim=1) - 1.0) < 1e-5)
        else:
            assert abs(torch.norm(interpolated_q) - 1.0) < 1e-5



