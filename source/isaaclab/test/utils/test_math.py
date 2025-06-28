# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first.

This is only needed because of warp dependency.
"""

from isaaclab.app import AppLauncher

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app


"""Rest everything follows."""

import math
import numpy as np
import scipy.spatial.transform as scipy_tf
import torch
import torch.utils.benchmark as benchmark
from math import pi as PI

import pytest

import isaaclab.utils.math as math_utils

DECIMAL_PRECISION = 5
"""Precision of the test.

This value is used since float operations are inexact. For reference:
https://github.com/pytorch/pytorch/issues/17678
"""


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_is_identity_pose(device):
    """Test is_identity_pose method."""
    # Single row identity pose
    identity_pos = torch.zeros(3, device=device)
    identity_rot = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device)
    assert math_utils.is_identity_pose(identity_pos, identity_rot) is True

    # Modified single row pose
    identity_pos = torch.tensor([1.0, 0.0, 0.0], device=device)
    identity_rot = torch.tensor((1.0, 1.0, 0.0, 0.0), device=device)
    assert math_utils.is_identity_pose(identity_pos, identity_rot) is False

    # Multi-row identity pose
    identity_pos = torch.zeros(3, 3, device=device)
    identity_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)
    assert math_utils.is_identity_pose(identity_pos, identity_rot) is True

    # Modified multi-row pose
    identity_pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device)
    identity_rot = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device)
    assert math_utils.is_identity_pose(identity_pos, identity_rot) is False


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_axis_angle_from_quat(device):
    """Test axis_angle_from_quat method."""
    # Quaternions of the form (2,4) and (2,2,4)
    quats = [
        torch.Tensor([[1.0, 0.0, 0.0, 0.0], [0.8418536, 0.142006, 0.0, 0.5206887]]).to(device),
        torch.Tensor([
            [[1.0, 0.0, 0.0, 0.0], [0.8418536, 0.142006, 0.0, 0.5206887]],
            [[1.0, 0.0, 0.0, 0.0], [0.9850375, 0.0995007, 0.0995007, 0.0995007]],
        ]).to(device),
    ]

    # Angles of the form (2,3) and (2,2,3)
    angles = [
        torch.Tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 1.1]]).to(device),
        torch.Tensor([[[0.0, 0.0, 0.0], [0.3, 0.0, 1.1]], [[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]]]).to(device),
    ]

    for quat, angle in zip(quats, angles):
        torch.testing.assert_close(math_utils.axis_angle_from_quat(quat), angle)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_axis_angle_from_quat_approximation(device):
    """Test the Taylor approximation from axis_angle_from_quat method.

    This test checks for unstable conversions where theta is very small.
    """
    # Generate a small rotation quaternion
    # Small angle
    theta = torch.Tensor([0.0000001]).to(device)
    # Arbitrary normalized axis of rotation in rads, (x,y,z)
    axis = [-0.302286, 0.205494, -0.930803]
    # Generate quaternion
    qw = torch.cos(theta / 2)
    quat_vect = [qw] + [d * torch.sin(theta / 2) for d in axis]
    quaternion = torch.tensor(quat_vect, dtype=torch.float32, device=device)

    # Convert quaternion to axis-angle
    axis_angle_computed = math_utils.axis_angle_from_quat(quaternion)

    # Expected axis-angle representation
    axis_angle_expected = torch.tensor([theta * d for d in axis], dtype=torch.float32, device=device)

    # Assert that the computed values are close to the expected values
    torch.testing.assert_close(axis_angle_computed, axis_angle_expected)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_quat_error_magnitude(device):
    """Test quat_error_magnitude method."""
    # No rotation
    q1 = torch.Tensor([1, 0, 0, 0]).to(device)
    q2 = torch.Tensor([1, 0, 0, 0]).to(device)
    expected_diff = torch.Tensor([0.0]).to(device)
    q12_diff = math_utils.quat_error_magnitude(q1, q2)
    assert math.isclose(q12_diff.item(), expected_diff.item(), rel_tol=1e-5)

    # PI/2 rotation
    q1 = torch.Tensor([1.0, 0, 0.0, 0]).to(device)
    q2 = torch.Tensor([0.7071068, 0.7071068, 0, 0]).to(device)
    expected_diff = torch.Tensor([PI / 2]).to(device)
    q12_diff = math_utils.quat_error_magnitude(q1, q2)
    assert math.isclose(q12_diff.item(), expected_diff.item(), rel_tol=1e-5)

    # PI rotation
    q1 = torch.Tensor([1.0, 0, 0.0, 0]).to(device)
    q2 = torch.Tensor([0.0, 0.0, 1.0, 0]).to(device)
    expected_diff = torch.Tensor([PI]).to(device)
    q12_diff = math_utils.quat_error_magnitude(q1, q2)
    assert math.isclose(q12_diff.item(), expected_diff.item(), rel_tol=1e-5)

    # Batched inputs
    q1 = torch.stack(
        [torch.Tensor([1, 0, 0, 0]), torch.Tensor([1.0, 0, 0.0, 0]), torch.Tensor([1.0, 0, 0.0, 0])], dim=0
    ).to(device)
    q2 = torch.stack(
        [torch.Tensor([1, 0, 0, 0]), torch.Tensor([0.7071068, 0.7071068, 0, 0]), torch.Tensor([0.0, 0.0, 1.0, 0])],
        dim=0,
    ).to(device)
    expected_diff = (
        torch.stack([torch.Tensor([0.0]), torch.Tensor([PI / 2]), torch.Tensor([PI])], dim=0).flatten().to(device)
    )
    q12_diff = math_utils.quat_error_magnitude(q1, q2)
    torch.testing.assert_close(q12_diff, expected_diff)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_quat_unique(device):
    """Test quat_unique method."""
    # Define test cases
    quats = math_utils.random_orientation(num=1024, device=device)

    # Test positive real quaternion
    pos_real_quats = math_utils.quat_unique(quats)

    # Test that the real part is positive
    assert torch.all(pos_real_quats[:, 0] > 0).item()

    non_pos_indices = quats[:, 0] < 0
    # Check imaginary part have sign flipped if real part is negative
    torch.testing.assert_close(pos_real_quats[non_pos_indices], -quats[non_pos_indices])
    torch.testing.assert_close(pos_real_quats[~non_pos_indices], quats[~non_pos_indices])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_quat_mul_with_quat_unique(device):
    """Test quat_mul method with different quaternions.

    This test checks that the quaternion multiplication is consistent when using positive real quaternions
    and regular quaternions. It makes sure that the result is the same regardless of the input quaternion sign
    (i.e. q and -q are same quaternion in the context of rotations).
    """

    quats_1 = math_utils.random_orientation(num=1024, device=device)
    quats_2 = math_utils.random_orientation(num=1024, device=device)
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


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_quat_error_mag_with_quat_unique(device):
    """Test quat_error_magnitude method with positive real quaternions."""

    quats_1 = math_utils.random_orientation(num=1024, device=device)
    quats_2 = math_utils.random_orientation(num=1024, device=device)
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


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_convention_converter(device):
    """Test convert_camera_frame_orientation_convention to and from ros, opengl, and world conventions."""
    quat_ros = torch.tensor([[-0.17591989, 0.33985114, 0.82047325, -0.42470819]], device=device)
    quat_opengl = torch.tensor([[0.33985113, 0.17591988, 0.42470818, 0.82047324]], device=device)
    quat_world = torch.tensor([[-0.3647052, -0.27984815, -0.1159169, 0.88047623]], device=device)

    # from ROS
    torch.testing.assert_close(
        math_utils.convert_camera_frame_orientation_convention(quat_ros, "ros", "opengl"), quat_opengl
    )
    torch.testing.assert_close(
        math_utils.convert_camera_frame_orientation_convention(quat_ros, "ros", "world"), quat_world
    )
    torch.testing.assert_close(math_utils.convert_camera_frame_orientation_convention(quat_ros, "ros", "ros"), quat_ros)
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


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_wrap_to_pi(device):
    """Test wrap_to_pi method."""
    # No wrapping needed
    angle = torch.Tensor([0.0]).to(device)
    expected_angle = torch.Tensor([0.0]).to(device)
    wrapped_angle = math_utils.wrap_to_pi(angle)
    torch.testing.assert_close(wrapped_angle, expected_angle)

    # Positive angle
    angle = torch.Tensor([PI]).to(device)
    expected_angle = torch.Tensor([PI]).to(device)
    wrapped_angle = math_utils.wrap_to_pi(angle)
    torch.testing.assert_close(wrapped_angle, expected_angle)

    # Negative angle
    angle = torch.Tensor([-PI]).to(device)
    expected_angle = torch.Tensor([-PI]).to(device)
    wrapped_angle = math_utils.wrap_to_pi(angle)
    torch.testing.assert_close(wrapped_angle, expected_angle)

    # Multiple angles
    angle = torch.Tensor([3 * PI, -3 * PI, 4 * PI, -4 * PI]).to(device)
    expected_angle = torch.Tensor([PI, -PI, 0.0, 0.0]).to(device)
    wrapped_angle = math_utils.wrap_to_pi(angle)
    torch.testing.assert_close(wrapped_angle, expected_angle)

    # Multiple angles from MATLAB docs
    # fmt: off
    angle = torch.Tensor([-2 * PI, - PI - 0.1, -PI, -2.8, 3.1, PI, PI + 0.001, PI + 1, 2 * PI, 2 * PI + 0.1]).to(device)
    expected_angle = torch.Tensor([0.0, PI - 0.1, -PI, -2.8, 3.1 , PI, -PI + 0.001, -PI + 1 , 0.0, 0.1]).to(device)
    # fmt: on
    wrapped_angle = math_utils.wrap_to_pi(angle)
    torch.testing.assert_close(wrapped_angle, expected_angle)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_orthogonalize_perspective_depth(device):
    """Test for converting perspective depth to orthogonal depth."""
    # Create a sample perspective depth image (N, H, W)
    perspective_depth = torch.tensor([[[10.0, 0.0, 100.0], [0.0, 3000.0, 0.0], [100.0, 0.0, 100.0]]], device=device)

    # Create sample intrinsic matrix (3, 3)
    intrinsics = torch.tensor([[500.0, 0.0, 5.0], [0.0, 500.0, 5.0], [0.0, 0.0, 1.0]], device=device)

    # Convert perspective depth to orthogonal depth
    orthogonal_depth = math_utils.orthogonalize_perspective_depth(perspective_depth, intrinsics)

    # Manually compute expected orthogonal depth based on the formula for comparison
    expected_orthogonal_depth = torch.tensor(
        [[[9.9990, 0.0000, 99.9932], [0.0000, 2999.8079, 0.0000], [99.9932, 0.0000, 99.9964]]], device=device
    )

    # Assert that the output is close to the expected result
    torch.testing.assert_close(orthogonal_depth, expected_orthogonal_depth)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_combine_frame_transform(device):
    """Test combine_frame_transforms function."""
    # create random poses
    pose01 = torch.rand(1, 7, device=device)
    pose01[:, 3:7] = torch.nn.functional.normalize(pose01[..., 3:7], dim=-1)

    pose12 = torch.rand(1, 7, device=device)
    pose12[:, 3:7] = torch.nn.functional.normalize(pose12[..., 3:7], dim=-1)

    # apply combination of poses
    pos02, quat02 = math_utils.combine_frame_transforms(
        pose01[..., :3], pose01[..., 3:7], pose12[:, :3], pose12[:, 3:7]
    )
    # apply combination of poses w.r.t. inverse to get original frame
    pos01, quat01 = math_utils.combine_frame_transforms(
        pos02,
        quat02,
        math_utils.quat_rotate(math_utils.quat_inv(pose12[:, 3:7]), -pose12[:, :3]),
        math_utils.quat_inv(pose12[:, 3:7]),
    )

    torch.testing.assert_close(pose01, torch.cat((pos01, quat01), dim=-1))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_interpolate_poses(device):
    """Test interpolate_poses function.

    This test checks the output from the :meth:`~isaaclab.utils.math_utils.interpolate_poses` function against
    the output from :func:`scipy.spatial.transform.Slerp` and :func:`np.linspace`.
    """
    for _ in range(100):
        mat1 = math_utils.generate_random_transformation_matrix()
        mat2 = math_utils.generate_random_transformation_matrix()
        pos_1, rmat1 = math_utils.unmake_pose(mat1)
        pos_2, rmat2 = math_utils.unmake_pose(mat2)

        # Compute expected results using scipy's Slerp
        key_rots = scipy_tf.Rotation.from_matrix(np.array([rmat1, rmat2]))

        # Create a Slerp object and interpolate create the interpolated rotation matrices
        num_steps = np.random.randint(3, 51)
        key_times = [0, 1]
        slerp = scipy_tf.Slerp(key_times, key_rots)
        interp_times = np.linspace(0, 1, num_steps)
        expected_quat = slerp(interp_times).as_matrix()

        # Test interpolation against expected result using np.linspace
        expected_pos = np.linspace(pos_1, pos_2, num_steps)

        # interpolate_poses using interpolate_poses and quat_slerp
        interpolated_poses, _ = math_utils.interpolate_poses(
            math_utils.make_pose(pos_1, rmat1), math_utils.make_pose(pos_2, rmat2), num_steps - 2
        )
        result_pos, result_quat = math_utils.unmake_pose(interpolated_poses)

        # Assert that the result is almost equal to the expected quaternion
        np.testing.assert_array_almost_equal(result_quat, expected_quat, decimal=DECIMAL_PRECISION)
        np.testing.assert_array_almost_equal(result_pos, expected_pos, decimal=DECIMAL_PRECISION)


def test_pose_inv():
    """Test pose_inv function.

    This test checks the output from the :meth:`~isaaclab.utils.math_utils.pose_inv` function against
    the output from :func:`np.linalg.inv`. Two test cases are performed:

    1. Checking the inverse of a random transformation matrix matches Numpy's built-in inverse.
    2. Checking the inverse of a batch of random transformation matrices matches Numpy's built-in inverse.
    """
    # Check against a single matrix
    for _ in range(100):
        test_mat = math_utils.generate_random_transformation_matrix(pos_boundary=10, rot_boundary=(2 * np.pi))
        result = np.array(math_utils.pose_inv(test_mat))
        expected = np.linalg.inv(np.array(test_mat))
        np.testing.assert_array_almost_equal(result, expected, decimal=DECIMAL_PRECISION)

    # Check against a batch of matrices
    test_mats = torch.stack([
        math_utils.generate_random_transformation_matrix(pos_boundary=10, rot_boundary=(2 * math.pi))
        for _ in range(100)
    ])
    result = np.array(math_utils.pose_inv(test_mats))
    expected = np.linalg.inv(np.array(test_mats))
    np.testing.assert_array_almost_equal(result, expected, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_quat_box_minus(device):
    """Test quat_box_minus method.

    Ensures that quat_box_minus correctly computes the axis-angle difference
    between two quaternions representing rotations around the same axis.
    """
    axis_angles = torch.tensor([0.0, 0.0, 1.0], device=device)
    angle_a = math.pi - 0.1
    angle_b = -math.pi + 0.1
    quat_a = math_utils.quat_from_angle_axis(torch.tensor([angle_a], device=device), axis_angles)
    quat_b = math_utils.quat_from_angle_axis(torch.tensor([angle_b], device=device), axis_angles)

    axis_diff = math_utils.quat_box_minus(quat_a, quat_b).squeeze(0)
    expected_diff = axis_angles * math_utils.wrap_to_pi(torch.tensor(angle_a - angle_b, device=device))
    torch.testing.assert_close(expected_diff, axis_diff, atol=1e-06, rtol=1e-06)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_quat_box_minus_and_quat_box_plus(device):
    """Test consistency of quat_box_plus and quat_box_minus.

    Checks that applying quat_box_plus to accumulate rotations and then using
    quat_box_minus to retrieve differences results in expected values.
    """

    # Perform closed-loop integration using quat_box_plus to accumulate rotations,
    # and then use quat_box_minus to compute the incremental differences between quaternions.
    # NOTE: Accuracy may decrease for very small angle increments due to numerical precision limits.
    for n in (2, 10, 100, 1000):
        # Define small incremental rotations around principal axes
        delta_angle = torch.tensor(
            [
                [0, 0, -math.pi / n],
                [0, -math.pi / n, 0],
                [-math.pi / n, 0, 0],
                [0, 0, math.pi / n],
                [0, math.pi / n, 0],
                [math.pi / n, 0, 0],
            ],
            device=device,
        )

        # Initialize quaternion trajectory starting from identity quaternion
        quat_trajectory = torch.zeros((len(delta_angle), 2 * n + 1, 4), device=device)
        quat_trajectory[:, 0, :] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(len(delta_angle), 1)

        # Integrate incremental rotations forward to form a closed loop trajectory
        for i in range(1, 2 * n + 1):
            quat_trajectory[:, i] = math_utils.quat_box_plus(quat_trajectory[:, i - 1], delta_angle)

        # Validate the loop closure: start and end quaternions should be approximately equal
        torch.testing.assert_close(quat_trajectory[:, 0], quat_trajectory[:, -1], atol=1e-04, rtol=1e-04)

        # Validate that the differences between consecutive quaternions match the original increments
        for i in range(2 * n):
            delta_result = math_utils.quat_box_minus(quat_trajectory[:, i + 1], quat_trajectory[:, i])
            torch.testing.assert_close(delta_result, delta_angle, atol=1e-04, rtol=1e-04)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_rigid_body_twist_transform(device):
    """Test rigid_body_twist_transform method.

    Verifies correct transformation of twists (linear and angular velocity) between coordinate frames.
    """
    num_bodies = 100
    # Frame A to B
    t_AB = torch.randn((num_bodies, 3), device=device)
    q_AB = math_utils.random_orientation(num=num_bodies, device=device)

    # Twists in A in frame A
    v_AA = torch.randn((num_bodies, 3), device=device)
    w_AA = torch.randn((num_bodies, 3), device=device)

    # Get twists in B in frame B
    v_BB, w_BB = math_utils.rigid_body_twist_transform(v_AA, w_AA, t_AB, q_AB)

    # Get back twists in A in frame A
    t_BA = -math_utils.quat_rotate_inverse(q_AB, t_AB)
    q_BA = math_utils.quat_conjugate(q_AB)
    v_AA_, w_AA_ = math_utils.rigid_body_twist_transform(v_BB, w_BB, t_BA, q_BA)

    # Check
    torch.testing.assert_close(v_AA_, v_AA)
    torch.testing.assert_close(w_AA_, w_AA)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_yaw_quat(device):
    """
    Test for yaw_quat methods.
    """
    # 90-degree (n/2 radians) rotations about the Y-axis
    quat_input = torch.tensor([0.7071, 0, 0.7071, 0], device=device)
    cloned_quat_input = quat_input.clone()

    # Calculated output that the function should return
    expected_output = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    # Compute the result using the existing implementation
    result = math_utils.yaw_quat(quat_input)

    # Verify original quat is not being modified
    torch.testing.assert_close(quat_input, cloned_quat_input)

    # check that the output is equivalent to the expected output
    torch.testing.assert_close(result, expected_output)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_quat_slerp(device):
    """Test quat_slerp function.

    This test checks the output from the :meth:`~isaaclab.utils.math_utils.quat_slerp` function against
    the output from :func:`scipy.spatial.transform.Slerp`.
    """
    # Generate 100 random rotation matrices
    random_rotation_matrices_1 = [math_utils.generate_random_rotation() for _ in range(100)]
    random_rotation_matrices_2 = [math_utils.generate_random_rotation() for _ in range(100)]

    tau_values = np.random.rand(10)  # Random values in the range [0, 1]

    for rmat1, rmat2 in zip(random_rotation_matrices_1, random_rotation_matrices_2):
        # Convert the rotation matrices to quaternions
        q1 = scipy_tf.Rotation.from_matrix(rmat1).as_quat()  # (x, y, z, w)
        q2 = scipy_tf.Rotation.from_matrix(rmat2).as_quat()  # (x, y, z, w)

        # Compute expected results using scipy's Slerp
        key_rots = scipy_tf.Rotation.from_quat(np.array([q1, q2]))
        key_times = [0, 1]
        slerp = scipy_tf.Slerp(key_times, key_rots)

        for tau in tau_values:
            expected = slerp(tau).as_quat()  # (x, y, z, w)
            result = math_utils.quat_slerp(torch.tensor(q1, device=device), torch.tensor(q2, device=device), tau)
            # Assert that the result is almost equal to the expected quaternion
            np.testing.assert_array_almost_equal(result.cpu(), expected, decimal=DECIMAL_PRECISION)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_matrix_from_quat(device):
    """test matrix_from_quat against scipy."""
    # prepare random quaternions and vectors
    n = 1024
    q_rand = math_utils.random_orientation(num=n, device=device)
    rot_mat = math_utils.matrix_from_quat(quaternions=q_rand)
    rot_mat_scipy = torch.tensor(
        scipy_tf.Rotation.from_quat(math_utils.convert_quat(quat=q_rand.to(device="cpu"), to="xyzw")).as_matrix(),
        device=device,
        dtype=torch.float32,
    )
    print()
    torch.testing.assert_close(rot_mat_scipy.to(device=device), rot_mat)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_quat_apply(device):
    """Test for quat_apply against scipy."""
    # prepare random quaternions and vectors
    n = 1024
    q_rand = math_utils.random_orientation(num=n, device=device)
    Rotation = scipy_tf.Rotation.from_quat(math_utils.convert_quat(quat=q_rand.to(device="cpu").numpy(), to="xyzw"))

    v_rand = math_utils.sample_uniform(-1000, 1000, (n, 3), device=device)

    # compute the result using the new implementation
    scipy_result = torch.tensor(Rotation.apply(v_rand.to(device="cpu").numpy()), device=device, dtype=torch.float)
    apply_result = math_utils.quat_apply(q_rand, v_rand)
    torch.testing.assert_close(scipy_result.to(device=device), apply_result, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_quat_apply_inverse(device):
    """Test for quat_apply against scipy."""

    # prepare random quaternions and vectors
    n = 1024
    q_rand = math_utils.random_orientation(num=n, device=device)
    Rotation = scipy_tf.Rotation.from_quat(math_utils.convert_quat(quat=q_rand.to(device="cpu").numpy(), to="xyzw"))

    v_rand = math_utils.sample_uniform(-1000, 1000, (n, 3), device=device)

    # compute the result using the new implementation
    scipy_result = torch.tensor(
        Rotation.apply(v_rand.to(device="cpu").numpy(), inverse=True), device=device, dtype=torch.float
    )
    apply_result = math_utils.quat_apply_inverse(q_rand, v_rand)
    torch.testing.assert_close(scipy_result.to(device=device), apply_result, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_quat_inv(device):
    """Test for quat_inv method.

    For random unit and non-unit quaternions q, the Hamilton products
    q ⊗ q⁻¹ and q⁻¹ ⊗ q must both equal the identity quaternion (1,0,0,0)
    within numerical precision.
    """
    num = 2048

    # -------- non-unit sample (average ‖q‖ ≈ 10) --------
    q_nonunit = torch.randn(num, 4, device=device) * 5.0

    # -------- unit sample (‖q‖ = 1) --------
    q_unit = torch.randn(num, 4, device=device)
    q_unit = q_unit / q_unit.norm(dim=-1, keepdim=True)

    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    for q in (q_nonunit, q_unit):
        q_inv = math_utils.quat_inv(q)

        id_batch = identity.expand_as(q)

        # left and right products must both be identity
        torch.testing.assert_close(math_utils.quat_mul(q, q_inv), id_batch, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(math_utils.quat_mul(q_inv, q), id_batch, atol=1e-4, rtol=1e-4)


def test_quat_apply_benchmarks():
    """Test for quat_apply and quat_apply_inverse methods compared to old methods using torch.bmm and torch.einsum.
    The new implementation uses :meth:`torch.einsum` instead of `torch.bmm` which allows
    for more flexibility in the input dimensions and is faster than `torch.bmm`.
    """

    # define old implementation for quat_rotate and quat_rotate_inverse
    # Based on commit: cdfa954fcc4394ca8daf432f61994e25a7b8e9e2

    @torch.jit.script
    def bmm_quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        shape = q.shape
        q_w = q[:, 0]
        q_vec = q[:, 1:]
        a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
        return a + b + c

    @torch.jit.script
    def bmm_quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        shape = q.shape
        q_w = q[:, 0]
        q_vec = q[:, 1:]
        a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c

    @torch.jit.script
    def einsum_quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_w = q[..., 0]
        q_vec = q[..., 1:]
        a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
        return a + b + c

    @torch.jit.script
    def einsum_quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_w = q[..., 0]
        q_vec = q[..., 1:]
        a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
        return a - b + c

    # check that implementation produces the same result as the new implementation
    for device in ["cpu", "cuda:0"]:
        # prepare random quaternions and vectors
        q_rand = math_utils.random_orientation(num=1024, device=device)
        v_rand = math_utils.sample_uniform(-1000, 1000, (1024, 3), device=device)

        # compute the result using the old implementation
        bmm_result = bmm_quat_rotate(q_rand, v_rand)
        bmm_result_inv = bmm_quat_rotate_inverse(q_rand, v_rand)

        # compute the result using the old implementation
        einsum_result = einsum_quat_rotate(q_rand, v_rand)
        einsum_result_inv = einsum_quat_rotate_inverse(q_rand, v_rand)

        # compute the result using the new implementation
        new_result = math_utils.quat_apply(q_rand, v_rand)
        new_result_inv = math_utils.quat_apply_inverse(q_rand, v_rand)

        # check that the result is close to the expected value
        torch.testing.assert_close(bmm_result, new_result, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(bmm_result_inv, new_result_inv, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(einsum_result, new_result, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(einsum_result_inv, new_result_inv, atol=1e-3, rtol=1e-3)

    # check the performance of the new implementation
    for device in ["cpu", "cuda:0"]:
        # prepare random quaternions and vectors
        # new implementation supports batched inputs
        q_shape = (1024, 2, 5, 4)
        v_shape = (1024, 2, 5, 3)
        # sample random quaternions and vectors
        num_quats = math.prod(q_shape[:-1])
        q_rand = math_utils.random_orientation(num=num_quats, device=device).reshape(q_shape)
        v_rand = math_utils.sample_uniform(-1000, 1000, v_shape, device=device)

        # create functions to test
        def iter_quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            """Iterative implementation of new quat_apply."""
            out = torch.empty_like(v)
            for i in range(q.shape[1]):
                for j in range(q.shape[2]):
                    out[:, i, j] = math_utils.quat_apply(q_rand[:, i, j], v_rand[:, i, j])
            return out

        def iter_quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            """Iterative implementation of new quat_apply_inverse."""
            out = torch.empty_like(v)
            for i in range(q.shape[1]):
                for j in range(q.shape[2]):
                    out[:, i, j] = math_utils.quat_apply_inverse(q_rand[:, i, j], v_rand[:, i, j])
            return out

        def iter_bmm_quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            """Iterative implementation of old quat_rotate using torch.bmm."""
            out = torch.empty_like(v)
            for i in range(q.shape[1]):
                for j in range(q.shape[2]):
                    out[:, i, j] = bmm_quat_rotate(q_rand[:, i, j], v_rand[:, i, j])
            return out

        def iter_bmm_quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            """Iterative implementation of old quat_rotate_inverse using torch.bmm."""
            out = torch.empty_like(v)
            for i in range(q.shape[1]):
                for j in range(q.shape[2]):
                    out[:, i, j] = bmm_quat_rotate_inverse(q_rand[:, i, j], v_rand[:, i, j])
            return out

        def iter_einsum_quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            """Iterative implementation of old quat_rotate using torch.einsum."""
            out = torch.empty_like(v)
            for i in range(q.shape[1]):
                for j in range(q.shape[2]):
                    out[:, i, j] = einsum_quat_rotate(q_rand[:, i, j], v_rand[:, i, j])
            return out

        def iter_einsum_quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            """Iterative implementation of old quat_rotate_inverse using torch.einsum."""
            out = torch.empty_like(v)
            for i in range(q.shape[1]):
                for j in range(q.shape[2]):
                    out[:, i, j] = einsum_quat_rotate_inverse(q_rand[:, i, j], v_rand[:, i, j])
            return out

        # benchmarks for iterative calls
        timer_iter_quat_apply = benchmark.Timer(
            stmt="iter_quat_apply(q_rand, v_rand)",
            globals={"iter_quat_apply": iter_quat_apply, "q_rand": q_rand, "v_rand": v_rand},
        )
        timer_iter_quat_apply_inverse = benchmark.Timer(
            stmt="iter_quat_apply_inverse(q_rand, v_rand)",
            globals={"iter_quat_apply_inverse": iter_quat_apply_inverse, "q_rand": q_rand, "v_rand": v_rand},
        )

        timer_iter_bmm_quat_rotate = benchmark.Timer(
            stmt="iter_bmm_quat_rotate(q_rand, v_rand)",
            globals={"iter_bmm_quat_rotate": iter_bmm_quat_rotate, "q_rand": q_rand, "v_rand": v_rand},
        )
        timer_iter_bmm_quat_rotate_inverse = benchmark.Timer(
            stmt="iter_bmm_quat_rotate_inverse(q_rand, v_rand)",
            globals={
                "iter_bmm_quat_rotate_inverse": iter_bmm_quat_rotate_inverse,
                "q_rand": q_rand,
                "v_rand": v_rand,
            },
        )

        timer_iter_einsum_quat_rotate = benchmark.Timer(
            stmt="iter_einsum_quat_rotate(q_rand, v_rand)",
            globals={"iter_einsum_quat_rotate": iter_einsum_quat_rotate, "q_rand": q_rand, "v_rand": v_rand},
        )
        timer_iter_einsum_quat_rotate_inverse = benchmark.Timer(
            stmt="iter_einsum_quat_rotate_inverse(q_rand, v_rand)",
            globals={
                "iter_einsum_quat_rotate_inverse": iter_einsum_quat_rotate_inverse,
                "q_rand": q_rand,
                "v_rand": v_rand,
            },
        )

        # create benchmaks for size independent calls
        timer_quat_apply = benchmark.Timer(
            stmt="math_utils.quat_apply(q_rand, v_rand)",
            globals={"math_utils": math_utils, "q_rand": q_rand, "v_rand": v_rand},
        )
        timer_quat_apply_inverse = benchmark.Timer(
            stmt="math_utils.quat_apply_inverse(q_rand, v_rand)",
            globals={"math_utils": math_utils, "q_rand": q_rand, "v_rand": v_rand},
        )
        timer_einsum_quat_rotate = benchmark.Timer(
            stmt="einsum_quat_rotate(q_rand, v_rand)",
            globals={"einsum_quat_rotate": einsum_quat_rotate, "q_rand": q_rand, "v_rand": v_rand},
        )
        timer_einsum_quat_rotate_inverse = benchmark.Timer(
            stmt="einsum_quat_rotate_inverse(q_rand, v_rand)",
            globals={"einsum_quat_rotate_inverse": einsum_quat_rotate_inverse, "q_rand": q_rand, "v_rand": v_rand},
        )

        # run the benchmark
        print("--------------------------------")
        print(f"Device: {device}")
        print("Time for quat_apply:", timer_quat_apply.timeit(number=1000))
        print("Time for einsum_quat_rotate:", timer_einsum_quat_rotate.timeit(number=1000))
        print("Time for iter_quat_apply:", timer_iter_quat_apply.timeit(number=1000))
        print("Time for iter_bmm_quat_rotate:", timer_iter_bmm_quat_rotate.timeit(number=1000))
        print("Time for iter_einsum_quat_rotate:", timer_iter_einsum_quat_rotate.timeit(number=1000))
        print("--------------------------------")
        print("Time for quat_apply_inverse:", timer_quat_apply_inverse.timeit(number=1000))
        print("Time for einsum_quat_rotate_inverse:", timer_einsum_quat_rotate_inverse.timeit(number=1000))
        print("Time for iter_quat_apply_inverse:", timer_iter_quat_apply_inverse.timeit(number=1000))
        print("Time for iter_bmm_quat_rotate_inverse:", timer_iter_bmm_quat_rotate_inverse.timeit(number=1000))
        print("Time for iter_einsum_quat_rotate_inverse:", timer_iter_einsum_quat_rotate_inverse.timeit(number=1000))
        print("--------------------------------")

        # check output values are the same
        torch.testing.assert_close(math_utils.quat_apply(q_rand, v_rand), iter_quat_apply(q_rand, v_rand))
        torch.testing.assert_close(
            math_utils.quat_apply(q_rand, v_rand), iter_bmm_quat_rotate(q_rand, v_rand), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            math_utils.quat_apply_inverse(q_rand, v_rand), iter_quat_apply_inverse(q_rand, v_rand)
        )
        torch.testing.assert_close(
            math_utils.quat_apply_inverse(q_rand, v_rand),
            iter_bmm_quat_rotate_inverse(q_rand, v_rand),
            atol=1e-3,
            rtol=1e-3,
        )


def test_interpolate_rotations():
    """Test interpolate_rotations function.

    This test checks the output from the :meth:`~isaaclab.utils.math_utils.interpolate_rotations` function against
    the output from :func:`scipy.spatial.transform.Slerp`.
    """
    # Generate NUM_ITERS random rotation matrices
    random_rotation_matrices_1 = [math_utils.generate_random_rotation() for _ in range(100)]
    random_rotation_matrices_2 = [math_utils.generate_random_rotation() for _ in range(100)]

    for rmat1, rmat2 in zip(random_rotation_matrices_1, random_rotation_matrices_2):
        # Compute expected results using scipy's Slerp
        key_rots = scipy_tf.Rotation.from_matrix(np.array([rmat1, rmat2]))

        # Create a Slerp object and interpolate create the interpolated matrices
        # Minimum 2 required because Interpolate_rotations returns one extra rotation matrix
        num_steps = np.random.randint(2, 51)
        key_times = [0, 1]
        slerp = scipy_tf.Slerp(key_times, key_rots)
        interp_times = np.linspace(0, 1, num_steps)
        expected = slerp(interp_times).as_matrix()

        # Test 1:
        # Interpolate_rotations using interpolate_rotations and quat_slerp
        # interpolate_rotations returns one extra rotation matrix hence num_steps-1
        result_quat = math_utils.interpolate_rotations(rmat1, rmat2, num_steps - 1)

        # Assert that the result is almost equal to the expected quaternion
        np.testing.assert_array_almost_equal(result_quat.cpu(), expected, decimal=DECIMAL_PRECISION)

        # Test 2:
        # Interpolate_rotations using axis_angle and ensure the result is still the same
        # interpolate_rotations returns one extra rotation matrix hence num_steps-1
        result_axis_angle = math_utils.interpolate_rotations(rmat1, rmat2, num_steps - 1, axis_angle=True)

        # Assert that the result is almost equal to the expected quaternion
        np.testing.assert_array_almost_equal(result_axis_angle.cpu(), expected, decimal=DECIMAL_PRECISION)


def test_euler_xyz_from_quat():
    """Test euler_xyz_from_quat function.

    This test checks the output from the :meth:`~isaaclab.utils.math_utils.euler_xyz_from_quat` function
    against the expected output for various quaternions.
    The test includes quaternions representing different rotations around the x, y, and z axes.
    The test is performed for both the default output range (-π, π] and the wrapped output range [0, 2π).
    """
    quats = [
        torch.Tensor([[1.0, 0.0, 0.0, 0.0]]),  # 0° around x, y, z
        torch.Tensor([
            [0.9238795, 0.3826834, 0.0, 0.0],  # 45° around x
            [0.9238795, 0.0, -0.3826834, 0.0],  # -45° around y
            [0.9238795, 0.0, 0.0, -0.3826834],  # -45° around z
        ]),
        torch.Tensor([
            [0.7071068, -0.7071068, 0.0, 0.0],  # -90° around x
            [0.7071068, 0.0, 0.0, -0.7071068],  # -90° around z
        ]),
        torch.Tensor([
            [0.3826834, -0.9238795, 0.0, 0.0],  # -135° around x
            [0.3826834, 0.0, 0.0, -0.9238795],  # -135° around y
        ]),
    ]

    expected_euler_angles = [
        torch.Tensor([[0.0, 0.0, 0.0]]),  # identity
        torch.Tensor([
            [torch.pi / 4, 0.0, 0.0],  # 45° about x
            [0.0, -torch.pi / 4, 0.0],  # -45° about y
            [0.0, 0.0, -torch.pi / 4],  # -45° about z
        ]),
        torch.Tensor([
            [-torch.pi / 2, 0.0, 0.0],  # -90° about x
            [0.0, 0.0, -torch.pi / 2],  # -90° about z
        ]),
        torch.Tensor([
            [-3 * torch.pi / 4, 0.0, 0.0],  # -135° about x
            [0.0, 0.0, -3 * torch.pi / 4],  # -135° about y
        ]),
    ]

    # Test 1: default no-wrap range from (-π, π]
    for quat, expected in zip(quats, expected_euler_angles):
        output = torch.stack(math_utils.euler_xyz_from_quat(quat), dim=-1)
        torch.testing.assert_close(output, expected)

    # Test 2: wrap to [0, 2π)
    for quat, expected in zip(quats, expected_euler_angles):
        wrapped = expected % (2 * torch.pi)
        output = torch.stack(math_utils.euler_xyz_from_quat(quat, wrap_to_2pi=True), dim=-1)
        torch.testing.assert_close(output, wrapped)
