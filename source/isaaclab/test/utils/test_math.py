# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest

"""Launch Isaac Sim Simulator first.

This is only needed because of warp dependency.
"""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app


"""Rest everything follows."""

import math
import numpy as np
import torch
import torch.utils.benchmark as benchmark
from math import pi as PI
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import isaaclab.utils.math as math_utils

# Number of iterations to run the batched tests
NUM_ITERS = 100
# This value is set to because "float operations are inexact".
# Details: https://github.com/pytorch/pytorch/issues/17678
DECIMAL_PRECISION = 5
ATOL = 10 ** (-DECIMAL_PRECISION)
RTOL = 10 ** (-DECIMAL_PRECISION)


class TestMathUtilities(unittest.TestCase):
    """Test fixture for checking math utilities in Isaac Lab."""

    def test_is_identity_pose(self):
        """Test is_identity_pose method."""
        identity_pos_one_row = torch.zeros(3)
        identity_rot_one_row = torch.tensor((1.0, 0.0, 0.0, 0.0))

        self.assertTrue(math_utils.is_identity_pose(identity_pos_one_row, identity_rot_one_row))

        identity_pos_one_row[0] = 1.0
        identity_rot_one_row[1] = 1.0

        self.assertFalse(math_utils.is_identity_pose(identity_pos_one_row, identity_rot_one_row))

        identity_pos_multi_row = torch.zeros(3, 3)
        identity_rot_multi_row = torch.zeros(3, 4)
        identity_rot_multi_row[:, 0] = 1.0

        self.assertTrue(math_utils.is_identity_pose(identity_pos_multi_row, identity_rot_multi_row))

        identity_pos_multi_row[0, 0] = 1.0
        identity_rot_multi_row[0, 1] = 1.0

        self.assertFalse(math_utils.is_identity_pose(identity_pos_multi_row, identity_rot_multi_row))

    def test_axis_angle_from_quat(self):
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
            with self.subTest(quat=quat, angle=angle):
                torch.testing.assert_close(math_utils.axis_angle_from_quat(quat), angle)

    def test_axis_angle_from_quat_approximation(self):
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

    def test_quat_error_magnitude(self):
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
            with self.subTest(q1=q1, q2=q2):
                # Compute the error
                q12_diff = math_utils.quat_error_magnitude(q1, q2)

                # Check that the error is close to the expected value
                if len(q1.shape) > 1:
                    torch.testing.assert_close(q12_diff, expected_diff)
                else:
                    self.assertAlmostEqual(q12_diff.item(), expected_diff.item(), places=5)

    def test_quat_unique(self):
        """Test quat_unique method."""
        # Define test cases
        quats = math_utils.random_orientation(num=1024, device="cpu")

        # Test positive real quaternion
        pos_real_quats = math_utils.quat_unique(quats)

        # Test that the real part is positive
        self.assertTrue(torch.all(pos_real_quats[:, 0] > 0).item())

        non_pos_indices = quats[:, 0] < 0
        # Check imaginary part have sign flipped if real part is negative
        torch.testing.assert_close(pos_real_quats[non_pos_indices], -quats[non_pos_indices])
        torch.testing.assert_close(pos_real_quats[~non_pos_indices], quats[~non_pos_indices])

    def test_quat_mul_with_quat_unique(self):
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

    def test_quat_error_mag_with_quat_unique(self):
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

    def test_convention_converter(self):
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

    def test_wrap_to_pi(self):
        """Test wrap_to_pi method."""
        # Define test cases
        # Each tuple contains: angle, expected wrapped angle
        test_cases = [
            # No wrapping needed
            (torch.Tensor([0.0]), torch.Tensor([0.0])),
            # Positive angle
            (torch.Tensor([PI]), torch.Tensor([PI])),
            # Negative angle
            (torch.Tensor([-PI]), torch.Tensor([-PI])),
            # Multiple angles
            (torch.Tensor([3 * PI, -3 * PI, 4 * PI, -4 * PI]), torch.Tensor([PI, -PI, 0.0, 0.0])),
            # Multiple angles from MATLAB docs
            # fmt: off
            (
                torch.Tensor([-2 * PI, - PI - 0.1, -PI, -2.8, 3.1, PI, PI + 0.001, PI + 1, 2 * PI, 2 * PI + 0.1]),
                torch.Tensor([0.0, PI - 0.1, -PI, -2.8, 3.1 , PI, -PI + 0.001, -PI + 1 , 0.0, 0.1])
            ),
            # fmt: on
        ]

        # Iterate over test cases
        for device in ["cpu", "cuda:0"]:
            for angle, expected_angle in test_cases:
                with self.subTest(angle=angle, device=device):
                    # move to the device
                    angle = angle.to(device)
                    expected_angle = expected_angle.to(device)
                    # Compute the wrapped angle
                    wrapped_angle = math_utils.wrap_to_pi(angle)
                    # Check that the wrapped angle is close to the expected value
                    torch.testing.assert_close(wrapped_angle, expected_angle)

    def test_quat_rotate_and_quat_rotate_inverse(self):
        """Test for quat_rotate and quat_rotate_inverse methods.

        The new implementation uses :meth:`torch.einsum` instead of `torch.bmm` which allows
        for more flexibility in the input dimensions and is faster than `torch.bmm`.
        """

        # define old implementation for quat_rotate and quat_rotate_inverse
        # Based on commit: cdfa954fcc4394ca8daf432f61994e25a7b8e9e2

        @torch.jit.script
        def old_quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            shape = q.shape
            q_w = q[:, 0]
            q_vec = q[:, 1:]
            a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
            b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
            c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
            return a + b + c

        @torch.jit.script
        def old_quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            shape = q.shape
            q_w = q[:, 0]
            q_vec = q[:, 1:]
            a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
            b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
            c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
            return a - b + c

        # check that implementation produces the same result as the new implementation
        for device in ["cpu", "cuda:0"]:
            # prepare random quaternions and vectors
            q_rand = math_utils.random_orientation(num=1024, device=device)
            v_rand = math_utils.sample_uniform(-1000, 1000, (1024, 3), device=device)

            # compute the result using the old implementation
            old_result = old_quat_rotate(q_rand, v_rand)
            old_result_inv = old_quat_rotate_inverse(q_rand, v_rand)

            # compute the result using the new implementation
            new_result = math_utils.quat_rotate(q_rand, v_rand)
            new_result_inv = math_utils.quat_rotate_inverse(q_rand, v_rand)

            # check that the result is close to the expected value
            torch.testing.assert_close(old_result, new_result)
            torch.testing.assert_close(old_result_inv, new_result_inv)

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
            def iter_quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                """Iterative implementation of new quat_rotate."""
                out = torch.empty_like(v)
                for i in range(q.shape[1]):
                    for j in range(q.shape[2]):
                        out[:, i, j] = math_utils.quat_rotate(q_rand[:, i, j], v_rand[:, i, j])
                return out

            def iter_quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                """Iterative implementation of new quat_rotate_inverse."""
                out = torch.empty_like(v)
                for i in range(q.shape[1]):
                    for j in range(q.shape[2]):
                        out[:, i, j] = math_utils.quat_rotate_inverse(q_rand[:, i, j], v_rand[:, i, j])
                return out

            def iter_old_quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                """Iterative implementation of old quat_rotate."""
                out = torch.empty_like(v)
                for i in range(q.shape[1]):
                    for j in range(q.shape[2]):
                        out[:, i, j] = old_quat_rotate(q_rand[:, i, j], v_rand[:, i, j])
                return out

            def iter_old_quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                """Iterative implementation of old quat_rotate_inverse."""
                out = torch.empty_like(v)
                for i in range(q.shape[1]):
                    for j in range(q.shape[2]):
                        out[:, i, j] = old_quat_rotate_inverse(q_rand[:, i, j], v_rand[:, i, j])
                return out

            # create benchmark
            timer_iter_quat_rotate = benchmark.Timer(
                stmt="iter_quat_rotate(q_rand, v_rand)",
                globals={"iter_quat_rotate": iter_quat_rotate, "q_rand": q_rand, "v_rand": v_rand},
            )
            timer_iter_quat_rotate_inverse = benchmark.Timer(
                stmt="iter_quat_rotate_inverse(q_rand, v_rand)",
                globals={"iter_quat_rotate_inverse": iter_quat_rotate_inverse, "q_rand": q_rand, "v_rand": v_rand},
            )

            timer_iter_old_quat_rotate = benchmark.Timer(
                stmt="iter_old_quat_rotate(q_rand, v_rand)",
                globals={"iter_old_quat_rotate": iter_old_quat_rotate, "q_rand": q_rand, "v_rand": v_rand},
            )
            timer_iter_old_quat_rotate_inverse = benchmark.Timer(
                stmt="iter_old_quat_rotate_inverse(q_rand, v_rand)",
                globals={
                    "iter_old_quat_rotate_inverse": iter_old_quat_rotate_inverse,
                    "q_rand": q_rand,
                    "v_rand": v_rand,
                },
            )

            timer_quat_rotate = benchmark.Timer(
                stmt="math_utils.quat_rotate(q_rand, v_rand)",
                globals={"math_utils": math_utils, "q_rand": q_rand, "v_rand": v_rand},
            )
            timer_quat_rotate_inverse = benchmark.Timer(
                stmt="math_utils.quat_rotate_inverse(q_rand, v_rand)",
                globals={"math_utils": math_utils, "q_rand": q_rand, "v_rand": v_rand},
            )

            # run the benchmark
            print("--------------------------------")
            print(f"Device: {device}")
            print("Time for quat_rotate:", timer_quat_rotate.timeit(number=1000))
            print("Time for iter_quat_rotate:", timer_iter_quat_rotate.timeit(number=1000))
            print("Time for iter_old_quat_rotate:", timer_iter_old_quat_rotate.timeit(number=1000))
            print("--------------------------------")
            print("Time for quat_rotate_inverse:", timer_quat_rotate_inverse.timeit(number=1000))
            print("Time for iter_quat_rotate_inverse:", timer_iter_quat_rotate_inverse.timeit(number=1000))
            print("Time for iter_old_quat_rotate_inverse:", timer_iter_old_quat_rotate_inverse.timeit(number=1000))
            print("--------------------------------")

            # check output values are the same
            torch.testing.assert_close(math_utils.quat_rotate(q_rand, v_rand), iter_quat_rotate(q_rand, v_rand))
            torch.testing.assert_close(math_utils.quat_rotate(q_rand, v_rand), iter_old_quat_rotate(q_rand, v_rand))
            torch.testing.assert_close(
                math_utils.quat_rotate_inverse(q_rand, v_rand), iter_quat_rotate_inverse(q_rand, v_rand)
            )
            torch.testing.assert_close(
                math_utils.quat_rotate_inverse(q_rand, v_rand),
                iter_old_quat_rotate_inverse(q_rand, v_rand),
            )

    def test_orthogonalize_perspective_depth(self):
        """Test for converting perspective depth to orthogonal depth."""
        for device in ["cpu", "cuda:0"]:
            # Create a sample perspective depth image (N, H, W)
            perspective_depth = torch.tensor(
                [[[10.0, 0.0, 100.0], [0.0, 3000.0, 0.0], [100.0, 0.0, 100.0]]], device=device
            )

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

    """
    Tests for math_utils.pose_inv function
    This class checks the pose_inv function's output against the np.linalg.inv function's output.
    1. test_single_numpy_comparison: Checks if the inverse of a random transformation matrix
    matches NumPy's built-in inverse.
    2. test_multi_numpy_comparison: Verifies the same for a batch of NUM_ITERS random matrices.
    """

    def test_single_numpy_comparison(self):
        for _ in range(NUM_ITERS):
            test_mat = math_utils.generate_random_transformation_matrix(pos_boundary=10, rot_boundary=(2 * np.pi))
            result = np.array(math_utils.pose_inv(test_mat))
            expected = np.linalg.inv(np.array(test_mat))
            np.testing.assert_array_almost_equal(result, expected, decimal=DECIMAL_PRECISION)

    def test_multi_numpy_comparison(self):
        # Generate NUM_ITERS random transformation matrices
        test_mats = torch.stack([
            math_utils.generate_random_transformation_matrix(pos_boundary=10, rot_boundary=(2 * math.pi))
            for _ in range(NUM_ITERS)
        ])
        result = np.array(math_utils.pose_inv(test_mats))
        expected = np.linalg.inv(np.array(test_mats))
        np.testing.assert_array_almost_equal(result, expected, decimal=DECIMAL_PRECISION)

    """
    Tests for math_utils.quat_slerp function.
    This class checks the quat_slerp function's output against the output from scipy.spatial.transform.Slerp.
    1. test_quat_slerp_multi_scipy_comparison: Generates 20x2 random rotation matrices, find the interpolation rotation
        and compares it to the output of scipy.spatial.transform.Slerp.
    """

    def test_quat_slerp_multi_scipy_comparison(self):
        # Generate NUM_ITERS random rotation matrices
        random_rotation_matrices_1 = [math_utils.generate_random_rotation() for _ in range(NUM_ITERS)]
        random_rotation_matrices_2 = [math_utils.generate_random_rotation() for _ in range(NUM_ITERS)]

        tau_values = np.random.rand(10)  # Random values in the range [0, 1]

        for rmat1, rmat2 in zip(random_rotation_matrices_1, random_rotation_matrices_2):
            # Convert the rotation matrices to quaternions
            q1 = R.from_matrix(rmat1).as_quat()  # (x, y, z, w)
            q2 = R.from_matrix(rmat2).as_quat()  # (x, y, z, w)

            # Compute expected results using scipy's Slerp
            key_rots = R.from_quat(np.array([q1, q2]))
            key_times = [0, 1]
            slerp = Slerp(key_times, key_rots)

            for tau in tau_values:
                expected = slerp(tau).as_quat()  # (x, y, z, w)
                result = math_utils.quat_slerp(torch.tensor(q1), torch.tensor(q2), tau)
                # Assert that the result is almost equal to the expected quaternion
                np.testing.assert_array_almost_equal(result, expected, decimal=DECIMAL_PRECISION)

    """
    Tests for math_utils.interpolate_rotations function.
    This class checks the interpolate_rotations function's output against the output from scipy.spatial.transform.Slerp.
    1. test_interpolate_rotations_multi_scipy_comparison: Generates 20x2 random rotation matrices, finds an array of interpolated rotations
        and compares it to the output of scipy.spatial.transform.Slerp.
    """

    def test_interpolate_rotations_multi_scipy_comparison(self):
        # Generate NUM_ITERS random rotation matrices
        random_rotation_matrices_1 = [math_utils.generate_random_rotation() for _ in range(NUM_ITERS)]
        random_rotation_matrices_2 = [math_utils.generate_random_rotation() for _ in range(NUM_ITERS)]

        for rmat1, rmat2 in zip(random_rotation_matrices_1, random_rotation_matrices_2):
            # Compute expected results using scipy's Slerp
            key_rots = R.from_matrix(np.array([rmat1, rmat2]))

            # Create a Slerp object and interpolate create the interpolated matrices
            # Minimum 2 required because Interpolate_rotations returns one extra rotation matrix
            num_steps = np.random.randint(2, 51)
            key_times = [0, 1]
            slerp = Slerp(key_times, key_rots)
            interp_times = np.linspace(0, 1, num_steps)
            expected = slerp(interp_times).as_matrix()

            # Test 1:
            # Interpolate_rotations using interpolate_rotations and quat_slerp
            # interpolate_rotations returns one extra rotation matrix hence num_steps-1
            result_quat = math_utils.interpolate_rotations(rmat1, rmat2, num_steps - 1)

            # Assert that the result is almost equal to the expected quaternion
            np.testing.assert_array_almost_equal(result_quat, expected, decimal=DECIMAL_PRECISION)

            # Test 2:
            # Interpolate_rotations using axis_angle and ensure the result is still the same
            # interpolate_rotations returns one extra rotation matrix hence num_steps-1
            result_axis_angle = math_utils.interpolate_rotations(rmat1, rmat2, num_steps - 1, axis_angle=True)

            # Assert that the result is almost equal to the expected quaternion
            np.testing.assert_array_almost_equal(result_axis_angle, expected, decimal=DECIMAL_PRECISION)

    """
    Tests for math_utils.interpolate_poses function.
    This class checks the interpolate_poses function's output against the output from scipy.spatial.transform.Slerp.
    1. test_interpolate_poses_multi_scipy_comparison: Generates 20x2 random transformation matrices,
        computes an array of interpolated transformations
        and compares it to the output of scipy.spatial.transform.Slerp and np.linspace
    """

    def test_interpolate_poses_multi_scipy_comparison(self):
        # Generate NUM_ITERS random transformation matrices
        random_mat_1 = [math_utils.generate_random_transformation_matrix() for _ in range(NUM_ITERS)]
        random_mat_2 = [math_utils.generate_random_transformation_matrix() for _ in range(NUM_ITERS)]

        for mat1, mat2 in zip(random_mat_1, random_mat_2):
            pos_1, rmat1 = math_utils.unmake_pose(mat1)
            pos_2, rmat2 = math_utils.unmake_pose(mat2)

            # Compute expected results using scipy's Slerp
            key_rots = R.from_matrix(np.array([rmat1, rmat2]))

            # Create a Slerp object and interpolate create the interpolated rotation matrices
            # Minimum 3 required because interpolate_poses returns extra staring and ending pose matrices
            num_steps = np.random.randint(3, 51)
            key_times = [0, 1]
            slerp = Slerp(key_times, key_rots)
            interp_times = np.linspace(0, 1, num_steps)
            expected_quat = slerp(interp_times).as_matrix()

            # Test interpolation against expected result using np.linspace
            expected_pos = np.linspace(pos_1, pos_2, num_steps)

            # interpolate_poses using interpolate_poses and quat_slerp
            # interpolate_poses returns extra staring and ending pose matrices hence num_steps-2
            interpolated_poses, _ = math_utils.interpolate_poses(
                math_utils.make_pose(pos_1, rmat1), math_utils.make_pose(pos_2, rmat2), num_steps - 2
            )
            result_pos, result_quat = math_utils.unmake_pose(interpolated_poses)

            # Assert that the result is almost equal to the expected quaternion
            np.testing.assert_array_almost_equal(result_quat, expected_quat, decimal=DECIMAL_PRECISION)
            np.testing.assert_array_almost_equal(result_pos, expected_pos, decimal=DECIMAL_PRECISION)


if __name__ == "__main__":
    run_tests()
