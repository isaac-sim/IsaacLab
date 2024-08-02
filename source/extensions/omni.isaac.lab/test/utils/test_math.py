# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import unittest

"""Launch Isaac Sim Simulator first.

This is only needed because of warp dependency.
"""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app


"""Rest everything follows."""

from math import pi as PI

import omni.isaac.lab.utils.math as math_utils


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


if __name__ == "__main__":
    run_tests()
