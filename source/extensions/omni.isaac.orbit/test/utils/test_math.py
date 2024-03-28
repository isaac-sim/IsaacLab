# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import unittest
from math import pi as PI

"""Launch Isaac Sim Simulator first.

This is only needed because of warp dependency.
"""

from omni.isaac.orbit.app import AppLauncher, run_tests

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app


import omni.isaac.orbit.utils.math as math_utils


class TestMathUtilities(unittest.TestCase):
    """Test fixture for checking math utilities in Orbit."""

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
                self.assertTrue(torch.allclose(math_utils.axis_angle_from_quat(quat), angle, atol=1e-7))

    def test_axis_angle_from_quat_approximation(self):
        """Test Taylor approximation from axis_angle_from_quat method
        for unstable conversions where theta is very small."""
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
        self.assertTrue(torch.allclose(axis_angle_computed, axis_angle_expected, atol=1e-7))

    def test_quat_error_magnitude(self):
        # Define test cases
        test_cases = [
            # q1, q2, expected error
            # No rotation
            (torch.Tensor([1, 0, 0, 0]), torch.Tensor([1, 0, 0, 0]), torch.Tensor([0.0])),
            # PI/2 rotation
            (torch.Tensor([1.0, 0, 0.0, 0]), torch.Tensor([0.7071068, 0.7071068, 0, 0]), torch.Tensor([PI / 2])),
            # PI rotation
            (torch.Tensor([1.0, 0, 0.0, 0]), torch.Tensor([0.0, 0.0, 1.0, 0]), torch.Tensor([PI])),
        ]
        # Test higher dimension
        test_cases += tuple([(torch.stack(tensors) for tensors in zip(*test_cases))])
        for q1, q2, expected_diff in test_cases:
            with self.subTest(q1=q1, q2=q2):
                q12_diff = math_utils.quat_error_magnitude(q1, q2)
                self.assertTrue(torch.allclose(q12_diff, torch.flatten(expected_diff), atol=1e-7))


if __name__ == "__main__":
    run_tests()
