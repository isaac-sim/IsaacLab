# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import unittest

"""Launch Isaac Sim Simulator first.

This is only needed because of warp dependency.
"""

from omni.isaac.orbit.app import AppLauncher

# launch omniverse app in headless mode
app_launcher = AppLauncher(headless=True)

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


if __name__ == "__main__":
    unittest.main()
