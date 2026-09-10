# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Unit test for testing utility functions of the trail library.

.. code-block:: bash
    # Usage
    ./isaaclab.sh -p -m pytest source/isaaclab_contrib/test/terrains/test_trail.py
"""

import numpy as np

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import unittest

import torch
from isaaclab_contrib.terrains.trail.utils import colors, math, numpy_arrays, transformations


class TestTrailUtilities(unittest.TestCase):
    """Test fixture for checking trail utilities."""

    def test_mirror_and_join(self):
        """Test mirror_and_join method."""
        xi = [0.0, 0.1, 0.2, 0.3]
        yi = [-0.5, -0.2, 1.2, 1.0]
        zi = [1.0, 2.0, 3.0, 4.0]
        object = np.stack([xi, yi, zi], axis=1)
        offset = 1.0
        mirrored_object = numpy_arrays.mirror_and_join(object=object, dim=0, dim_flip=-1, offset=offset)
        # check x (mirrored axis)
        for dim_x in range(len(xi)):
            self.assertEqual(mirrored_object[4 + dim_x, 0], xi[3] + offset + dim_x * 0.1)
        # check y (should remain constant)
        for dim_y in range(len(yi)):
            self.assertEqual(mirrored_object[4 + dim_y, 1], yi[dim_y])
        # check z (should be flipped)
        for dim_z in range(len(zi)):
            self.assertEqual(mirrored_object[4 + dim_z, 2], zi[3 - dim_z])

    def test_get_bounding_box(self):
        """Test get_bounding_box method."""
        xi = [0.1, 0.5, 0.8, 10.0]
        yi = [100.0, -99.0, -5.7, 170.0]
        zi = [-0.2, 1.0, 1.0, 0.0]
        object = np.stack([xi, yi, zi], axis=1)
        bounding_box = numpy_arrays.get_bounding_box(object=object)
        # check x
        self.assertEqual(bounding_box[0, 0], 0.1)
        self.assertEqual(bounding_box[1, 0], 10.0)
        # check y
        self.assertEqual(bounding_box[0, 1], -99.0)
        self.assertEqual(bounding_box[1, 1], 170.0)
        # check z
        self.assertEqual(bounding_box[0, 2], -0.2)
        self.assertEqual(bounding_box[1, 2], 1.0)

    def test_decay_at_boundaries(self):
        """Test decay_at_boundaries method."""
        xi = np.linspace(0.0, 10.0, 11)
        yi = xi * 0.0 + 1.0
        zi = xi * 0.0 + 1.0
        object = np.stack([xi, yi, zi], axis=1)
        numpy_arrays.decay_at_boundaries(object=object, vec=None, dim=1, threshold=2.0)
        numpy_arrays.decay_at_boundaries(object=object, vec=None, dim=2, threshold=3.0)

        # check y
        self.assertAlmostEqual(object[0, 1], 0.0, places=5)
        self.assertAlmostEqual(object[1, 1], 0.5, places=5)
        for n in range(7):
            self.assertAlmostEqual(object[2 + n, 1], 1.0, places=5)
        self.assertAlmostEqual(object[9, 1], 0.5, places=5)
        self.assertAlmostEqual(object[10, 1], 0.0, places=5)

        # check z
        self.assertAlmostEqual(object[0, 2], 0.0, places=5)
        self.assertAlmostEqual(object[1, 2], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(object[2, 2], 2.0 / 3.0, places=5)
        for n in range(5):
            self.assertAlmostEqual(object[3 + n, 1], 1.0, places=5)
        self.assertAlmostEqual(object[8, 2], 2.0 / 3.0, places=5)
        self.assertAlmostEqual(object[9, 2], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(object[10, 2], 0.0, places=5)

    def test_transformation(self):
        """Test transformation methods."""
        vec = [0.1, 0.3, -0.8, 1.0]

        # check translation
        T = transformations.translation(vec=[0.5, 0.4, 0.3])
        vecT = T.dot(vec)
        self.assertAlmostEqual(vecT[0], 0.6, places=5)
        self.assertAlmostEqual(vecT[1], 0.7, places=5)
        self.assertAlmostEqual(vecT[2], -0.5, places=5)

        # check roll
        T = transformations.roll(angle=np.pi)
        vecT = T.dot(vec)
        self.assertAlmostEqual(vecT[0], vec[0], places=5)
        self.assertAlmostEqual(vecT[1], -vec[1], places=5)
        self.assertAlmostEqual(vecT[2], -vec[2], places=5)

        # check translate and roll
        T = transformations.translate_and_roll(vec=[0.5, 0.4, 0.3], angle=np.pi)
        vecT = T.dot(vec)
        self.assertAlmostEqual(vecT[0], 0.6, places=5)
        self.assertAlmostEqual(vecT[1], -vec[1] + 0.4, places=5)
        self.assertAlmostEqual(vecT[2], -vec[2] + 0.3, places=5)

    def test_interp(self):
        """Test interp method."""
        # check two floats
        self.assertAlmostEqual(math.interp(param0=1.2, param1=1.4, x=0.0), 1.2, places=5)
        self.assertAlmostEqual(math.interp(param0=1.2, param1=1.4, x=0.5), 1.3, places=5)
        self.assertAlmostEqual(math.interp(param0=1.2, param1=1.4, x=1.0), 1.4, places=5)

        # check tuple[float] and float
        value = math.interp(param0=(1.0, 2.0), param1=3.0, x=0.5)
        self.assertAlmostEqual(value[0], 2.0, places=5)
        self.assertAlmostEqual(value[1], 2.5, places=5)

        # check two float and tuples
        value = math.interp(param0=1.0, param1=(2.0, 3.0), x=0.5)
        self.assertAlmostEqual(value[0], 1.5, places=5)
        self.assertAlmostEqual(value[1], 2.0, places=5)

        # check two tuples
        value = math.interp(param0=(1.0, 2.0), param1=(2.0, 3.0), x=0.5)
        self.assertAlmostEqual(value[0], 1.5, places=5)
        self.assertAlmostEqual(value[1], 2.5, places=5)

    def test_sample(self):
        """Test sample method."""
        # check float
        self.assertEqual(math.sample(1.2), 1.2)
        # check tuple[float]
        self.assertEqual(math.sample((1.2, 1.2)), 1.2)
        # check int
        self.assertEqual(math.sample(1), 1)
        # check tuple[int]
        self.assertEqual(math.sample((1, 1)), 1)

    def test_in_limits(self):
        """Test in_limits method."""
        # check float
        self.assertTrue(math.in_limits(value=1.5, limits=(1.0, 2.0)))
        self.assertFalse(math.in_limits(value=1.5, limits=(1.0, 1.4)))
        self.assertTrue(math.in_limits(value=1.5, limits=1.5))
        self.assertFalse(math.in_limits(value=1.5, limits=1.4))
        # check int
        self.assertTrue(math.in_limits(value=7, limits=(-5, 8)))
        self.assertFalse(math.in_limits(value=7, limits=(9, 10)))
        self.assertTrue(math.in_limits(value=7, limits=7))
        self.assertFalse(math.in_limits(value=7, limits=-7))

    def test_sample_sign(self):
        """Test sample_sign method."""
        self.assertEqual(abs(math.sample_sign()), 1.0)

    def test_rgb_hsv_conversion(self):
        """Test rgb_to_hsv and hsv_to_rgb with known values and round-trip checks."""
        rgb = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # red
                [0.0, 1.0, 0.0],  # green
                [0.0, 0.0, 1.0],  # blue
                [1.0, 1.0, 0.0],  # yellow
                [1.0, 0.0, 1.0],  # magenta
            ],
            dtype=torch.float32,
        )

        expected_hsv = np.array(
            [
                [0.0, 1.0, 1.0],
                [1.0 / 3.0, 1.0, 1.0],
                [2.0 / 3.0, 1.0, 1.0],
                [1.0 / 6.0, 1.0, 1.0],
                [5.0 / 6.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        hsv = colors.rgb_to_hsv(rgb).cpu().numpy()
        np.testing.assert_allclose(hsv, expected_hsv, atol=1e-5, rtol=1e-5)

        rgb_roundtrip = colors.hsv_to_rgb(hsv)
        np.testing.assert_allclose(rgb_roundtrip, rgb.cpu().numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
