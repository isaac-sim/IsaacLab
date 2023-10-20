# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import unittest


class TestTorchOperations(unittest.TestCase):
    """Tests for assuring torch related operations used in Orbit."""

    def test_array_slicing(self):
        """Check that using ellipsis and slices work for torch tensors."""

        size = (400, 300, 5)
        my_tensor = torch.rand(size, device="cuda:0")

        self.assertEqual(my_tensor[..., 0].shape, (400, 300))
        self.assertEqual(my_tensor[:, :, 0].shape, (400, 300))
        self.assertEqual(my_tensor[slice(None), slice(None), 0].shape, (400, 300))
        with self.assertRaises(IndexError):
            my_tensor[..., ..., 0]

        self.assertEqual(my_tensor[0, ...].shape, (300, 5))
        self.assertEqual(my_tensor[0, :, :].shape, (300, 5))
        self.assertEqual(my_tensor[0, slice(None), slice(None)].shape, (300, 5))
        self.assertEqual(my_tensor[0, ..., ...].shape, (300, 5))

        self.assertEqual(my_tensor[..., 0, 0].shape, (400,))
        self.assertEqual(my_tensor[slice(None), 0, 0].shape, (400,))
        self.assertEqual(my_tensor[:, 0, 0].shape, (400,))


if __name__ == "__main__":
    unittest.main()
