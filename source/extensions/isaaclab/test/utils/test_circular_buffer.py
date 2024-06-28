# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import unittest

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows from here."""

from isaaclab.utils import CircularBuffer


class TestCircularBuffer(unittest.TestCase):
    """Test fixture for checking the circular buffer implementation."""

    def setUp(self):
        self.max_len = 5
        self.batch_size = 3
        self.device = "cpu"
        self.buffer = CircularBuffer(self.max_len, self.batch_size, self.device)

    """
    Test cases for CircularBuffer class.
    """

    def test_initialization(self):
        """Test initialization of the circular buffer."""
        self.assertEqual(self.buffer.max_length, self.max_len)
        self.assertEqual(self.buffer.batch_size, self.batch_size)
        self.assertEqual(self.buffer.device, self.device)
        self.assertEqual(self.buffer.current_length.tolist(), [0, 0, 0])

    def test_reset(self):
        """Test resetting the circular buffer."""
        # append some data
        data = torch.ones((self.batch_size, 2), device=self.device)
        self.buffer.append(data)
        # reset the buffer
        self.buffer.reset()

        # check if the buffer is empty
        self.assertEqual(self.buffer.current_length.tolist(), [0, 0, 0])

    def test_append_and_retrieve(self):
        """Test appending and retrieving data from the circular buffer."""
        # append some data
        data1 = torch.tensor([[1, 1], [1, 1], [1, 1]], device=self.device)
        data2 = torch.tensor([[2, 2], [2, 2], [2, 2]], device=self.device)

        self.buffer.append(data1)
        self.buffer.append(data2)

        self.assertEqual(self.buffer.current_length.tolist(), [2, 2, 2])

        retrieved_data = self.buffer[torch.tensor([0, 0, 0], device=self.device)]
        self.assertTrue(torch.equal(retrieved_data, data2))

        retrieved_data = self.buffer[torch.tensor([1, 1, 1], device=self.device)]
        self.assertTrue(torch.equal(retrieved_data, data1))

    def test_buffer_overflow(self):
        """Test buffer overflow.

        If the buffer is full, the oldest data should be overwritten.
        """
        # add data in ascending order
        for count in range(self.max_len + 2):
            data = torch.full((self.batch_size, 4), count, device=self.device)
            self.buffer.append(data)

        # check buffer length is correct
        self.assertEqual(self.buffer.current_length.tolist(), [self.max_len, self.max_len, self.max_len])

        # retrieve most recent data
        key = torch.tensor([0, 0, 0], device=self.device)
        retrieved_data = self.buffer[key]
        expected_data = torch.full_like(data, self.max_len + 1)

        self.assertTrue(torch.equal(retrieved_data, expected_data))

        # retrieve the oldest data
        key = torch.tensor([self.max_len - 1, self.max_len - 1, self.max_len - 1], device=self.device)
        retrieved_data = self.buffer[key]
        expected_data = torch.full_like(data, 2)

        self.assertTrue(torch.equal(retrieved_data, expected_data))

    def test_empty_buffer_access(self):
        """Test accessing an empty buffer."""
        with self.assertRaises(RuntimeError):
            self.buffer[torch.tensor([0, 0, 0], device=self.device)]

    def test_invalid_batch_size(self):
        """Test appending data with an invalid batch size."""
        data = torch.ones((self.batch_size + 1, 2), device=self.device)
        with self.assertRaises(ValueError):
            self.buffer.append(data)

        with self.assertRaises(ValueError):
            self.buffer[torch.tensor([0, 0], device=self.device)]

    def test_key_greater_than_pushes(self):
        """Test retrieving data with a key greater than the number of pushes.

        In this case, the oldest data should be returned.
        """
        data1 = torch.tensor([[1, 1], [1, 1], [1, 1]], device=self.device)
        data2 = torch.tensor([[2, 2], [2, 2], [2, 2]], device=self.device)

        self.buffer.append(data1)
        self.buffer.append(data2)

        retrieved_data = self.buffer[torch.tensor([5, 5, 5], device=self.device)]
        self.assertTrue(torch.equal(retrieved_data, data1))


if __name__ == "__main__":
    run_tests()
