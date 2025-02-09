# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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

        # check if the buffer has zeros entries
        self.assertEqual(self.buffer.current_length.tolist(), [0, 0, 0])

    def test_reset_subset(self):
        """Test resetting a subset of batches in the circular buffer."""
        data1 = torch.ones((self.batch_size, 2), device=self.device)
        data2 = 2.0 * data1.clone()
        data3 = 3.0 * data1.clone()
        self.buffer.append(data1)
        self.buffer.append(data2)
        # reset the buffer
        reset_batch_id = 1
        self.buffer.reset(batch_ids=[reset_batch_id])
        # check that correct batch is reset
        self.assertEqual(self.buffer.current_length.tolist()[reset_batch_id], 0)
        # Append new set of data
        self.buffer.append(data3)
        # check if the correct number of entries are in each batch
        expected_length = [3, 3, 3]
        expected_length[reset_batch_id] = 1
        self.assertEqual(self.buffer.current_length.tolist(), expected_length)
        # check that all entries of the recently reset and appended batch are equal
        for i in range(self.max_len):
            torch.testing.assert_close(self.buffer.buffer[reset_batch_id, 0], self.buffer.buffer[reset_batch_id, i])

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

    def test_return_buffer_prop(self):
        """Test retrieving the whole buffer for correct size and contents.
        Returning the whole buffer should have the shape [batch_size,max_len,data.shape[1:]]
        """
        num_overflow = 2
        for i in range(self.buffer.max_length + num_overflow):
            data = torch.tensor([[i]], device=self.device).repeat(3, 2)
            self.buffer.append(data)

        retrieved_buffer = self.buffer.buffer
        # check shape
        self.assertTrue(retrieved_buffer.shape == torch.Size([self.buffer.batch_size, self.buffer.max_length, 2]))
        # check that batch is first dimension
        torch.testing.assert_close(retrieved_buffer[0], retrieved_buffer[1])
        # check oldest
        torch.testing.assert_close(
            retrieved_buffer[:, 0], torch.tensor([[num_overflow]], device=self.device).repeat(3, 2)
        )
        # check most recent
        torch.testing.assert_close(
            retrieved_buffer[:, -1],
            torch.tensor([[self.buffer.max_length + num_overflow - 1]], device=self.device).repeat(3, 2),
        )
        # check that it is returned oldest first
        for idx in range(self.buffer.max_length - 1):
            self.assertTrue(torch.all(torch.le(retrieved_buffer[:, idx], retrieved_buffer[:, idx + 1])))


if __name__ == "__main__":
    run_tests()
