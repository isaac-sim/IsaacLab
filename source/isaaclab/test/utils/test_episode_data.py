# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows from here."""

import torch
import unittest

from isaaclab.utils.datasets import EpisodeData


class TestEpisodeData(unittest.TestCase):
    """Test EpisodeData implementation."""

    """
    Test cases for EpisodeData class.
    """

    def test_is_empty(self):
        """Test checking whether the episode is empty."""
        for device in ("cuda:0", "cpu"):
            with self.subTest(device=device):
                episode = EpisodeData()
                self.assertTrue(episode.is_empty())

                episode.add("key", torch.tensor([1, 2, 3], device=device))
                self.assertFalse(episode.is_empty())

    def test_add_tensors(self):
        """Test appending tensor data to the episode."""
        for device in ("cuda:0", "cpu"):
            with self.subTest(device=device):
                dummy_data_0 = torch.tensor([0], device=device)
                dummy_data_1 = torch.tensor([1], device=device)
                expected_added_data = torch.cat((dummy_data_0.unsqueeze(0), dummy_data_1.unsqueeze(0)))
                episode = EpisodeData()

                # test adding data to a key that does not exist
                episode.add("key", dummy_data_0)
                self.assertTrue(torch.equal(episode.data.get("key"), dummy_data_0.unsqueeze(0)))

                # test adding data to a key that exists
                episode.add("key", dummy_data_1)
                self.assertTrue(torch.equal(episode.data.get("key"), expected_added_data))

                # test adding data to a key with "/" in the name
                episode.add("first/second", dummy_data_0)
                self.assertTrue(torch.equal(episode.data.get("first").get("second"), dummy_data_0.unsqueeze(0)))

                # test adding data to a key with "/" in the name that already exists
                episode.add("first/second", dummy_data_1)
                self.assertTrue(torch.equal(episode.data.get("first").get("second"), expected_added_data))

    def test_add_dict_tensors(self):
        """Test appending dict data to the episode."""
        for device in ("cuda:0", "cpu"):
            with self.subTest(device=device):
                dummy_dict_data_0 = {
                    "key_0": torch.tensor([0], device=device),
                    "key_1": {"key_1_0": torch.tensor([1], device=device), "key_1_1": torch.tensor([2], device=device)},
                }
                dummy_dict_data_1 = {
                    "key_0": torch.tensor([3], device=device),
                    "key_1": {"key_1_0": torch.tensor([4], device=device), "key_1_1": torch.tensor([5], device=device)},
                }

                episode = EpisodeData()

                # test adding dict data to a key that does not exist
                episode.add("key", dummy_dict_data_0)
                self.assertTrue(torch.equal(episode.data.get("key").get("key_0"), torch.tensor([[0]], device=device)))
                self.assertTrue(
                    torch.equal(episode.data.get("key").get("key_1").get("key_1_0"), torch.tensor([[1]], device=device))
                )
                self.assertTrue(
                    torch.equal(episode.data.get("key").get("key_1").get("key_1_1"), torch.tensor([[2]], device=device))
                )

                # test adding dict data to a key that exists
                episode.add("key", dummy_dict_data_1)
                self.assertTrue(
                    torch.equal(episode.data.get("key").get("key_0"), torch.tensor([[0], [3]], device=device))
                )
                self.assertTrue(
                    torch.equal(
                        episode.data.get("key").get("key_1").get("key_1_0"), torch.tensor([[1], [4]], device=device)
                    )
                )
                self.assertTrue(
                    torch.equal(
                        episode.data.get("key").get("key_1").get("key_1_1"), torch.tensor([[2], [5]], device=device)
                    )
                )

    def test_get_initial_state(self):
        """Test getting the initial state of the episode."""
        for device in ("cuda:0", "cpu"):
            with self.subTest(device=device):
                dummy_initial_state = torch.tensor([1, 2, 3], device=device)
                episode = EpisodeData()

                episode.add("initial_state", dummy_initial_state)
                self.assertTrue(torch.equal(episode.get_initial_state(), dummy_initial_state.unsqueeze(0)))

    def test_get_next_action(self):
        """Test getting next actions."""
        for device in ("cuda:0", "cpu"):
            with self.subTest(device=device):
                # dummy actions
                action1 = torch.tensor([1, 2, 3], device=device)
                action2 = torch.tensor([4, 5, 6], device=device)
                action3 = torch.tensor([7, 8, 9], device=device)

                episode = EpisodeData()
                self.assertIsNone(episode.get_next_action())

                episode.add("actions", action1)
                episode.add("actions", action2)
                episode.add("actions", action3)

                # check if actions are returned in the correct order
                self.assertTrue(torch.equal(episode.get_next_action(), action1))
                self.assertTrue(torch.equal(episode.get_next_action(), action2))
                self.assertTrue(torch.equal(episode.get_next_action(), action3))

                # check if None is returned when all actions are exhausted
                self.assertIsNone(episode.get_next_action())


if __name__ == "__main__":
    run_tests()
