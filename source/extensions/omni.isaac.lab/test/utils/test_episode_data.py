# Copyright (c) 2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows from here."""

import torch
import unittest

from omni.isaac.lab.utils.datasets import EpisodeData


class TestEpisodeData(unittest.TestCase):
    """Test EpisodeData implementation."""

    """
    Test cases for EpisodeData class.
    """

    def test_is_empty(self):
        """Test checking whether the episode is empty."""
        episode = EpisodeData()
        self.assertTrue(episode.is_empty())

        episode.add("key", torch.Tensor([1, 2, 3], device="cpu"))
        self.assertFalse(episode.is_empty())

    def test_add(self):
        """Test appending data to the episode."""
        dummy_data = torch.tensor([1], device="cpu")
        episode = EpisodeData()

        # test adding data to a key that does not exist
        episode.add("key", dummy_data)
        self.assertTrue(torch.equal(episode.data.get("key"), dummy_data.unsqueeze(0)))

        # test adding data to a key that exists
        episode.add("key", dummy_data)
        expected_data = torch.tensor([[1], [1]], device="cpu")
        self.assertTrue(torch.equal(episode.data.get("key"), expected_data))

        # test adding data to a key with "/" in the name
        episode.add("first/second", dummy_data)
        self.assertTrue(torch.equal(episode.data.get("first").get("second"), dummy_data.unsqueeze(0)))

        # test adding data to a key with "/" in the name that already exists
        episode.add("first/second", dummy_data)
        expected_data = torch.tensor([[1], [1]], device="cpu")
        self.assertTrue(torch.equal(episode.data.get("first").get("second"), expected_data))

    def test_get_initial_state(self):
        """Test getting the initial state of the episode."""
        dummy_initial_state = torch.tensor([1, 2, 3], device="cpu")
        episode = EpisodeData()

        episode.add("initial_state", dummy_initial_state)
        self.assertTrue(torch.equal(episode.get_initial_state(), dummy_initial_state.unsqueeze(0)))

    def test_get_next_action(self):
        """Test getting next actions."""
        # dummy actions
        action1 = torch.tensor([1, 2, 3], device="cpu")
        action2 = torch.tensor([4, 5, 6], device="cpu")
        action3 = torch.tensor([7, 8, 9], device="cpu")

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
