# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows from here."""

import torch

import pytest

from isaaclab.utils.datasets import EpisodeData


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_is_empty(device):
    """Test checking whether the episode is empty."""
    episode = EpisodeData()
    assert episode.is_empty()

    episode.add("key", torch.tensor([1, 2, 3], device=device))
    assert not episode.is_empty()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_add_tensors(device):
    """Test appending tensor data to the episode."""
    dummy_data_0 = torch.tensor([0], device=device)
    dummy_data_1 = torch.tensor([1], device=device)
    expected_added_data = torch.cat((dummy_data_0.unsqueeze(0), dummy_data_1.unsqueeze(0)))
    episode = EpisodeData()

    # test adding data to a key that does not exist
    episode.add("key", dummy_data_0)
    key_data = episode.data.get("key")
    assert key_data is not None
    assert torch.equal(key_data, dummy_data_0.unsqueeze(0))

    # test adding data to a key that exists
    episode.add("key", dummy_data_1)
    key_data = episode.data.get("key")
    assert key_data is not None
    assert torch.equal(key_data, expected_added_data)

    # test adding data to a key with "/" in the name
    episode.add("first/second", dummy_data_0)
    first_data = episode.data.get("first")
    assert first_data is not None
    second_data = first_data.get("second")
    assert second_data is not None
    assert torch.equal(second_data, dummy_data_0.unsqueeze(0))

    # test adding data to a key with "/" in the name that already exists
    episode.add("first/second", dummy_data_1)
    first_data = episode.data.get("first")
    assert first_data is not None
    second_data = first_data.get("second")
    assert second_data is not None
    assert torch.equal(second_data, expected_added_data)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_add_dict_tensors(device):
    """Test appending dict data to the episode."""
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
    key_data = episode.data.get("key")
    assert key_data is not None
    key_0_data = key_data.get("key_0")
    assert key_0_data is not None
    assert torch.equal(key_0_data, torch.tensor([[0]], device=device))
    key_1_data = key_data.get("key_1")
    assert key_1_data is not None
    key_1_0_data = key_1_data.get("key_1_0")
    assert key_1_0_data is not None
    assert torch.equal(key_1_0_data, torch.tensor([[1]], device=device))
    key_1_1_data = key_1_data.get("key_1_1")
    assert key_1_1_data is not None
    assert torch.equal(key_1_1_data, torch.tensor([[2]], device=device))

    # test adding dict data to a key that exists
    episode.add("key", dummy_dict_data_1)
    key_data = episode.data.get("key")
    assert key_data is not None
    key_0_data = key_data.get("key_0")
    assert key_0_data is not None
    assert torch.equal(key_0_data, torch.tensor([[0], [3]], device=device))
    key_1_data = key_data.get("key_1")
    assert key_1_data is not None
    key_1_0_data = key_1_data.get("key_1_0")
    assert key_1_0_data is not None
    assert torch.equal(key_1_0_data, torch.tensor([[1], [4]], device=device))
    key_1_1_data = key_1_data.get("key_1_1")
    assert key_1_1_data is not None
    assert torch.equal(key_1_1_data, torch.tensor([[2], [5]], device=device))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_get_initial_state(device):
    """Test getting the initial state of the episode."""
    dummy_initial_state = torch.tensor([1, 2, 3], device=device)
    episode = EpisodeData()

    episode.add("initial_state", dummy_initial_state)
    initial_state = episode.get_initial_state()
    assert initial_state is not None
    assert torch.equal(initial_state, dummy_initial_state.unsqueeze(0))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_get_next_action(device):
    """Test getting next actions."""
    # dummy actions
    action1 = torch.tensor([1, 2, 3], device=device)
    action2 = torch.tensor([4, 5, 6], device=device)
    action3 = torch.tensor([7, 8, 9], device=device)

    episode = EpisodeData()
    assert episode.get_next_action() is None

    episode.add("actions", action1)
    episode.add("actions", action2)
    episode.add("actions", action3)

    # check if actions are returned in the correct order
    next_action = episode.get_next_action()
    assert next_action is not None
    assert torch.equal(next_action, action1)
    next_action = episode.get_next_action()
    assert next_action is not None
    assert torch.equal(next_action, action2)
    next_action = episode.get_next_action()
    assert next_action is not None
    assert torch.equal(next_action, action3)

    # check if None is returned when all actions are exhausted
    assert episode.get_next_action() is None
