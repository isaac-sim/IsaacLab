# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.datasets import EpisodeData

pytestmark = pytest.mark.unit


@pytest.fixture(params=test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def device(request):
    return request.param


@pytest.mark.parametrize("key", ["key", "first/second"])
def test_add_tensors(device, key):
    episode = EpisodeData()
    assert episode.is_empty()
    values = torch.arange(2, device=device).reshape(2, 1)

    for index, value in enumerate(values):
        episode.add(key, value)
        data = episode.data
        for part in key.split("/"):
            data = data[part]
        torch.testing.assert_close(torch.stack(data), values[: index + 1])
        assert not episode.is_empty()


def test_add_dict_tensors(device):
    episode = EpisodeData()
    values = torch.arange(6, device=device).reshape(2, 3, 1)

    for index, row in enumerate(values):
        episode.add("key", {"key_0": row[0], "key_1": {"key_1_0": row[1], "key_1_1": row[2]}})
        data = episode.data["key"]
        for column, stored in enumerate([data["key_0"], data["key_1"]["key_1_0"], data["key_1"]["key_1_1"]]):
            torch.testing.assert_close(torch.stack(stored), values[: index + 1, column])


def test_get_initial_state(device):
    episode = EpisodeData()
    assert episode.get_initial_state() is None
    initial_state = torch.tensor([1, 2, 3], device=device)

    episode.add("initial_state", initial_state)

    torch.testing.assert_close(torch.stack(episode.get_initial_state()), initial_state.unsqueeze(0))


def test_get_next_action(device):
    episode = EpisodeData()
    assert episode.get_next_action() is None
    actions = torch.arange(1, 10, device=device).reshape(3, 3)

    for action in actions:
        episode.add("actions", action)

    for action in actions:
        torch.testing.assert_close(episode.get_next_action(), action)

    assert episode.get_next_action() is None
