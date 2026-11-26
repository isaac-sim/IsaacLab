# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for indexing recorded episode samples."""

import pytest
import torch

from isaaclab.utils.datasets import EpisodeData

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def test_episode_index_bounds() -> None:
    """Out-of-range indices return None while valid negative indices count from the end."""
    episode = EpisodeData()
    actions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    states = torch.tensor([[5.0], [6.0], [7.0]])
    targets = torch.tensor([[8.0], [9.0]])
    episode.data = {
        "actions": actions,
        "states": {"robot": {"position": states, "velocity": targets}},
        "joint_targets": {"robot": targets},
    }

    assert episode.get_action(2) is None
    assert episode.get_action(-3) is None
    torch.testing.assert_close(episode.get_action(-2), actions[0])
    torch.testing.assert_close(episode.get_action(-1), actions[1])

    # A nested sample is unavailable when any leaf lacks the requested index.
    assert episode.get_state(2) is None
    assert episode.get_state(-3) is None
    last_state = episode.get_state(-1)
    torch.testing.assert_close(last_state["robot"]["position"], states[-1, None])
    torch.testing.assert_close(last_state["robot"]["velocity"], targets[-1, None])

    assert episode.get_joint_target(2) is None
    assert episode.get_joint_target(-3) is None
    torch.testing.assert_close(episode.get_joint_target(-1)["robot"], targets[-1])


def test_empty_episode_tensors_have_no_samples() -> None:
    """Negative indexing into empty sample tensors must not raise."""
    episode = EpisodeData()
    empty = torch.empty((0, 2))
    episode.data = {"actions": empty, "states": {"robot": empty}, "joint_targets": empty}
    assert episode.get_action(-1) is None
    assert episode.get_state(-1) is None
    assert episode.get_joint_target(-1) is None
    assert episode.get_next_action() is None
    assert episode.get_next_state() is None
    assert episode.get_next_joint_target() is None
    assert episode.next_action_index == 0
    assert episode.next_state_index == 0
