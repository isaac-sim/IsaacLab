# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for benchmark backend descriptors (pure data, Isaac-Sim-free)."""

import pytest

from isaaclab.test.benchmark.backend_descriptor import BACKEND_DESCRIPTORS, BackendDescriptor


def test_all_four_backends_present():
    assert set(BACKEND_DESCRIPTORS) == {"rsl_rl", "rl_games", "skrl", "sb3"}
    for d in BACKEND_DESCRIPTORS.values():
        assert isinstance(d, BackendDescriptor)


def test_key_matches_framework():
    """Every dict key must equal the descriptor's framework field."""
    for key, desc in BACKEND_DESCRIPTORS.items():
        assert key == desc.framework, f"Key {key!r} does not match descriptor.framework {desc.framework!r}"


@pytest.mark.parametrize(
    "framework, reward_tag, ep_length_tag, tfevents_pattern",
    [
        ("rsl_rl", "Train/mean_reward", "Train/mean_episode_length", "events*"),
        ("rl_games", "rewards/iter", "episode_lengths/iter", "summaries/events*"),
        ("skrl", "Reward / Total reward (mean)", "Episode / Total timesteps (mean)", None),
        ("sb3", None, None, "PPO_*/events*"),
    ],
)
def test_backend_tags(framework, reward_tag, ep_length_tag, tfevents_pattern):
    """Each backend descriptor exposes the expected literal TensorBoard tags and event glob."""
    d = BACKEND_DESCRIPTORS[framework]
    if reward_tag is not None:
        assert d.reward_tag == reward_tag
    if ep_length_tag is not None:
        assert d.ep_length_tag == ep_length_tag
    if tfevents_pattern is not None:
        assert d.tfevents_pattern == tfevents_pattern
