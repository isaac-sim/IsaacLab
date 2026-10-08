# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Statistics of the smooth exploratory action process."""

import math

import pytest

pytest.importorskip("physicsnemo_robotics")

import torch
from isaaclab_physicsnemo import SmoothRandomActions, UniformRandomActions


def test_actions_have_requested_scale_and_correlation():
    actions = SmoothRandomActions(20000, 2, scale=0.5, correlation_steps=5.0, seed=3)
    previous = actions()
    current = actions()

    torch.testing.assert_close(current.std(), torch.tensor(0.5), atol=0.01, rtol=0.0)
    lag_one_correlation = torch.corrcoef(torch.stack((previous.flatten(), current.flatten())))[0, 1]
    torch.testing.assert_close(lag_one_correlation, torch.tensor(math.exp(-1.0 / 5.0)), atol=0.01, rtol=0.0)


def test_finished_environments_restart_independently():
    actions = SmoothRandomActions(20000, 1, scale=(0.1, 2.0), correlation_steps=1000.0, seed=3)
    previous_scales = actions.scales.clone()
    previous = actions() / actions.scales
    dones = torch.arange(20000) % 2 == 0

    current = actions(dones) / actions.scales

    continuing = torch.corrcoef(torch.stack((previous[~dones, 0], current[~dones, 0])))[0, 1]
    restarted = torch.corrcoef(torch.stack((previous[dones, 0], current[dones, 0])))[0, 1]
    assert continuing > 0.99
    assert abs(restarted) < 0.05
    torch.testing.assert_close(actions.scales[~dones], previous_scales[~dones], atol=0.0, rtol=0.0)
    assert torch.all(actions.scales[dones] != previous_scales[dones])


def test_scale_range_spans_gentle_and_saturating_episodes():
    """Each episode draws its own standard deviation, so the data covers the whole command range."""
    actions = SmoothRandomActions(20000, 1, scale=(0.1, 2.0), seed=3)
    samples = actions()

    assert actions.scales.min() >= 0.1 and actions.scales.max() <= 2.0
    torch.testing.assert_close(actions.scales.mean(), torch.tensor(1.05), atol=0.02, rtol=0.0)
    torch.testing.assert_close((samples / actions.scales).std(), torch.tensor(1.0), atol=0.02, rtol=0.0)


def test_uniform_targets_hold_for_a_control_period():
    """Recording at the physics rate keeps each target for one control period, and resets redraw it."""
    actions = UniformRandomActions(1000, 12, high=1.5, hold_steps=4, seed=3)
    first = [actions() for _ in range(4)]
    dones = torch.arange(1000) % 2 == 0
    fifth = actions()
    sixth = actions(dones)

    assert all(torch.equal(sample, first[0]) for sample in first[1:])
    assert first[0].abs().max() <= 1.5 and first[0].abs().max() > 1.4
    assert not torch.equal(fifth, first[0])
    torch.testing.assert_close(sixth[~dones], fifth[~dones], atol=0.0, rtol=0.0)
    assert torch.all(sixth[dones] != fifth[dones])
