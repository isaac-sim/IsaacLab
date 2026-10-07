# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Statistics of the smooth exploratory action process."""

import math

import pytest

pytest.importorskip("physicsnemo_robotics")

import torch
from isaaclab_physicsnemo import SmoothRandomActions


def test_actions_have_requested_scale_and_correlation():
    actions = SmoothRandomActions(20000, 2, scale=0.5, correlation_steps=5.0, seed=3)
    previous = actions()
    current = actions()

    torch.testing.assert_close(current.std(), torch.tensor(0.5), atol=0.01, rtol=0.0)
    lag_one_correlation = torch.corrcoef(torch.stack((previous.flatten(), current.flatten())))[0, 1]
    torch.testing.assert_close(lag_one_correlation, torch.tensor(math.exp(-1.0 / 5.0)), atol=0.01, rtol=0.0)


def test_finished_environments_restart_independently():
    actions = SmoothRandomActions(20000, 1, scale=1.0, correlation_steps=1000.0, seed=3)
    previous = actions()
    dones = torch.arange(20000) % 2 == 0

    current = actions(dones)

    continuing = torch.corrcoef(torch.stack((previous[~dones, 0], current[~dones, 0])))[0, 1]
    restarted = torch.corrcoef(torch.stack((previous[dones, 0], current[dones, 0])))[0, 1]
    assert continuing > 0.99
    assert abs(restarted) < 0.05
