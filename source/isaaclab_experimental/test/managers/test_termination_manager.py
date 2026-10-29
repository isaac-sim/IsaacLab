# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Warp termination manager."""

from types import SimpleNamespace

import pytest
import warp as wp
from isaaclab_experimental.managers import TerminationManager

from isaaclab.managers import TerminationTermCfg


def copy_flags(env, out: wp.array):
    """Warp-first termination term that reports the environment's preset flags."""
    wp.copy(out, env.flags)


def test_episode_termination_metric_averages_over_all_environments():
    """Like the stable manager, the metric is the share of all environments whose last episode the term ended.

    Only environment 0 is reset. It terminated, so averaging over the reset selection would log 1.0.
    """
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        flags=wp.array([True, True, False, False], dtype=wp.bool, device="cpu"),
    )
    manager = TerminationManager({"fell": TerminationTermCfg(func=copy_flags)}, env)
    manager.compute()

    env_mask = wp.array([True, False, False, False], dtype=wp.bool, device="cpu")
    extras = manager.reset(env_mask=env_mask)

    assert float(extras["Episode_Termination/fell"]) == pytest.approx(0.5)
