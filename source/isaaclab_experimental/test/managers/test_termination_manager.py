# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Warp termination manager."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_experimental.managers import TerminationManager
from isaaclab_experimental.utils import CapturedStage

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


@wp.kernel
def _above_kernel(values: wp.array(dtype=wp.float32), threshold: float, out: wp.array(dtype=wp.bool)):
    i = wp.tid()
    out[i] = values[i] > threshold


def value_above(env, out: wp.array, threshold: float):
    """Warp-first termination term that compares the environment's values with a threshold."""
    wp.launch(_above_kernel, dim=env.num_envs, inputs=[env.values, threshold, out], device=env.device)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")
def test_set_term_cfg_applies_to_the_recorded_termination_stage(monkeypatch):
    """A threshold replaced after the stage recorded is used on the next call, not the recorded one."""
    monkeypatch.setattr(CapturedStage, "enabled", True)
    env = SimpleNamespace(
        num_envs=4,
        device="cuda:0",
        sim=SimpleNamespace(is_playing=lambda: True),
        values=wp.array([0.1, 0.3, 0.5, 0.7], dtype=wp.float32, device="cuda:0"),
    )
    manager = TerminationManager({"above": TerminationTermCfg(func=value_above, params={"threshold": 0.6})}, env)
    manager.compute()

    term_cfg = manager.get_term_cfg("above")
    term_cfg.params["threshold"] = 0.2
    manager.set_term_cfg("above", term_cfg)
    dones = manager.compute()
    wp.synchronize()

    assert torch.equal(dones.cpu(), torch.tensor([False, True, True, True]))
