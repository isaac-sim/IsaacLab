# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for device-owned Warp reward weights."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import warp as wp
from isaaclab_experimental.managers import RewardManager, RewardTermCfg

from isaaclab.utils.warp import WarpLaunchCache


@wp.kernel
def _fill_unit_reward(out: wp.array(dtype=wp.float32)):
    """Test helper: write 1.0 into every env's term output."""
    out[wp.tid()] = 1.0


def _unit_reward(env, out: wp.array(dtype=wp.float32)) -> None:
    env.term_call_count += 1
    wp.launch(_fill_unit_reward, dim=env.num_envs, inputs=[out], device=env.device)


def test_device_owned_weight_can_enable_zero_weight_term() -> None:
    """A curriculum-owned weight should be authoritative over the initial config scalar."""
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: False),
        term_call_count=0,
    )
    env._warp_launch = WarpLaunchCache(device=env.device)
    manager = RewardManager({"scheduled": RewardTermCfg(func=_unit_reward, weight=0.0)}, env)

    torch.testing.assert_close(manager.compute(dt=1.0), torch.zeros(4))
    assert env.term_call_count == 0

    manager.get_term_weight_wp("scheduled").fill_(2.0)
    torch.testing.assert_close(manager.compute(dt=1.0), torch.full((4,), 2.0))
    assert env.term_call_count == 1
    assert manager.get_term_cfg("scheduled").weight == 2.0


def test_set_term_cfg_invalidates_recorded_work() -> None:
    """Setting a term config should invalidate graphs and recorded launches."""
    env = SimpleNamespace(
        num_envs=4,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: False),
        term_call_count=0,
        invalidate_wp_graphs=Mock(),
    )
    env._warp_launch = WarpLaunchCache(device=env.device)
    manager = RewardManager({"scheduled": RewardTermCfg(func=_unit_reward, weight=1.0)}, env)

    replacement = RewardTermCfg(func=_unit_reward, weight=2.0)
    manager.set_term_cfg("scheduled", replacement)

    assert manager.get_term_cfg("scheduled") is replacement
    env.invalidate_wp_graphs.assert_called_once_with()

    env.invalidate_wp_graphs.reset_mock()
    replacement.weight = 3.0
    manager.set_term_cfg("scheduled", replacement)
    env.invalidate_wp_graphs.assert_called_once_with()


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")
def test_reward_reset_replay_uses_changed_episode_length() -> None:
    """Reward reset normalization should specialize replay when the episode duration changes."""
    env = SimpleNamespace(
        num_envs=4,
        device="cuda:0",
        sim=SimpleNamespace(is_playing=lambda: False),
        term_call_count=0,
        max_episode_length_s=2.0,
    )
    env._warp_launch = WarpLaunchCache(mode="replay", debug=True, device=env.device)
    manager = RewardManager({"scheduled": RewardTermCfg(func=_unit_reward, weight=1.0)}, env)
    env_mask = wp.ones(env.num_envs, dtype=wp.bool, device=env.device)

    manager._episode_sums_wp.fill_(4.0)
    first = manager.reset(env_mask=env_mask)["Episode_Reward/scheduled"].clone()
    torch.testing.assert_close(first, torch.tensor(2.0, device=env.device))

    env.max_episode_length_s = 4.0
    manager._episode_sums_wp.fill_(4.0)
    second = manager.reset(env_mask=env_mask)["Episode_Reward/scheduled"].clone()
    torch.testing.assert_close(second, torch.tensor(1.0, device=env.device))
