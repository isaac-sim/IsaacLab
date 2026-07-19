# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for device-owned Warp reward weights."""

from types import SimpleNamespace

import torch
import warp as wp
from isaaclab_experimental.managers import RewardManager, RewardTermCfg


@wp.kernel
def _fill_unit_reward(out: wp.array(dtype=wp.float32)):
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
    manager = RewardManager({"scheduled": RewardTermCfg(func=_unit_reward, weight=0.0)}, env)

    torch.testing.assert_close(manager.compute(dt=1.0), torch.zeros(4))
    assert env.term_call_count == 0

    manager.get_term_weight_wp("scheduled").fill_(2.0)
    torch.testing.assert_close(manager.compute(dt=1.0), torch.full((4,), 2.0))
    assert env.term_call_count == 1
    assert manager.get_term_cfg("scheduled").weight == 2.0
