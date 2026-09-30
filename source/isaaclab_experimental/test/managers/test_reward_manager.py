# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Warp reward manager."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_experimental.managers import ManagerTermBase, RewardManager
from isaaclab_experimental.utils import WarpGraphCache
from isaaclab_experimental.utils.warp import WarpCapturable

from isaaclab.managers import RewardTermCfg

wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

DEVICE = "cuda:0"
NUM_ENVS = 4
CALLS: Counter[float] = Counter()


@WarpCapturable(lambda params: params.get("sensor_cfg") is None, reason="refreshes the sensor on the host")
def constant_reward(env, out: wp.array, value: float, sensor_cfg: str | None = None):
    """Function term that is capturable only without a sensor."""
    CALLS[value] += 1
    out.fill_(value)


class ConstantReward(ManagerTermBase):
    """Class term that decides its capturability per instance."""

    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)
        self._warp_capturable = cfg.params["capturable"]

    def __call__(self, env, out: wp.array, value: float, capturable: bool):
        CALLS[value] += 1
        out.fill_(value)


def test_capturability_is_decided_per_term_instance():
    """The same term is recorded for one parameter set and runs eagerly for another.

    An eager instance runs its Python on every step; a recorded one only while its graph is recorded.
    The guard of the predicate term raises if the instance with a sensor is ever recorded.
    """
    graph_cache = WarpGraphCache(DEVICE)
    env = SimpleNamespace(
        num_envs=NUM_ENVS,
        device=DEVICE,
        sim=SimpleNamespace(is_playing=lambda: True),
        max_episode_length_s=1.0,
        _warp_graph_cache=graph_cache,
    )
    cfg = {
        "function_recorded": RewardTermCfg(func=constant_reward, weight=1.0, params={"value": 1.0}),
        "function_eager": RewardTermCfg(
            func=constant_reward, weight=1.0, params={"value": 2.0, "sensor_cfg": "height_scanner"}
        ),
        "class_recorded": RewardTermCfg(func=ConstantReward, weight=1.0, params={"value": 4.0, "capturable": True}),
        "class_eager": RewardTermCfg(func=ConstantReward, weight=1.0, params={"value": 8.0, "capturable": False}),
    }
    manager = RewardManager(cfg, env)
    graph_cache.arm()
    CALLS.clear()

    for _ in range(3):
        reward = graph_cache.call_steps("RewardManager_compute", manager.stage_steps("compute"), dt=0.5)
    wp.synchronize()

    assert CALLS == {1.0: 1, 2.0: 3, 4.0: 1, 8.0: 3}
    assert torch.equal(reward, torch.full((NUM_ENVS,), (1.0 + 2.0 + 4.0 + 8.0) * 0.5, device=DEVICE))
    graph_cache.close()


def test_set_term_cfg_applies_to_the_recorded_reward_stage():
    """Raising a weight from zero after the stage recorded adds the term, as ``modify_reward_weight`` expects.

    A zero-weight term is skipped while the stage records, so a replay without recording again never runs it.
    """
    graph_cache = WarpGraphCache(DEVICE)
    env = SimpleNamespace(
        num_envs=NUM_ENVS,
        device=DEVICE,
        sim=SimpleNamespace(is_playing=lambda: True),
        max_episode_length_s=1.0,
        _warp_graph_cache=graph_cache,
    )
    manager = RewardManager({"late": RewardTermCfg(func=constant_reward, weight=0.0, params={"value": 3.0})}, env)
    graph_cache.arm()
    graph_cache.call_steps("RewardManager_compute", manager.stage_steps("compute"), dt=0.5)

    term_cfg = manager.get_term_cfg("late")
    term_cfg.weight = 2.0
    manager.set_term_cfg("late", term_cfg)
    reward = graph_cache.call_steps("RewardManager_compute", manager.stage_steps("compute"), dt=0.5)
    wp.synchronize()

    assert torch.equal(reward, torch.full((NUM_ENVS,), 3.0 * 2.0 * 0.5, device=DEVICE))
    graph_cache.close()
