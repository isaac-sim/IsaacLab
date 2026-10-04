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
from isaaclab_experimental.utils import CapturedStage
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


@WarpCapturable(False, reason="synchronizes with the host")
class ConstantReward(ManagerTermBase):
    """Class term that is never capturable."""

    def __call__(self, env, out: wp.array, value: float):
        CALLS[value] += 1
        out.fill_(value)


def test_capturability_is_decided_per_term_from_its_parameters(monkeypatch):
    """The same function term is recorded for one parameter set and runs eagerly for another.

    An eager term runs its Python on every step; a recorded one only while its graph is recorded.
    The guard of the predicate term raises if the instance with a sensor is ever recorded.
    """
    monkeypatch.setattr(CapturedStage, "enabled", True)
    env = SimpleNamespace(
        num_envs=NUM_ENVS, device=DEVICE, sim=SimpleNamespace(is_playing=lambda: True), max_episode_length_s=1.0
    )
    cfg = {
        "function_recorded": RewardTermCfg(func=constant_reward, weight=1.0, params={"value": 1.0}),
        "function_eager": RewardTermCfg(
            func=constant_reward, weight=1.0, params={"value": 2.0, "sensor_cfg": "height_scanner"}
        ),
        "class_eager": RewardTermCfg(func=ConstantReward, weight=1.0, params={"value": 4.0}),
    }
    manager = RewardManager(cfg, env)
    CALLS.clear()

    for _ in range(3):
        reward = manager.compute(dt=0.5)
    wp.synchronize()

    assert CALLS == {1.0: 1, 2.0: 3, 4.0: 3}
    # each eager term splits the recording, so it holds one graph before each of the two eager terms and one after
    assert manager._captured_stages[RewardManager.compute].num_graphs == 3
    assert torch.equal(reward, torch.full((NUM_ENVS,), (1.0 + 2.0 + 4.0) * 0.5, device=DEVICE))


def test_set_term_cfg_records_the_reward_stage_again_only_when_it_runs_a_different_term(monkeypatch):
    """``modify_reward_weight`` sets its term on every reset past its threshold, changing the weight in place.

    The replayed reward matches the eager reward after every setting. A weight change needs no new recording, since
    the stage reads the weights on the device; a weight turning zero or nonzero, or a parameter change, records the
    stage again once. Every recording runs the always-on term's Python once, so its call count is the number of
    recordings.
    """
    env = SimpleNamespace(
        num_envs=NUM_ENVS, device=DEVICE, sim=SimpleNamespace(is_playing=lambda: True), max_episode_length_s=1.0
    )
    manager = RewardManager(
        {
            "always": RewardTermCfg(func=constant_reward, weight=1.0, params={"value": 1.0}),
            "late": RewardTermCfg(func=constant_reward, weight=0.0, params={"value": 3.0}),
        },
        env,
    )
    settings = [(0.0, 3.0), (2.0, 3.0), (2.0, 3.0), (4.0, 3.0), (4.0, 5.0), (0.0, 5.0)]
    late = manager.get_term_cfg("late")

    def run() -> tuple[list[torch.Tensor], list[int]]:
        rewards, calls = [], []
        for weight, value in settings:
            late.weight = weight
            late.params["value"] = value
            manager.set_term_cfg("late", late)
            rewards.append(manager.compute(dt=0.5).clone())
            calls.append(CALLS[1.0])
        return rewards, calls

    monkeypatch.setattr(CapturedStage, "enabled", True)
    CALLS.clear()
    replayed, recordings = run()
    monkeypatch.setattr(CapturedStage, "enabled", False)
    eager, _ = run()

    assert recordings == [1, 2, 2, 2, 3, 4]
    for (weight, value), replayed_reward, eager_reward in zip(settings, replayed, eager):
        assert torch.equal(replayed_reward, eager_reward)
        assert torch.equal(replayed_reward, torch.full((NUM_ENVS,), (1.0 + weight * value) * 0.5, device=DEVICE))
