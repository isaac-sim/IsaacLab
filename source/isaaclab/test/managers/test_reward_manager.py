# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

import pytest
import torch

from isaaclab.managers import RewardManager, RewardTermCfg
from isaaclab.utils import configclass

pytestmark = pytest.mark.unit


def grilled_chicken(env):
    return 1


def grilled_chicken_with_bbq(env, bbq: bool):
    return 0


def grilled_chicken_with_curry(env, hot: bool):
    return 0


def grilled_chicken_with_yoghurt(env, hot: bool, bland: float):
    return 0


@pytest.fixture
def env(make_env):
    return make_env(dt=0.1)


def test_active_terms(env):
    """Terms are listed in configuration order in ``active_terms`` and in the string representation."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken_with_bbq, weight=5, params={"bbq": True}),
        "term_3": RewardTermCfg(func=grilled_chicken_with_yoghurt, weight=1.0, params={"hot": False, "bland": 2.0}),
    }
    rew_man = RewardManager(cfg, env)
    assert rew_man.active_terms == ["term_1", "term_2", "term_3"]
    rew_man_str = str(rew_man)
    assert "contains 3 active terms" in rew_man_str
    assert "term_3" in rew_man_str


def test_config_equivalence(env):
    """Dictionary, un-annotated and annotated configurations produce the same manager."""
    cfg = {
        "my_term": RewardTermCfg(func=grilled_chicken, weight=10),
        "your_term": RewardTermCfg(func=grilled_chicken_with_bbq, weight=2.0, params={"bbq": True}),
        "his_term": RewardTermCfg(func=grilled_chicken_with_yoghurt, weight=1.0, params={"hot": False, "bland": 2.0}),
    }

    @configclass
    class MyRewardManagerCfg:
        my_term = RewardTermCfg(func=grilled_chicken, weight=10.0)
        your_term = RewardTermCfg(func=grilled_chicken_with_bbq, weight=2.0, params={"bbq": True})
        his_term = RewardTermCfg(func=grilled_chicken_with_yoghurt, weight=1.0, params={"hot": False, "bland": 2.0})

    @configclass
    class MyRewardManagerAnnotatedCfg:
        my_term: RewardTermCfg = RewardTermCfg(func=grilled_chicken, weight=10.0)
        your_term: RewardTermCfg = RewardTermCfg(func=grilled_chicken_with_bbq, weight=2.0, params={"bbq": True})
        his_term: RewardTermCfg = RewardTermCfg(
            func=grilled_chicken_with_yoghurt, weight=1.0, params={"hot": False, "bland": 2.0}
        )

    managers = [RewardManager(c, env) for c in (cfg, MyRewardManagerCfg(), MyRewardManagerAnnotatedCfg())]
    for rew_man in managers[1:]:
        assert rew_man.active_terms == managers[0].active_terms
        assert rew_man._term_cfgs == managers[0]._term_cfgs


def test_compute(env):
    """The reward is the dt-scaled weighted sum of the terms; zero-weight terms are skipped."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken_with_curry, weight=0.0, params={"hot": False}),
    }
    rewards = RewardManager(cfg, env).compute(dt=env.dt)
    assert rewards.shape == (env.num_envs,)
    torch.testing.assert_close(rewards, torch.full((env.num_envs,), cfg["term_1"].weight * env.dt))


def test_config_empty(env):
    """An empty configuration yields no terms and zero rewards."""
    rew_man = RewardManager(None, env)
    assert len(rew_man.active_terms) == 0
    assert "contains 0 active terms" in str(rew_man)
    torch.testing.assert_close(rew_man.compute(dt=env.dt), torch.zeros(env.num_envs))


@pytest.mark.parametrize(
    ("term_2", "error"),
    [
        (RewardTermCfg(func=grilled_chicken_with_bbq, params={"bbq": True}), TypeError),
        (RewardTermCfg(func="a:grilled_chicken_with_no_bbq", weight=0.1, params={"hot": False}), ValueError),
        (RewardTermCfg(func=grilled_chicken_with_yoghurt, weight=2.0, params={"hot": False}), ValueError),
    ],
    ids=["missing_weight", "invalid_module", "missing_params"],
)
def test_invalid_reward_config(env, term_2, error):
    """Missing weights, unresolvable functions and unmatched parameters are rejected on construction."""
    cfg = {"term_1": RewardTermCfg(func=grilled_chicken, weight=10), "term_2": term_2}
    with pytest.raises(error):
        RewardManager(cfg, env)
