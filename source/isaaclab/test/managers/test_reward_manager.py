# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import MagicMock

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
def env():
    # simulation double that has not started playing
    sim = MagicMock()
    sim.is_playing.return_value = False
    return namedtuple("ManagerBasedRLEnv", ["num_envs", "dt", "device", "sim"])(20, 0.1, "cpu", sim)


def test_config_equivalence(env):
    """Test the equivalence of reward manager created from different config types."""
    # create from dictionary
    cfg = {
        "my_term": RewardTermCfg(func=grilled_chicken, weight=10),
        "your_term": RewardTermCfg(func=grilled_chicken_with_bbq, weight=2.0, params={"bbq": True}),
        "his_term": RewardTermCfg(
            func=grilled_chicken_with_yoghurt,
            weight=1.0,
            params={"hot": False, "bland": 2.0},
        ),
    }
    rew_man_from_dict = RewardManager(cfg, env)

    # create from config class
    @configclass
    class MyRewardManagerCfg:
        """Reward manager config with no type annotations."""

        my_term = RewardTermCfg(func=grilled_chicken, weight=10.0)
        your_term = RewardTermCfg(func=grilled_chicken_with_bbq, weight=2.0, params={"bbq": True})
        his_term = RewardTermCfg(func=grilled_chicken_with_yoghurt, weight=1.0, params={"hot": False, "bland": 2.0})

    cfg = MyRewardManagerCfg()
    rew_man_from_cfg = RewardManager(cfg, env)

    # create from config class
    @configclass
    class MyRewardManagerAnnotatedCfg:
        """Reward manager config with type annotations."""

        my_term: RewardTermCfg = RewardTermCfg(func=grilled_chicken, weight=10.0)
        your_term: RewardTermCfg = RewardTermCfg(func=grilled_chicken_with_bbq, weight=2.0, params={"bbq": True})
        his_term: RewardTermCfg = RewardTermCfg(
            func=grilled_chicken_with_yoghurt, weight=1.0, params={"hot": False, "bland": 2.0}
        )

    cfg = MyRewardManagerAnnotatedCfg()
    rew_man_from_annotated_cfg = RewardManager(cfg, env)

    # check equivalence
    # parsed terms
    assert rew_man_from_dict.active_terms == rew_man_from_annotated_cfg.active_terms
    assert rew_man_from_cfg.active_terms == rew_man_from_annotated_cfg.active_terms
    assert rew_man_from_dict.active_terms == rew_man_from_cfg.active_terms
    # parsed term configs
    assert rew_man_from_dict._term_cfgs == rew_man_from_annotated_cfg._term_cfgs
    assert rew_man_from_cfg._term_cfgs == rew_man_from_annotated_cfg._term_cfgs
    assert rew_man_from_dict._term_cfgs == rew_man_from_cfg._term_cfgs


def test_compute():
    """A partial reset reports each term's accumulated reward and preserves other environments."""
    num_envs, dt, max_episode_length_s = 3, 0.25, 2.0
    sim = MagicMock()
    sim.is_playing.return_value = False
    env = SimpleNamespace(num_envs=num_envs, device="cpu", sim=sim, max_episode_length_s=max_episode_length_s, step=0)
    cfg = {
        "scaled": RewardTermCfg(func=env_index_scaled, weight=-2.0, params={"factor": 1.0}),
        "scalar": RewardTermCfg(func=grilled_chicken, weight=4.0),
        "disabled": RewardTermCfg(func=grilled_chicken_with_bbq, weight=0.0, params={"bbq": True}),
    }
    rew_man = RewardManager(cfg, env)

    for step, expected in ((1, [1.0, 0.5, 0.0]), (2, [1.0, 0.0, -1.0])):
        env.step = step
        torch.testing.assert_close(rew_man.compute(dt=dt), torch.tensor(expected))

    reset_env_2 = rew_man.reset(torch.tensor([2]))
    assert {name: value.item() for name, value in reset_env_2.items()} == {
        "Episode_Reward/scaled": -1.5,
        "Episode_Reward/scalar": 1.0,
        "Episode_Reward/disabled": 0.0,
    }
    reset_env_1 = rew_man.reset(torch.tensor([1]))
    assert reset_env_1["Episode_Reward/scaled"].item() == -0.75
    assert reset_env_1["Episode_Reward/scalar"].item() == 1.0
    assert all(value.item() == 0.0 for value in rew_man.reset(torch.tensor([2])).values())


def test_config_empty(env):
    """Test the creation of reward manager with empty config."""
    rew_man = RewardManager(None, env)
    assert len(rew_man.active_terms) == 0

    # compute reward
    rewards = rew_man.compute(dt=env.dt)

    # check all rewards are zero
    torch.testing.assert_close(rewards, torch.zeros_like(rewards))


def test_active_terms(env):
    """Test the correct reading of active terms."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken_with_bbq, weight=5, params={"bbq": True}),
        "term_3": RewardTermCfg(func=grilled_chicken_with_curry, weight=0.0, params={"hot": False}),
    }
    rew_man = RewardManager(cfg, env)

    assert len(rew_man.active_terms) == 3
    # the string representation lists every active term
    assert "term_3" in str(rew_man)


def test_missing_weight(env):
    """Test the missing of weight in the config."""
    # TODO: The error should be raised during the config parsing, not during the reward manager creation.
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken_with_bbq, params={"bbq": True}),
    }
    with pytest.raises(TypeError):
        RewardManager(cfg, env)


def test_invalid_reward_func_module(env):
    """Test the handling of invalid reward function's module in string representation."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
        "term_2": RewardTermCfg(func=grilled_chicken_with_bbq, weight=5, params={"bbq": True}),
        "term_3": RewardTermCfg(func="a:grilled_chicken_with_no_bbq", weight=0.1, params={"hot": False}),
    }
    with pytest.raises(ValueError):
        RewardManager(cfg, env)


def test_invalid_reward_config(env):
    """Test the handling of invalid reward function's config parameters."""
    cfg = {
        "term_1": RewardTermCfg(func=grilled_chicken_with_bbq, weight=0.1, params={"hot": False}),
        "term_2": RewardTermCfg(func=grilled_chicken_with_yoghurt, weight=2.0, params={"hot": False}),
    }
    with pytest.raises(ValueError):
        RewardManager(cfg, env)


def env_index_scaled(env, factor: float):
    return factor * env.step * torch.arange(env.num_envs, dtype=torch.float, device=env.device)
