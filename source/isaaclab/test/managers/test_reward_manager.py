# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
import pytest
from collections import namedtuple

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import RewardManager, RewardManagerBaseCfg, RewardTerm, RewardTermCfg
from isaaclab.utils import configclass


def grilled_chicken(env):
    return 1


def grilled_chicken_with_bbq(env, bbq: bool):
    return 0


def grilled_chicken_with_curry(env, hot: bool):
    return 0


def grilled_chicken_with_yoghurt(env, hot: bool, bland: float):
    return 0


class DummyRewardTerm(RewardTerm):
    """Dummy reward term that returns dummy reward."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

    def compute(self) -> torch.Tensor:
        return torch.ones(self._env.num_envs, device=self._env.device)


@configclass
class DummyRewardManagerCfg(RewardManagerBaseCfg):
    """Dummy reward configurations."""

    @configclass
    class DummyRewardTermCfg(RewardTermCfg):
        """Configuration for the dummy reward term."""

        class_type: type[RewardTerm] = DummyRewardTerm

    reward_term = DummyRewardTermCfg()


def create_dummy_env(device: str = "cpu") -> ManagerBasedEnv:
    """Create a dummy environment."""
    return namedtuple("ManagerBasedEnv", ["num_envs", "device"])(20, device)


def test_str():
    """Test the string representation of the reward manager."""
    # create reward manager
    cfg = DummyRewardManagerCfg()
    reward_manager = RewardManager(cfg, create_dummy_env())
    assert len(reward_manager.active_terms) == 1
    # print the expected string
    print()
    print(reward_manager)


def test_compute():
    """Test the computation of the reward."""
    for device in ("cuda:0", "cpu"):
        env = create_dummy_env(device)
        # create reward manager
        cfg = DummyRewardManagerCfg()
        reward_manager = RewardManager(cfg, env)

        # compute the reward
        reward = reward_manager.compute()

        # check the computed reward
        assert reward.shape == (env.num_envs,)
        assert torch.all(reward == 1.0)



