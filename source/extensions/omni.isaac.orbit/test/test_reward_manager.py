# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from collections import namedtuple

from omni.isaac.orbit.utils.mdp.reward_manager import RewardManager


class DefaultRewardManager(RewardManager):
    def grilled_chicken(self, env):
        return 1

    def grilled_chicken_with_bbq(self, env, bbq: bool):
        return 0

    def grilled_chicken_with_curry(self, env, hot: bool):
        return 0

    def grilled_chicken_with_yoghurt(self, env, hot: bool, bland: float):
        return 0


class TestRewardManager(unittest.TestCase):
    """Test cases for various situations with reward manager."""

    def setUp(self) -> None:
        self.env = namedtuple("IsaacEnv", [])()
        self.device = "cpu"
        self.num_envs = 20
        self.dt = 0.1

    def test_str(self):
        cfg = {
            "grilled_chicken": {"weight": 10},
            "grilled_chicken_with_bbq": {"weight": 5, "bbq": True},
            "grilled_chicken_with_yoghurt": {"weight": 1.0, "hot": False, "bland": 2.0},
        }
        self.rew_man = DefaultRewardManager(cfg, self.env, self.num_envs, self.dt, self.device)
        self.assertEqual(len(self.rew_man.active_terms), 3)
        # print the expected string
        print()
        print(self.rew_man)

    def test_config_terms(self):
        cfg = {"grilled_chicken": {"weight": 10}, "grilled_chicken_with_curry": {"weight": 0.0, "hot": False}}
        self.rew_man = DefaultRewardManager(cfg, self.env, self.num_envs, self.dt, self.device)

        self.assertEqual(len(self.rew_man.active_terms), 1)

    def test_compute(self):
        cfg = {"grilled_chicken": {"weight": 10}, "grilled_chicken_with_curry": {"weight": 0.0, "hot": False}}
        self.rew_man = DefaultRewardManager(cfg, self.env, self.num_envs, self.dt, self.device)
        # compute expected reward
        expected_reward = cfg["grilled_chicken"]["weight"] * self.dt
        # compute reward using manager
        rewards = self.rew_man.compute()
        # check the reward for environment index 0
        self.assertEqual(float(rewards[0]), expected_reward)

    def test_active_terms(self):
        cfg = {
            "grilled_chicken": {"weight": 10},
            "grilled_chicken_with_bbq": {"weight": 5, "bbq": True},
            "grilled_chicken_with_curry": {"weight": 0.0, "hot": False},
        }
        self.rew_man = DefaultRewardManager(cfg, self.env, self.num_envs, self.dt, self.device)

        self.assertEqual(len(self.rew_man.active_terms), 2)

    def test_invalid_reward_name(self):
        cfg = {
            "grilled_chicken": {"weight": 10},
            "grilled_chicken_with_bbq": {"weight": 5, "bbq": True},
            "grilled_chicken_with_no_bbq": {"weight": 0.1, "hot": False},
        }
        with self.assertRaises(AttributeError):
            self.rew_man = DefaultRewardManager(cfg, self.env, self.num_envs, self.dt, self.device)

    def test_invalid_reward_weight_config(self):
        cfg = {"grilled_chicken": {}}
        with self.assertRaises(KeyError):
            self.rew_man = DefaultRewardManager(cfg, self.env, self.num_envs, self.dt, self.device)

    def test_invalid_reward_config(self):
        cfg = {
            "grilled_chicken_with_bbq": {"weight": 0.1, "hot": False},
            "grilled_chicken_with_yoghurt": {"weight": 2.0, "hot": False},
        }
        with self.assertRaises(ValueError):
            self.rew_man = DefaultRewardManager(cfg, self.env, self.num_envs, self.dt, self.device)


if __name__ == "__main__":
    unittest.main()
