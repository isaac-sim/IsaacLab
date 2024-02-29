# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.orbit.app import AppLauncher

# launch omniverse app
config = {"headless": True}
simulation_app = AppLauncher(config).app

"""Rest everything follows."""

import traceback
import unittest
from collections import namedtuple

import carb

from omni.isaac.orbit.managers import RewardManager, RewardTermCfg
from omni.isaac.orbit.utils import configclass


def grilled_chicken(env):
    return 1


def grilled_chicken_with_bbq(env, bbq: bool):
    return 0


def grilled_chicken_with_curry(env, hot: bool):
    return 0


def grilled_chicken_with_yoghurt(env, hot: bool, bland: float):
    return 0


class TestRewardManager(unittest.TestCase):
    """Test cases for various situations with reward manager."""

    def setUp(self) -> None:
        self.env = namedtuple("RLTaskEnv", ["num_envs", "dt", "device"])(20, 0.1, "cpu")

    def test_str(self):
        """Test the string representation of the reward manager."""
        cfg = {
            "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
            "term_2": RewardTermCfg(func=grilled_chicken_with_bbq, weight=5, params={"bbq": True}),
            "term_3": RewardTermCfg(
                func=grilled_chicken_with_yoghurt,
                weight=1.0,
                params={"hot": False, "bland": 2.0},
            ),
        }
        self.rew_man = RewardManager(cfg, self.env)
        self.assertEqual(len(self.rew_man.active_terms), 3)
        # print the expected string
        print()
        print(self.rew_man)

    def test_config_equivalence(self):
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
        rew_man_from_dict = RewardManager(cfg, self.env)

        # create from config class
        @configclass
        class MyRewardManagerCfg:
            """Reward manager config with no type annotations."""

            my_term = RewardTermCfg(func=grilled_chicken, weight=10.0)
            your_term = RewardTermCfg(func=grilled_chicken_with_bbq, weight=2.0, params={"bbq": True})
            his_term = RewardTermCfg(func=grilled_chicken_with_yoghurt, weight=1.0, params={"hot": False, "bland": 2.0})

        cfg = MyRewardManagerCfg()
        rew_man_from_cfg = RewardManager(cfg, self.env)

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
        rew_man_from_annotated_cfg = RewardManager(cfg, self.env)

        # check equivalence
        # parsed terms
        self.assertEqual(rew_man_from_dict.active_terms, rew_man_from_annotated_cfg.active_terms)
        self.assertEqual(rew_man_from_cfg.active_terms, rew_man_from_annotated_cfg.active_terms)
        self.assertEqual(rew_man_from_dict.active_terms, rew_man_from_cfg.active_terms)
        # parsed term configs
        self.assertEqual(rew_man_from_dict._term_cfgs, rew_man_from_annotated_cfg._term_cfgs)
        self.assertEqual(rew_man_from_cfg._term_cfgs, rew_man_from_annotated_cfg._term_cfgs)
        self.assertEqual(rew_man_from_dict._term_cfgs, rew_man_from_cfg._term_cfgs)

    def test_compute(self):
        """Test the computation of reward."""
        cfg = {
            "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
            "term_2": RewardTermCfg(func=grilled_chicken_with_curry, weight=0.0, params={"hot": False}),
        }
        self.rew_man = RewardManager(cfg, self.env)
        # compute expected reward
        expected_reward = cfg["term_1"].weight * self.env.dt
        # compute reward using manager
        rewards = self.rew_man.compute(dt=self.env.dt)
        # check the reward for environment index 0
        self.assertEqual(float(rewards[0]), expected_reward)
        self.assertEqual(tuple(rewards.shape), (self.env.num_envs,))

    def test_active_terms(self):
        """Test the correct reading of active terms."""
        cfg = {
            "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
            "term_2": RewardTermCfg(func=grilled_chicken_with_bbq, weight=5, params={"bbq": True}),
            "term_3": RewardTermCfg(func=grilled_chicken_with_curry, weight=0.0, params={"hot": False}),
        }
        self.rew_man = RewardManager(cfg, self.env)

        self.assertEqual(len(self.rew_man.active_terms), 3)

    def test_missing_weight(self):
        """Test the missing of weight in the config."""
        # TODO: The error should be raised during the config parsing, not during the reward manager creation.
        cfg = {
            "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
            "term_2": RewardTermCfg(func=grilled_chicken_with_bbq, params={"bbq": True}),
        }
        with self.assertRaises(TypeError):
            self.rew_man = RewardManager(cfg, self.env)

    def test_invalid_reward_func_module(self):
        """Test the handling of invalid reward function's module in string representation."""
        cfg = {
            "term_1": RewardTermCfg(func=grilled_chicken, weight=10),
            "term_2": RewardTermCfg(func=grilled_chicken_with_bbq, weight=5, params={"bbq": True}),
            "term_3": RewardTermCfg(func="a:grilled_chicken_with_no_bbq", weight=0.1, params={"hot": False}),
        }
        with self.assertRaises(ValueError):
            self.rew_man = RewardManager(cfg, self.env)

    def test_invalid_reward_config(self):
        """Test the handling of invalid reward function's config parameters."""
        cfg = {
            "term_1": RewardTermCfg(func=grilled_chicken_with_bbq, weight=0.1, params={"hot": False}),
            "term_2": RewardTermCfg(func=grilled_chicken_with_yoghurt, weight=2.0, params={"hot": False}),
        }
        with self.assertRaises(ValueError):
            self.rew_man = RewardManager(cfg, self.env)


if __name__ == "__main__":
    try:
        unittest.main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
