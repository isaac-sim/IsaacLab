# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import unittest
from collections import namedtuple

from omni.isaac.orbit.utils.mdp.observation_manager import ObservationManager


class DefaultObservationManager(ObservationManager):
    def grilled_chicken(self, env):
        return torch.ones(env.num_envs, 4, device=self.device)

    def grilled_chicken_with_bbq(self, env, bbq: bool):
        return bbq * torch.ones(env.num_envs, 1, device=self.device)

    def grilled_chicken_with_curry(self, env, hot: bool):
        return hot * 2 * torch.ones(env.num_envs, 1, device=self.device)

    def grilled_chicken_with_yoghurt(self, env, hot: bool, bland: float):
        return hot * bland * torch.ones(env.num_envs, 5, device=self.device)


class TestObservationManager(unittest.TestCase):
    """Test cases for various situations with observation manager."""

    def setUp(self) -> None:
        self.env = namedtuple("IsaacEnv", ["num_envs"])(20)
        self.device = "cpu"

    def test_str(self):
        cfg = {
            "policy": {
                "grilled_chicken": {"scale": 10},
                "grilled_chicken_with_bbq": {"scale": 5, "bbq": True},
                "grilled_chicken_with_yoghurt": {"scale": 1.0, "hot": False, "bland": 2.0},
            }
        }
        self.obs_man = DefaultObservationManager(cfg, self.env, self.device)
        self.assertEqual(len(self.obs_man.active_terms["policy"]), 3)
        # print the expected string
        print()
        print(self.obs_man)

    def test_config_terms(self):
        cfg = {"policy": {"grilled_chicken": {"scale": 10}, "grilled_chicken_with_curry": {"scale": 0.0, "hot": False}}}
        self.obs_man = DefaultObservationManager(cfg, self.env, self.device)

        self.assertEqual(len(self.obs_man.active_terms["policy"]), 2)

    def test_compute(self):
        cfg = {"policy": {"grilled_chicken": {"scale": 10}, "grilled_chicken_with_curry": {"scale": 0.0, "hot": False}}}
        self.obs_man = DefaultObservationManager(cfg, self.env, self.device)
        # compute observation using manager
        observations = self.obs_man.compute()
        # check the observation shape
        self.assertEqual((self.env.num_envs, 5), observations["policy"].shape)

    def test_active_terms(self):
        cfg = {
            "policy": {
                "grilled_chicken": {"scale": 10},
                "grilled_chicken_with_bbq": {"scale": 5, "bbq": True},
                "grilled_chicken_with_curry": {"scale": 0.0, "hot": False},
            }
        }
        self.obs_man = DefaultObservationManager(cfg, self.env, self.device)

        self.assertEqual(len(self.obs_man.active_terms["policy"]), 3)

    def test_invalid_observation_name(self):
        cfg = {
            "policy": {
                "grilled_chicken": {"scale": 10},
                "grilled_chicken_with_bbq": {"scale": 5, "bbq": True},
                "grilled_chicken_with_no_bbq": {"scale": 0.1, "hot": False},
            }
        }
        with self.assertRaises(AttributeError):
            self.obs_man = DefaultObservationManager(cfg, self.env, self.device)

    def test_invalid_observation_config(self):
        cfg = {
            "policy": {
                "grilled_chicken_with_bbq": {"scale": 0.1, "hot": False},
                "grilled_chicken_with_yoghurt": {"scale": 2.0, "hot": False},
            }
        }
        with self.assertRaises(ValueError):
            self.obs_man = DefaultObservationManager(cfg, self.env, self.device)


if __name__ == "__main__":
    unittest.main()
