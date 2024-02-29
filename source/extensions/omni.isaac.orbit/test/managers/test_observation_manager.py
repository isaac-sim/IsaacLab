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

import torch
import traceback
import unittest
from collections import namedtuple

import carb

from omni.isaac.orbit.managers import ManagerTermBase, ObservationGroupCfg, ObservationManager, ObservationTermCfg
from omni.isaac.orbit.utils import configclass


def grilled_chicken(env):
    return torch.ones(env.num_envs, 4, device=env.device)


def grilled_chicken_with_bbq(env, bbq: bool):
    return bbq * torch.ones(env.num_envs, 1, device=env.device)


def grilled_chicken_with_curry(env, hot: bool):
    return hot * 2 * torch.ones(env.num_envs, 1, device=env.device)


def grilled_chicken_with_yoghurt(env, hot: bool, bland: float):
    return hot * bland * torch.ones(env.num_envs, 5, device=env.device)


def grilled_chicken_with_yoghurt_and_bbq(env, hot: bool, bland: float, bbq: bool = False):
    return hot * bland * bbq * torch.ones(env.num_envs, 3, device=env.device)


class complex_function_class(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: object):
        self.cfg = cfg
        self.env = env
        # define some variables
        self._time_passed = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._time_passed[env_ids] = 0.0

    def __call__(self, env: object, interval: float) -> torch.Tensor:
        self._time_passed += interval
        return self._time_passed.clone().unsqueeze(-1)


class non_callable_complex_function_class(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: object):
        self.cfg = cfg
        self.env = env
        # define some variables
        self._cost = 2 * self.env.num_envs

    def call_me(self, env: object) -> torch.Tensor:
        return torch.ones(env.num_envs, 2, device=env.device) * self._cost


class TestObservationManager(unittest.TestCase):
    """Test cases for various situations with observation manager."""

    def setUp(self) -> None:
        self.env = namedtuple("BaseEnv", ["num_envs", "device"])(20, "cpu")

    def test_str(self):
        """Test the string representation of the observation manager."""

        @configclass
        class MyObservationManagerCfg:
            """Test config class for observation manager."""

            @configclass
            class SampleGroupCfg(ObservationGroupCfg):
                """Test config class for policy observation group."""

                term_1 = ObservationTermCfg(func="__main__:grilled_chicken", scale=10)
                term_2 = ObservationTermCfg(func=grilled_chicken, scale=2)
                term_3 = ObservationTermCfg(func=grilled_chicken_with_bbq, scale=5, params={"bbq": True})
                term_4 = ObservationTermCfg(
                    func=grilled_chicken_with_yoghurt, scale=1.0, params={"hot": False, "bland": 2.0}
                )
                term_5 = ObservationTermCfg(
                    func=grilled_chicken_with_yoghurt_and_bbq, scale=1.0, params={"hot": False, "bland": 2.0}
                )

            policy: ObservationGroupCfg = SampleGroupCfg()

        # create observation manager
        cfg = MyObservationManagerCfg()
        self.obs_man = ObservationManager(cfg, self.env)
        self.assertEqual(len(self.obs_man.active_terms["policy"]), 5)
        # print the expected string
        print()
        print(self.obs_man)

    def test_config_equivalence(self):
        """Test the equivalence of observation manager created from different config types."""

        # create from config class
        @configclass
        class MyObservationManagerCfg:
            """Test config class for observation manager."""

            @configclass
            class SampleGroupCfg(ObservationGroupCfg):
                """Test config class for policy observation group."""

                your_term = ObservationTermCfg(func="__main__:grilled_chicken", scale=10)
                his_term = ObservationTermCfg(func=grilled_chicken, scale=2)
                my_term = ObservationTermCfg(func=grilled_chicken_with_bbq, scale=5, params={"bbq": True})
                her_term = ObservationTermCfg(
                    func=grilled_chicken_with_yoghurt, scale=1.0, params={"hot": False, "bland": 2.0}
                )

            policy = SampleGroupCfg()
            critic = SampleGroupCfg(concatenate_terms=False, her_term=None)

        cfg = MyObservationManagerCfg()
        obs_man_from_cfg = ObservationManager(cfg, self.env)

        # create from config class
        @configclass
        class MyObservationManagerAnnotatedCfg:
            """Test config class for observation manager with annotations on terms."""

            @configclass
            class SampleGroupCfg(ObservationGroupCfg):
                """Test config class for policy observation group."""

                your_term: ObservationTermCfg = ObservationTermCfg(func="__main__:grilled_chicken", scale=10)
                his_term: ObservationTermCfg = ObservationTermCfg(func=grilled_chicken, scale=2)
                my_term: ObservationTermCfg = ObservationTermCfg(
                    func=grilled_chicken_with_bbq, scale=5, params={"bbq": True}
                )
                her_term: ObservationTermCfg = ObservationTermCfg(
                    func=grilled_chicken_with_yoghurt, scale=1.0, params={"hot": False, "bland": 2.0}
                )

            policy: ObservationGroupCfg = SampleGroupCfg()
            critic: ObservationGroupCfg = SampleGroupCfg(concatenate_terms=False, her_term=None)

        cfg = MyObservationManagerAnnotatedCfg()
        obs_man_from_annotated_cfg = ObservationManager(cfg, self.env)

        # check equivalence
        # parsed terms
        self.assertEqual(obs_man_from_cfg.active_terms, obs_man_from_annotated_cfg.active_terms)
        self.assertEqual(obs_man_from_cfg.group_obs_term_dim, obs_man_from_annotated_cfg.group_obs_term_dim)
        self.assertEqual(obs_man_from_cfg.group_obs_dim, obs_man_from_annotated_cfg.group_obs_dim)
        # parsed term configs
        self.assertEqual(obs_man_from_cfg._group_obs_term_cfgs, obs_man_from_annotated_cfg._group_obs_term_cfgs)
        self.assertEqual(obs_man_from_cfg._group_obs_concatenate, obs_man_from_annotated_cfg._group_obs_concatenate)

    def test_config_terms(self):
        """Test the number of terms in the observation manager."""

        @configclass
        class MyObservationManagerCfg:
            """Test config class for observation manager."""

            @configclass
            class SampleGroupCfg(ObservationGroupCfg):
                """Test config class for policy observation group."""

                term_1 = ObservationTermCfg(func=grilled_chicken, scale=10)
                term_2 = ObservationTermCfg(func=grilled_chicken_with_curry, scale=0.0, params={"hot": False})

            policy: ObservationGroupCfg = SampleGroupCfg()
            critic: ObservationGroupCfg = SampleGroupCfg(term_2=None)

        # create observation manager
        cfg = MyObservationManagerCfg()
        self.obs_man = ObservationManager(cfg, self.env)

        self.assertEqual(len(self.obs_man.active_terms["policy"]), 2)
        self.assertEqual(len(self.obs_man.active_terms["critic"]), 1)

    def test_compute(self):
        """Test the observation computation."""

        @configclass
        class MyObservationManagerCfg:
            """Test config class for observation manager."""

            @configclass
            class PolicyCfg(ObservationGroupCfg):
                """Test config class for policy observation group."""

                term_1 = ObservationTermCfg(func=grilled_chicken, scale=10)
                term_2 = ObservationTermCfg(func=grilled_chicken_with_curry, scale=0.0, params={"hot": False})

            policy: ObservationGroupCfg = PolicyCfg()

        # create observation manager
        cfg = MyObservationManagerCfg()
        self.obs_man = ObservationManager(cfg, self.env)
        # compute observation using manager
        observations = self.obs_man.compute()
        # check the observation shape
        self.assertEqual((self.env.num_envs, 5), observations["policy"].shape)

    def test_invalid_observation_config(self):
        """Test the invalid observation config."""

        @configclass
        class MyObservationManagerCfg:
            """Test config class for observation manager."""

            @configclass
            class PolicyCfg(ObservationGroupCfg):
                """Test config class for policy observation group."""

                term_1 = ObservationTermCfg(func=grilled_chicken_with_bbq, scale=0.1, params={"hot": False})
                term_2 = ObservationTermCfg(func=grilled_chicken_with_yoghurt, scale=2.0, params={"hot": False})

            policy: ObservationGroupCfg = PolicyCfg()

        # create observation manager
        cfg = MyObservationManagerCfg()
        # check the invalid config
        with self.assertRaises(ValueError):
            self.obs_man = ObservationManager(cfg, self.env)

    def test_callable_class_term(self):
        """Test the observation computation with callable class term."""

        @configclass
        class MyObservationManagerCfg:
            """Test config class for observation manager."""

            @configclass
            class PolicyCfg(ObservationGroupCfg):
                """Test config class for policy observation group."""

                term_1 = ObservationTermCfg(func=grilled_chicken, scale=10)
                term_2 = ObservationTermCfg(func=complex_function_class, scale=0.2, params={"interval": 0.5})

            policy: ObservationGroupCfg = PolicyCfg()

        # create observation manager
        cfg = MyObservationManagerCfg()
        self.obs_man = ObservationManager(cfg, self.env)
        # compute observation using manager
        observations = self.obs_man.compute()
        # check the observation
        self.assertEqual((self.env.num_envs, 5), observations["policy"].shape)
        self.assertAlmostEqual(observations["policy"][0, -1].item(), 0.2 * 0.5)

        # check memory in term
        num_exec_count = 10
        for _ in range(num_exec_count):
            observations = self.obs_man.compute()
        self.assertAlmostEqual(observations["policy"][0, -1].item(), 0.2 * 0.5 * (num_exec_count + 1))

        # check reset works
        self.obs_man.reset(env_ids=[0, 4, 9, 14, 19])
        observations = self.obs_man.compute()
        self.assertAlmostEqual(observations["policy"][0, -1].item(), 0.2 * 0.5)
        self.assertAlmostEqual(observations["policy"][1, -1].item(), 0.2 * 0.5 * (num_exec_count + 2))

    def test_non_callable_class_term(self):
        """Test the observation computation with non-callable class term."""

        @configclass
        class MyObservationManagerCfg:
            """Test config class for observation manager."""

            @configclass
            class PolicyCfg(ObservationGroupCfg):
                """Test config class for policy observation group."""

                term_1 = ObservationTermCfg(func=grilled_chicken, scale=10)
                term_2 = ObservationTermCfg(func=non_callable_complex_function_class, scale=0.2)

            policy: ObservationGroupCfg = PolicyCfg()

        # create observation manager config
        cfg = MyObservationManagerCfg()
        # create observation manager
        with self.assertRaises(NotImplementedError):
            self.obs_man = ObservationManager(cfg, self.env)


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
