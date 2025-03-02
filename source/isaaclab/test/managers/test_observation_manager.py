# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: torch.Tensor | None
from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
import unittest
from collections import namedtuple

from isaaclab.managers import ManagerTermBase, ObservationGroupCfg, ObservationManager, ObservationTermCfg
from isaaclab.utils import configclass, modifiers


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


def grilled_chicken_image(env, bland: float, channel: int = 1):
    return bland * torch.ones(env.num_envs, 128, 256, channel, device=env.device)


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


class MyDataClass:

    def __init__(self, num_envs: int, device: str):
        self.pos_w = torch.rand((num_envs, 3), device=device)
        self.lin_vel_w = torch.rand((num_envs, 3), device=device)


def pos_w_data(env) -> torch.Tensor:
    return env.data.pos_w


def lin_vel_w_data(env) -> torch.Tensor:
    return env.data.lin_vel_w


class TestObservationManager(unittest.TestCase):
    """Test cases for various situations with observation manager."""

    def setUp(self) -> None:
        # set up the environment
        self.dt = 0.01
        self.num_envs = 20
        self.device = "cuda:0"
        # create dummy environment
        self.env = namedtuple("ManagerBasedEnv", ["num_envs", "device", "data", "dt"])(
            self.num_envs, self.device, MyDataClass(self.num_envs, self.device), self.dt
        )

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
        obs_man_str = str(self.obs_man)
        print()
        print(obs_man_str)
        obs_man_str_split = obs_man_str.split("|")
        term_1_str_index = obs_man_str_split.index(" term_1           ")
        term_1_str_shape = obs_man_str_split[term_1_str_index + 1].strip()
        self.assertEqual(term_1_str_shape, "(4,)")

    def test_str_with_history(self):
        """Test the string representation of the observation manager with history terms."""

        TERM_1_HISTORY = 5

        @configclass
        class MyObservationManagerCfg:
            """Test config class for observation manager."""

            @configclass
            class SampleGroupCfg(ObservationGroupCfg):
                """Test config class for policy observation group."""

                term_1 = ObservationTermCfg(func="__main__:grilled_chicken", scale=10, history_length=TERM_1_HISTORY)
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
        obs_man_str = str(self.obs_man)
        print()
        print(obs_man_str)
        obs_man_str_split = obs_man_str.split("|")
        term_1_str_index = obs_man_str_split.index(" term_1           ")
        term_1_str_shape = obs_man_str_split[term_1_str_index + 1].strip()
        self.assertEqual(term_1_str_shape, "(20,)")

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

            @configclass
            class SampleMixedGroupCfg(ObservationGroupCfg):
                """Test config class for policy observation group with a mix of vector and matrix terms."""

                concatenate_terms = False
                term_1 = ObservationTermCfg(func=grilled_chicken, scale=2.0)
                term_2 = ObservationTermCfg(func=grilled_chicken_image, scale=1.5, params={"bland": 0.5})

            @configclass
            class SampleImageGroupCfg(ObservationGroupCfg):

                term_1 = ObservationTermCfg(func=grilled_chicken_image, scale=1.5, params={"bland": 0.5, "channel": 1})
                term_2 = ObservationTermCfg(func=grilled_chicken_image, scale=0.5, params={"bland": 0.1, "channel": 3})

            policy: ObservationGroupCfg = SampleGroupCfg()
            critic: ObservationGroupCfg = SampleGroupCfg(term_2=None)
            mixed: ObservationGroupCfg = SampleMixedGroupCfg()
            image: ObservationGroupCfg = SampleImageGroupCfg()

        # create observation manager
        cfg = MyObservationManagerCfg()
        self.obs_man = ObservationManager(cfg, self.env)

        self.assertEqual(len(self.obs_man.active_terms["policy"]), 2)
        self.assertEqual(len(self.obs_man.active_terms["critic"]), 1)
        self.assertEqual(len(self.obs_man.active_terms["mixed"]), 2)
        self.assertEqual(len(self.obs_man.active_terms["image"]), 2)

        # create a new obs manager but where mixed group has invalid config
        cfg = MyObservationManagerCfg()
        cfg.mixed.concatenate_terms = True

        with self.assertRaises(RuntimeError):
            ObservationManager(cfg, self.env)

    def test_compute(self):
        """Test the observation computation."""

        pos_scale_tuple = (2.0, 3.0, 1.0)

        @configclass
        class MyObservationManagerCfg:
            """Test config class for observation manager."""

            @configclass
            class PolicyCfg(ObservationGroupCfg):
                """Test config class for policy observation group."""

                term_1 = ObservationTermCfg(func=grilled_chicken, scale=10)
                term_2 = ObservationTermCfg(func=grilled_chicken_with_curry, scale=0.0, params={"hot": False})
                term_3 = ObservationTermCfg(func=pos_w_data, scale=pos_scale_tuple)
                term_4 = ObservationTermCfg(func=lin_vel_w_data, scale=1.5)

            @configclass
            class CriticCfg(ObservationGroupCfg):
                term_1 = ObservationTermCfg(func=pos_w_data, scale=pos_scale_tuple)
                term_2 = ObservationTermCfg(func=lin_vel_w_data, scale=1.5)
                term_3 = ObservationTermCfg(func=pos_w_data, scale=pos_scale_tuple)
                term_4 = ObservationTermCfg(func=lin_vel_w_data, scale=1.5)

            @configclass
            class ImageCfg(ObservationGroupCfg):

                term_1 = ObservationTermCfg(func=grilled_chicken_image, scale=1.5, params={"bland": 0.5, "channel": 1})
                term_2 = ObservationTermCfg(func=grilled_chicken_image, scale=0.5, params={"bland": 0.1, "channel": 3})

            policy: ObservationGroupCfg = PolicyCfg()
            critic: ObservationGroupCfg = CriticCfg()
            image: ObservationGroupCfg = ImageCfg()

        # create observation manager
        cfg = MyObservationManagerCfg()
        self.obs_man = ObservationManager(cfg, self.env)
        # compute observation using manager
        observations = self.obs_man.compute()

        # obtain the group observations
        obs_policy: torch.Tensor = observations["policy"]
        obs_critic: torch.Tensor = observations["critic"]
        obs_image: torch.Tensor = observations["image"]

        # check the observation shape
        self.assertEqual((self.env.num_envs, 11), obs_policy.shape)
        self.assertEqual((self.env.num_envs, 12), obs_critic.shape)
        self.assertEqual((self.env.num_envs, 128, 256, 4), obs_image.shape)
        # check that the scales are applied correctly
        torch.testing.assert_close(
            self.env.data.pos_w * torch.tensor(pos_scale_tuple, device=self.env.device), obs_critic[:, :3]
        )
        torch.testing.assert_close(self.env.data.lin_vel_w * 1.5, obs_critic[:, 3:6])
        # make sure that the data are the same for same terms
        # -- within group
        torch.testing.assert_close(obs_critic[:, 0:3], obs_critic[:, 6:9])
        torch.testing.assert_close(obs_critic[:, 3:6], obs_critic[:, 9:12])
        # -- between groups
        torch.testing.assert_close(obs_policy[:, 5:8], obs_critic[:, 0:3])
        torch.testing.assert_close(obs_policy[:, 8:11], obs_critic[:, 3:6])

    def test_compute_with_history(self):
        """Test the observation computation with history buffers."""
        HISTORY_LENGTH = 5

        @configclass
        class MyObservationManagerCfg:
            """Test config class for observation manager."""

            @configclass
            class PolicyCfg(ObservationGroupCfg):
                """Test config class for policy observation group."""

                term_1 = ObservationTermCfg(func=grilled_chicken, history_length=HISTORY_LENGTH)
                # total observation size: term_dim (4) * history_len (5) = 20
                term_2 = ObservationTermCfg(func=lin_vel_w_data)
                # total observation size: term_dim (3) = 3

            policy: ObservationGroupCfg = PolicyCfg()

        # create observation manager
        cfg = MyObservationManagerCfg()
        self.obs_man = ObservationManager(cfg, self.env)
        # compute observation using manager
        observations = self.obs_man.compute()
        # obtain the group observations
        obs_policy: torch.Tensor = observations["policy"]
        # check the observation shape
        self.assertEqual((self.env.num_envs, 23), obs_policy.shape)
        # check the observation data
        expected_obs_term_1_data = torch.ones(self.env.num_envs, 4 * HISTORY_LENGTH, device=self.env.device)
        expected_obs_term_2_data = lin_vel_w_data(self.env)
        expected_obs_data_t0 = torch.concat((expected_obs_term_1_data, expected_obs_term_2_data), dim=-1)
        print(expected_obs_data_t0, obs_policy)
        self.assertTrue(torch.equal(expected_obs_data_t0, obs_policy))
        # test that the history buffer holds previous data
        for _ in range(HISTORY_LENGTH):
            observations = self.obs_man.compute()
            obs_policy = observations["policy"]
        expected_obs_term_1_data = torch.ones(self.env.num_envs, 4 * HISTORY_LENGTH, device=self.env.device)
        expected_obs_data_t5 = torch.concat((expected_obs_term_1_data, expected_obs_term_2_data), dim=-1)
        self.assertTrue(torch.equal(expected_obs_data_t5, obs_policy))
        # test reset
        self.obs_man.reset()
        observations = self.obs_man.compute()
        obs_policy = observations["policy"]
        self.assertTrue(torch.equal(expected_obs_data_t0, obs_policy))
        # test reset of specific env ids
        reset_env_ids = [2, 4, 16]
        self.obs_man.reset(reset_env_ids)
        self.assertTrue(torch.equal(expected_obs_data_t0[reset_env_ids], obs_policy[reset_env_ids]))

    def test_compute_with_2d_history(self):
        """Test the observation computation with history buffers for 2D observations."""
        HISTORY_LENGTH = 5

        @configclass
        class MyObservationManagerCfg:
            """Test config class for observation manager."""

            @configclass
            class FlattenedPolicyCfg(ObservationGroupCfg):
                """Test config class for policy observation group."""

                term_1 = ObservationTermCfg(
                    func=grilled_chicken_image, params={"bland": 1.0, "channel": 1}, history_length=HISTORY_LENGTH
                )
                # total observation size: term_dim (128, 256) * history_len (5) = 163840

            @configclass
            class PolicyCfg(ObservationGroupCfg):
                """Test config class for policy observation group."""

                term_1 = ObservationTermCfg(
                    func=grilled_chicken_image,
                    params={"bland": 1.0, "channel": 1},
                    history_length=HISTORY_LENGTH,
                    flatten_history_dim=False,
                )
                # total observation size: (5, 128, 256, 1)

            flat_obs_policy: ObservationGroupCfg = FlattenedPolicyCfg()
            policy: ObservationGroupCfg = PolicyCfg()

        # create observation manager
        cfg = MyObservationManagerCfg()
        self.obs_man = ObservationManager(cfg, self.env)
        # compute observation using manager
        observations = self.obs_man.compute()
        # obtain the group observations
        obs_policy_flat: torch.Tensor = observations["flat_obs_policy"]
        obs_policy: torch.Tensor = observations["policy"]
        # check the observation shapes
        self.assertEqual((self.env.num_envs, 163840), obs_policy_flat.shape)
        self.assertEqual((self.env.num_envs, HISTORY_LENGTH, 128, 256, 1), obs_policy.shape)

    def test_compute_with_group_history(self):
        """Test the observation computation with group level history buffer configuration."""
        TERM_HISTORY_LENGTH = 5
        GROUP_HISTORY_LENGTH = 10

        @configclass
        class MyObservationManagerCfg:
            """Test config class for observation manager."""

            @configclass
            class PolicyCfg(ObservationGroupCfg):
                """Test config class for policy observation group."""

                history_length = GROUP_HISTORY_LENGTH
                # group level history length will override all terms
                term_1 = ObservationTermCfg(func=grilled_chicken, history_length=TERM_HISTORY_LENGTH)
                # total observation size: term_dim (4) * history_len (5) = 20
                # with override total obs size: term_dim (4) * history_len (10) = 40
                term_2 = ObservationTermCfg(func=lin_vel_w_data)
                # total observation size: term_dim (3) = 3
                # with override total obs size: term_dim (3) * history_len (10) = 30

            policy: ObservationGroupCfg = PolicyCfg()

        # create observation manager
        cfg = MyObservationManagerCfg()
        self.obs_man = ObservationManager(cfg, self.env)
        # compute observation using manager
        observations = self.obs_man.compute()
        # obtain the group observations
        obs_policy: torch.Tensor = observations["policy"]
        # check the total observation shape
        self.assertEqual((self.env.num_envs, 70), obs_policy.shape)
        # check the observation data is initialized properly
        expected_obs_term_1_data = torch.ones(self.env.num_envs, 4 * GROUP_HISTORY_LENGTH, device=self.env.device)
        expected_obs_term_2_data = lin_vel_w_data(self.env).repeat(1, GROUP_HISTORY_LENGTH)
        expected_obs_data_t0 = torch.concat((expected_obs_term_1_data, expected_obs_term_2_data), dim=-1)
        self.assertTrue(torch.equal(expected_obs_data_t0, obs_policy))
        # test that the history buffer holds previous data
        for _ in range(GROUP_HISTORY_LENGTH):
            observations = self.obs_man.compute()
            obs_policy = observations["policy"]
        expected_obs_term_1_data = torch.ones(self.env.num_envs, 4 * GROUP_HISTORY_LENGTH, device=self.env.device)
        expected_obs_term_2_data = lin_vel_w_data(self.env).repeat(1, GROUP_HISTORY_LENGTH)
        expected_obs_data_t10 = torch.concat((expected_obs_term_1_data, expected_obs_term_2_data), dim=-1)
        self.assertTrue(torch.equal(expected_obs_data_t10, obs_policy))
        # test reset
        self.obs_man.reset()
        observations = self.obs_man.compute()
        obs_policy = observations["policy"]
        self.assertTrue(torch.equal(expected_obs_data_t0, obs_policy))
        # test reset of specific env ids
        reset_env_ids = [2, 4, 16]
        self.obs_man.reset(reset_env_ids)
        self.assertTrue(torch.equal(expected_obs_data_t0[reset_env_ids], obs_policy[reset_env_ids]))

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

    def test_modifier_compute(self):
        """Test the observation computation with modifiers."""

        modifier_1 = modifiers.ModifierCfg(func=modifiers.bias, params={"value": 1.0})
        modifier_2 = modifiers.ModifierCfg(func=modifiers.scale, params={"multiplier": 2.0})
        modifier_3 = modifiers.ModifierCfg(func=modifiers.clip, params={"bounds": (-0.5, 0.5)})
        modifier_4 = modifiers.IntegratorCfg(dt=self.env.dt)

        @configclass
        class MyObservationManagerCfg:
            """Test config class for observation manager."""

            @configclass
            class PolicyCfg(ObservationGroupCfg):
                """Test config class for policy observation group."""

                concatenate_terms = False
                term_1 = ObservationTermCfg(func=pos_w_data, modifiers=[])
                term_2 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier_1])
                term_3 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier_1, modifier_4])

            @configclass
            class CriticCfg(ObservationGroupCfg):
                """Test config class for critic observation group"""

                concatenate_terms = False
                term_1 = ObservationTermCfg(func=pos_w_data, modifiers=[])
                term_2 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier_1])
                term_3 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier_1, modifier_2])
                term_4 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier_1, modifier_2, modifier_3])

            policy: ObservationGroupCfg = PolicyCfg()
            critic: ObservationGroupCfg = CriticCfg()

        # create observation manager
        cfg = MyObservationManagerCfg()
        self.obs_man = ObservationManager(cfg, self.env)
        # compute observation using manager
        observations = self.obs_man.compute()

        # obtain the group observations
        obs_policy: dict[str, torch.Tensor] = observations["policy"]
        obs_critic: dict[str, torch.Tensor] = observations["critic"]

        # check correct application of modifications
        torch.testing.assert_close(obs_policy["term_1"] + 1.0, obs_policy["term_2"])
        torch.testing.assert_close(obs_critic["term_1"] + 1.0, obs_critic["term_2"])
        torch.testing.assert_close(2.0 * (obs_critic["term_1"] + 1.0), obs_critic["term_3"])
        self.assertTrue(torch.min(obs_critic["term_4"]) >= -0.5)
        self.assertTrue(torch.max(obs_critic["term_4"]) <= 0.5)

    def test_modifier_invalid_config(self):
        """Test modifier initialization with invalid config."""

        modifier = modifiers.ModifierCfg(func=modifiers.clip, params={"min": -0.5, "max": 0.5})

        @configclass
        class MyObservationManagerCfg:
            """Test config class for observation manager."""

            @configclass
            class PolicyCfg(ObservationGroupCfg):
                """Test config class for policy observation group."""

                concatenate_terms = False
                term_1 = ObservationTermCfg(func=pos_w_data, modifiers=[modifier])

            policy: ObservationGroupCfg = PolicyCfg()

        # create observation manager
        cfg = MyObservationManagerCfg()

        with self.assertRaises(ValueError):
            self.obs_man = ObservationManager(cfg, self.env)


if __name__ == "__main__":
    run_tests()
