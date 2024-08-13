# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
import unittest
from collections import namedtuple

from omni.isaac.lab.managers import EventManager, EventTermCfg
from omni.isaac.lab.utils import configclass

DummyEnv = namedtuple("ManagerBasedRLEnv", ["num_envs", "dt", "device", "dummy1", "dummy2"])
"""Dummy environment for testing."""


def increment_by_one(env, env_ids: torch.Tensor):
    env.dummy1[env_ids] += 1


def reset_to_zero(env, env_ids: torch.Tensor):
    env.dummy1[env_ids] = 0


def increment_by_value(env, env_ids: torch.Tensor, value: int):
    env.dummy1[env_ids] += value


class TestEventManager(unittest.TestCase):
    """Test cases for various situations with event manager."""

    def setUp(self) -> None:
        # create values
        num_envs = 32
        device = "cpu"
        # create dummy tensors
        dummy1 = torch.zeros((num_envs, 2), device=device)
        dummy2 = torch.rand((num_envs, 10), device=device)
        # create dummy environment
        self.env = DummyEnv(num_envs, 0.1, device, dummy1, dummy2)

    def test_str(self):
        """Test the string representation of the event manager."""
        cfg = {
            "term_1": EventTermCfg(func=increment_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
            "term_2": EventTermCfg(func=reset_to_zero, mode="reset"),
            "term_3": EventTermCfg(func=increment_by_value, mode="custom", params={"value": 10}),
            "term_4": EventTermCfg(func=increment_by_value, mode="custom", params={"value": 2}),
        }
        self.event_man = EventManager(cfg, self.env)

        # print the expected string
        print()
        print(self.event_man)

    def test_config_equivalence(self):
        """Test the equivalence of event manager created from different config types."""
        # create from dictionary
        cfg = {
            "term_1": EventTermCfg(func=increment_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
            "term_2": EventTermCfg(func=reset_to_zero, mode="reset"),
            "term_3": EventTermCfg(func=increment_by_value, mode="custom", params={"value": 10}),
        }
        event_man_from_dict = EventManager(cfg, self.env)

        # create from config class
        @configclass
        class MyEventManagerCfg:
            """Event manager config with no type annotations."""

            term_1 = EventTermCfg(func=increment_by_one, mode="interval", interval_range_s=(0.1, 0.1))
            term_2 = EventTermCfg(func=reset_to_zero, mode="reset")
            term_3 = EventTermCfg(func=increment_by_value, mode="custom", params={"value": 10})

        cfg = MyEventManagerCfg()
        event_man_from_cfg = EventManager(cfg, self.env)

        # create from config class
        @configclass
        class MyEventManagerAnnotatedCfg:
            """Event manager config with type annotations."""

            term_1: EventTermCfg = EventTermCfg(func=increment_by_one, mode="interval", interval_range_s=(0.1, 0.1))
            term_2: EventTermCfg = EventTermCfg(func=reset_to_zero, mode="reset")
            term_3: EventTermCfg = EventTermCfg(func=increment_by_value, mode="custom", params={"value": 10})

        cfg = MyEventManagerAnnotatedCfg()
        event_man_from_annotated_cfg = EventManager(cfg, self.env)

        # check equivalence
        # parsed terms
        self.assertDictEqual(event_man_from_dict.active_terms, event_man_from_annotated_cfg.active_terms)
        self.assertDictEqual(event_man_from_cfg.active_terms, event_man_from_annotated_cfg.active_terms)
        self.assertDictEqual(event_man_from_dict.active_terms, event_man_from_cfg.active_terms)
        # parsed term configs
        self.assertDictEqual(event_man_from_dict._mode_term_cfgs, event_man_from_annotated_cfg._mode_term_cfgs)
        self.assertDictEqual(event_man_from_cfg._mode_term_cfgs, event_man_from_annotated_cfg._mode_term_cfgs)
        self.assertDictEqual(event_man_from_dict._mode_term_cfgs, event_man_from_cfg._mode_term_cfgs)

    def test_active_terms(self):
        """Test the correct reading of active terms."""
        cfg = {
            "term_1": EventTermCfg(func=increment_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
            "term_2": EventTermCfg(func=reset_to_zero, mode="reset"),
            "term_3": EventTermCfg(func=increment_by_value, mode="custom", params={"value": 10}),
            "term_4": EventTermCfg(func=increment_by_value, mode="custom", params={"value": 2}),
        }
        self.event_man = EventManager(cfg, self.env)

        self.assertEqual(len(self.event_man.active_terms), 3)
        self.assertEqual(len(self.event_man.active_terms["interval"]), 1)
        self.assertEqual(len(self.event_man.active_terms["reset"]), 1)
        self.assertEqual(len(self.event_man.active_terms["custom"]), 2)

    def test_invalid_event_func_module(self):
        """Test the handling of invalid event function's module in string representation."""
        cfg = {
            "term_1": EventTermCfg(func=increment_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
            "term_2": EventTermCfg(func="a:reset_to_zero", mode="reset"),
        }
        with self.assertRaises(ValueError):
            self.event_man = EventManager(cfg, self.env)

    def test_invalid_event_config(self):
        """Test the handling of invalid event function's config parameters."""
        cfg = {
            "term_1": EventTermCfg(func=increment_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
            "term_2": EventTermCfg(func=reset_to_zero, mode="reset"),
            "term_3": EventTermCfg(func=increment_by_value, mode="custom"),
        }
        with self.assertRaises(ValueError):
            self.event_man = EventManager(cfg, self.env)


if __name__ == "__main__":
    run_tests()
