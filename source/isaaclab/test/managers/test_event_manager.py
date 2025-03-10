# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch
import pytest
from collections import namedtuple

from isaaclab.managers import EventManager, EventTermCfg
from isaaclab.utils import configclass

DummyEnv = namedtuple("ManagerBasedRLEnv", ["num_envs", "dt", "device", "dummy1", "dummy2"])
"""Dummy environment for testing."""


def reset_dummy1_to_zero(env, env_ids: torch.Tensor):
    env.dummy1[env_ids] = 0.0


def increment_dummy1_by_one(env, env_ids: torch.Tensor):
    env.dummy1[env_ids] += 1.0


def change_dummy1_by_value(env, env_ids: torch.Tensor, value: int):
    env.dummy1[env_ids] += value


def reset_dummy2_to_zero(env, env_ids: torch.Tensor):
    env.dummy2[env_ids] = 0.0


def increment_dummy2_by_one(env, env_ids: torch.Tensor):
    env.dummy2[env_ids] += 1.0


@pytest.fixture
def env():
    """Create a dummy environment."""
    return DummyEnv(20, 0.01, "cpu", torch.zeros(20, device="cpu"), torch.zeros(20, device="cpu"))


def test_str(env):
    """Test the string representation of the event manager."""
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
        "term_2": EventTermCfg(func=reset_dummy1_to_zero, mode="reset"),
        "term_3": EventTermCfg(func=change_dummy1_by_value, mode="custom", params={"value": 10}),
        "term_4": EventTermCfg(func=change_dummy1_by_value, mode="custom", params={"value": 2}),
    }
    event_man = EventManager(cfg, env)
    assert len(event_man.active_terms) == 3
    # print the expected string
    print()
    print(event_man)


def test_config_equivalence(env):
    """Test the equivalence of event manager created from different config types."""
    # create from dictionary
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
        "term_2": EventTermCfg(func=reset_dummy1_to_zero, mode="reset"),
        "term_3": EventTermCfg(func=change_dummy1_by_value, mode="custom", params={"value": 10}),
    }
    event_man_from_dict = EventManager(cfg, env)

    # create from config class
    @configclass
    class MyEventManagerCfg:
        """Event manager config with no type annotations."""

        term_1 = EventTermCfg(func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1))
        term_2 = EventTermCfg(func=reset_dummy1_to_zero, mode="reset")
        term_3 = EventTermCfg(func=change_dummy1_by_value, mode="custom", params={"value": 10})

    cfg = MyEventManagerCfg()
    event_man_from_cfg = EventManager(cfg, env)

    # create from config class
    @configclass
    class MyEventManagerAnnotatedCfg:
        """Event manager config with type annotations."""

        term_1: EventTermCfg = EventTermCfg(
            func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1)
        )
        term_2: EventTermCfg = EventTermCfg(func=reset_dummy1_to_zero, mode="reset")
        term_3: EventTermCfg = EventTermCfg(func=change_dummy1_by_value, mode="custom", params={"value": 10})

    cfg = MyEventManagerAnnotatedCfg()
    event_man_from_annotated_cfg = EventManager(cfg, env)

    # check equivalence
    # parsed terms
    assert event_man_from_dict.active_terms == event_man_from_annotated_cfg.active_terms
    assert event_man_from_cfg.active_terms == event_man_from_annotated_cfg.active_terms
    assert event_man_from_dict.active_terms == event_man_from_cfg.active_terms
    # parsed term configs
    assert event_man_from_dict._mode_term_cfgs == event_man_from_annotated_cfg._mode_term_cfgs
    assert event_man_from_cfg._mode_term_cfgs == event_man_from_annotated_cfg._mode_term_cfgs
    assert event_man_from_dict._mode_term_cfgs == event_man_from_cfg._mode_term_cfgs


def test_active_terms(env):
    """Test the correct reading of active terms."""
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
        "term_2": EventTermCfg(func=reset_dummy1_to_zero, mode="reset"),
        "term_3": EventTermCfg(func=change_dummy1_by_value, mode="custom", params={"value": 10}),
        "term_4": EventTermCfg(func=change_dummy1_by_value, mode="custom", params={"value": 2}),
    }
    event_man = EventManager(cfg, env)

    assert len(event_man.active_terms) == 3
    assert len(event_man.active_terms["interval"]) == 1
    assert len(event_man.active_terms["reset"]) == 1
    assert len(event_man.active_terms["custom"]) == 2


def test_config_empty(env):
    """Test the creation of event manager with empty config."""
    event_man = EventManager(None, env)
    assert len(event_man.active_terms) == 0

    # print the expected string
    print()
    print(event_man)


def test_invalid_event_func_module(env):
    """Test the handling of invalid event function's module in string representation."""
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
        "term_2": EventTermCfg(func="a:reset_dummy1_to_zero", mode="reset"),
    }
    with pytest.raises(ValueError):
        EventManager(cfg, env)


def test_invalid_event_config(env):
    """Test the handling of invalid event function's config parameters."""
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
        "term_2": EventTermCfg(func=change_dummy1_by_value, mode="custom", params={"hot": False}),
    }
    with pytest.raises(ValueError):
        EventManager(cfg, env)


def test_apply_interval_mode_without_global_time(env):
    """Test the application of event terms that are in interval mode without global time.

    We test that the event terms are applied at the correct intervals and that the dummy values are updated
    correctly.
    """
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
        "term_2": EventTermCfg(func=increment_dummy2_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
    }

    event_man = EventManager(cfg, env)

    # obtain the initial time left for the interval terms
    term_2_interval_time = event_man._interval_term_time_left[1].clone()

    # apply the event terms for a few steps
    for count in range(10):
        event_man.apply("interval", env_ids=None)

        # check if the dummy values are updated correctly
        assert torch.allclose(env.dummy1, torch.ones_like(env.dummy1) * (count + 1))
        assert torch.allclose(env.dummy2, torch.ones_like(env.dummy2) * (count + 1))

        # check if the time left for term 2 is updated correctly
        assert torch.allclose(
            event_man._interval_term_time_left[1],
            term_2_interval_time - torch.ones_like(term_2_interval_time) * (count + 1) * 0.1,
        )


def test_apply_interval_mode_with_global_time(env):
    """Test the application of event terms that are in interval mode with global time.

    We test that the event terms are applied at the correct intervals and that the dummy values are updated
    correctly.
    """
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
        "term_2": EventTermCfg(func=increment_dummy2_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
    }

    event_man = EventManager(cfg, env)

    # obtain the initial time left for the interval terms
    term_2_interval_time = event_man._interval_term_time_left[1].clone()

    # apply the event terms for a few steps
    for count in range(10):
        event_man.apply("interval", env_ids=None, global_env_step_count=count)

        # check if the dummy values are updated correctly
        assert torch.allclose(env.dummy1, torch.ones_like(env.dummy1) * (count + 1))
        assert torch.allclose(env.dummy2, torch.ones_like(env.dummy2) * (count + 1))

        # check if the time left for term 2 is updated correctly
        assert torch.allclose(
            event_man._interval_term_time_left[1],
            term_2_interval_time - torch.ones_like(term_2_interval_time) * (count + 1) * 0.1,
        )


def test_apply_reset_mode(env):
    """Test the application of event terms that are in reset mode.

    We test that the event terms are applied at the correct intervals and that the dummy values are updated
    correctly.
    """
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
        "term_2": EventTermCfg(func=reset_dummy1_to_zero, mode="reset"),
    }

    event_man = EventManager(cfg, env)

    # manually keep track of the expected values for dummy1 and trigger count
    expected_dummy1_value = torch.zeros_like(env.dummy1)
    term_2_trigger_step_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # apply the event terms for a few steps
    for count in range(10):
        event_man.apply("reset", env_ids=None, global_env_step_count=count)

        # check if the dummy values are updated correctly
        assert torch.allclose(env.dummy1, expected_dummy1_value)

        # check if the trigger count for term 2 is updated correctly
        trigger_ids = (count - term_2_trigger_step_id) >= 10
        if torch.any(trigger_ids):
            expected_dummy1_value[trigger_ids] = 0.0
            term_2_trigger_step_id[trigger_ids] = count


def test_apply_reset_mode_subset_env_ids(env):
    """Test the application of event terms that are in reset mode over a subset of env ids.

    We test that the event terms are applied at the correct intervals and that the dummy values are updated
    correctly.
    """
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
        "term_2": EventTermCfg(func=reset_dummy1_to_zero, mode="reset"),
    }

    event_man = EventManager(cfg, env)

    # manually keep track of the expected values for dummy1 and trigger count
    expected_dummy1_value = torch.zeros_like(env.dummy1)
    term_2_trigger_step_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # apply the event terms for a few steps
    for count in range(10):
        # randomly select a subset of env ids
        env_ids = (torch.rand(env.num_envs, device=env.device) < 0.5).nonzero().flatten()
        # apply the event terms for the selected env ids
        event_man.apply("reset", env_ids=env_ids, global_env_step_count=count)

        # check if the dummy values are updated correctly
        assert torch.allclose(env.dummy1, expected_dummy1_value)

        # check if the trigger count for term 2 is updated correctly
        trigger_ids = (count - term_2_trigger_step_id[env_ids]) >= 10
        if torch.any(trigger_ids):
            expected_dummy1_value[env_ids[trigger_ids]] = 0.0
            term_2_trigger_step_id[env_ids[trigger_ids]] = count


if __name__ == "__main__":
    run_tests()
