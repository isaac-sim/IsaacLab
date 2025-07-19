# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""
from collections.abc import Sequence

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""


import torch
from collections import namedtuple

import pytest

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventManager, EventTermCfg, ManagerTermBase, ManagerTermBaseCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

DummyEnv = namedtuple("ManagerBasedRLEnv", ["num_envs", "dt", "device", "sim", "dummy1", "dummy2"])
"""Dummy environment for testing."""


def reset_dummy1_to_zero(env, env_ids: torch.Tensor):
    env.dummy1[env_ids] = 0


def increment_dummy1_by_one(env, env_ids: torch.Tensor):
    env.dummy1[env_ids] += 1


def change_dummy1_by_value(env, env_ids: torch.Tensor, value: int):
    env.dummy1[env_ids] += value


def reset_dummy2_to_zero(env, env_ids: torch.Tensor):
    env.dummy2[env_ids] = 0


def increment_dummy2_by_one(env, env_ids: torch.Tensor):
    env.dummy2[env_ids] += 1


class reset_dummy2_to_zero_class(ManagerTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
    ) -> None:
        env.dummy2[env_ids] = 0


class increment_dummy2_by_one_class(ManagerTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
    ) -> None:
        env.dummy2[env_ids] += 1


@pytest.fixture
def env():
    num_envs = 32
    device = "cpu"
    # create dummy tensors
    dummy1 = torch.zeros((num_envs, 2), device=device)
    dummy2 = torch.zeros((num_envs, 10), device=device)
    # create sim
    sim = SimulationContext()
    # create dummy environment
    return DummyEnv(num_envs, 0.01, device, sim, dummy1, dummy2)


def test_str(env):
    """Test the string representation of the event manager."""
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
        "term_2": EventTermCfg(func=reset_dummy1_to_zero, mode="reset"),
        "term_3": EventTermCfg(func=change_dummy1_by_value, mode="custom", params={"value": 10}),
        "term_4": EventTermCfg(func=change_dummy1_by_value, mode="custom", params={"value": 2}),
    }
    event_man = EventManager(cfg, env)

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

        term_1: EventTermCfg = EventTermCfg(func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1))
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


def test_class_terms(env):
    """Test the correct preparation of function and class event terms."""
    cfg = {
        "term_1": EventTermCfg(func=reset_dummy2_to_zero, mode="reset"),
        "term_2": EventTermCfg(func=increment_dummy2_by_one_class, mode="interval", interval_range_s=(0.1, 0.1)),
        "term_3": EventTermCfg(func=reset_dummy2_to_zero_class, mode="reset"),
    }

    event_man = EventManager(cfg, env)
    assert len(event_man.active_terms) == 2
    assert len(event_man.active_terms["interval"]) == 1
    assert len(event_man.active_terms["reset"]) == 2
    assert len(event_man._mode_class_term_cfgs) == 2
    assert len(event_man._mode_class_term_cfgs["reset"]) == 1


def test_config_empty(env):
    """Test the creation of reward manager with empty config."""
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
        "term_2": EventTermCfg(func=reset_dummy1_to_zero, mode="reset"),
        "term_3": EventTermCfg(func=change_dummy1_by_value, mode="custom"),
    }
    with pytest.raises(ValueError):
        EventManager(cfg, env)


def test_apply_interval_mode_without_global_time(env):
    """Test the application of event terms that are in interval mode without global time.

    During local time, each environment instance has its own time for the interval term.
    """
    # make two intervals -- one is fixed and the other is random
    term_1_interval_range_s = (10 * env.dt, 10 * env.dt)
    term_2_interval_range_s = (2 * env.dt, 10 * env.dt)

    cfg = {
        "term_1": EventTermCfg(
            func=increment_dummy1_by_one,
            mode="interval",
            interval_range_s=term_1_interval_range_s,
            is_global_time=False,
        ),
        "term_2": EventTermCfg(
            func=increment_dummy2_by_one,
            mode="interval",
            interval_range_s=term_2_interval_range_s,
            is_global_time=False,
        ),
    }

    event_man = EventManager(cfg, env)

    # obtain the initial time left for the interval terms
    term_2_interval_time = event_man._interval_term_time_left[1].clone()
    expected_dummy2_value = torch.zeros_like(env.dummy2)

    for count in range(50):
        # apply the event terms
        event_man.apply("interval", dt=env.dt)
        # manually decrement the interval time for term2 since it is randomly sampled
        term_2_interval_time -= env.dt

        # check the values
        # we increment the dummy1 by 1 every 10 steps. at the 9th count (aka 10th apply), the value should be 1
        torch.testing.assert_close(env.dummy1, (count + 1) // 10 * torch.ones_like(env.dummy1))

        # we increment the dummy2 by 1 every 2 to 10 steps based on the random interval
        expected_dummy2_value += term_2_interval_time.unsqueeze(1) < 1e-6
        torch.testing.assert_close(env.dummy2, expected_dummy2_value)

        # check the time sampled at the end of the interval is valid
        # -- fixed interval
        if (count + 1) % 10 == 0:
            term_1_interval_time_init = event_man._interval_term_time_left[0].clone()
            expected_time_interval_init = torch.full_like(term_1_interval_time_init, term_1_interval_range_s[1])
            torch.testing.assert_close(term_1_interval_time_init, expected_time_interval_init)
        # -- random interval
        env_ids = (term_2_interval_time < 1e-6).nonzero(as_tuple=True)[0]
        if len(env_ids) > 0:
            term_2_interval_time[env_ids] = event_man._interval_term_time_left[1][env_ids]


def test_apply_interval_mode_with_global_time(env):
    """Test the application of event terms that are in interval mode with global time.

    During global time, all the environment instances share the same time for the interval term.
    """
    # make two intervals -- one is fixed and the other is random
    term_1_interval_range_s = (10 * env.dt, 10 * env.dt)
    term_2_interval_range_s = (2 * env.dt, 10 * env.dt)

    cfg = {
        "term_1": EventTermCfg(
            func=increment_dummy1_by_one,
            mode="interval",
            interval_range_s=term_1_interval_range_s,
            is_global_time=True,
        ),
        "term_2": EventTermCfg(
            func=increment_dummy2_by_one,
            mode="interval",
            interval_range_s=term_2_interval_range_s,
            is_global_time=True,
        ),
    }

    event_man = EventManager(cfg, env)

    # obtain the initial time left for the interval terms
    term_2_interval_time = event_man._interval_term_time_left[1].clone()
    expected_dummy2_value = torch.zeros_like(env.dummy2)

    for count in range(50):
        # apply the event terms
        event_man.apply("interval", dt=env.dt)
        # manually decrement the interval time for term2 since it is randomly sampled
        term_2_interval_time -= env.dt

        # check the values
        # we increment the dummy1 by 1 every 10 steps. at the 9th count (aka 10th apply), the value should be 1
        torch.testing.assert_close(env.dummy1, (count + 1) // 10 * torch.ones_like(env.dummy1))

        # we increment the dummy2 by 1 every 2 to 10 steps based on the random interval
        expected_dummy2_value += term_2_interval_time < 1e-6
        torch.testing.assert_close(env.dummy2, expected_dummy2_value)

        # check the time sampled at the end of the interval is valid
        # -- fixed interval
        if (count + 1) % 10 == 0:
            term_1_interval_time_init = event_man._interval_term_time_left[0].clone()
            expected_time_interval_init = torch.full_like(term_1_interval_time_init, term_1_interval_range_s[1])
            torch.testing.assert_close(term_1_interval_time_init, expected_time_interval_init)
        # -- random interval
        if term_2_interval_time < 1e-6:
            term_2_interval_time = event_man._interval_term_time_left[1].clone()


def test_apply_reset_mode(env):
    """Test the application of event terms that are in reset mode."""
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy1_by_one, mode="reset"),
        "term_2": EventTermCfg(func=reset_dummy1_to_zero, mode="reset", min_step_count_between_reset=10),
    }

    event_man = EventManager(cfg, env)

    # manually keep track of the expected values for dummy1 and trigger count
    expected_dummy1_value = torch.zeros_like(env.dummy1)
    term_2_trigger_step_id = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)

    for count in range(50):
        # apply the event terms for all the env ids
        if count % 3 == 0:
            event_man.apply("reset", global_env_step_count=count)

            # we increment the dummy1 by 1 every call to reset mode due to term 1
            expected_dummy1_value[:] += 1
            # manually update the expected value for term 2
            if (count - term_2_trigger_step_id[0]) >= 10 or count == 0:
                expected_dummy1_value = torch.zeros_like(env.dummy1)
                term_2_trigger_step_id[:] = count

        # check the values of trigger count
        # -- term 1
        expected_trigger_count = torch.full((env.num_envs,), 3 * (count // 3), dtype=torch.int32, device=env.device)
        torch.testing.assert_close(event_man._reset_term_last_triggered_step_id[0], expected_trigger_count)
        # -- term 2
        torch.testing.assert_close(event_man._reset_term_last_triggered_step_id[1], term_2_trigger_step_id)

        # check the values of dummy1
        torch.testing.assert_close(env.dummy1, expected_dummy1_value)


def test_apply_reset_mode_subset_env_ids(env):
    """Test the application of event terms that are in reset mode over a subset of environment ids."""
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy1_by_one, mode="reset"),
        "term_2": EventTermCfg(func=reset_dummy1_to_zero, mode="reset", min_step_count_between_reset=10),
    }

    event_man = EventManager(cfg, env)

    # since we are applying the event terms over a subset of env ids, we need to keep track of the trigger count
    # manually for the sake of testing
    term_2_trigger_step_id = torch.zeros((env.num_envs,), dtype=torch.int32, device=env.device)
    term_2_trigger_once = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    expected_dummy1_value = torch.zeros_like(env.dummy1)

    for count in range(50):
        # randomly select a subset of environment ids
        env_ids = (torch.rand(env.num_envs, device=env.device) < 0.5).nonzero().flatten()
        # apply the event terms for the selected env ids
        event_man.apply("reset", env_ids=env_ids, global_env_step_count=count)

        # modify the trigger count for term 2
        trigger_ids = (count - term_2_trigger_step_id[env_ids]) >= 10
        trigger_ids |= (term_2_trigger_step_id[env_ids] == 0) & ~term_2_trigger_once[env_ids]
        term_2_trigger_step_id[env_ids[trigger_ids]] = count
        term_2_trigger_once[env_ids[trigger_ids]] = True
        # we increment the dummy1 by 1 every call to reset mode
        # every 10th call, we reset the dummy1 to 0
        expected_dummy1_value[env_ids] += 1  # effect of term 1
        expected_dummy1_value[env_ids[trigger_ids]] = 0  # effect of term 2

        # check the values of trigger count
        # -- term 1
        expected_trigger_count = torch.full((len(env_ids),), count, dtype=torch.int32, device=env.device)
        torch.testing.assert_close(event_man._reset_term_last_triggered_step_id[0][env_ids], expected_trigger_count)
        # -- term 2
        torch.testing.assert_close(event_man._reset_term_last_triggered_step_id[1], term_2_trigger_step_id)

        # check the values of dummy1
        torch.testing.assert_close(env.dummy1, expected_dummy1_value)
