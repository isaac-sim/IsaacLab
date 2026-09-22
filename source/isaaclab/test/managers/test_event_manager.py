# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

import pytest
import torch

from isaaclab.managers import EventManager, EventTermCfg, ManagerTermBase
from isaaclab.utils import configclass

pytestmark = pytest.mark.unit


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
    def __call__(self, env, env_ids: torch.Tensor) -> None:
        env.dummy2[env_ids] = 0


class increment_dummy2_by_one_class(ManagerTermBase):
    def __call__(self, env, env_ids: torch.Tensor) -> None:
        env.dummy2[env_ids] += 1


@pytest.fixture
def env(make_env):
    num_envs = 2
    return make_env(num_envs=num_envs, dummy1=torch.zeros((num_envs, 2)), dummy2=torch.zeros((num_envs, 10)))


def test_active_terms(env):
    """Terms are grouped by mode, class terms are tracked separately and listed in the string representation."""
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy2_by_one_class, mode="interval", interval_range_s=(0.1, 0.1)),
        "term_2": EventTermCfg(func=reset_dummy1_to_zero, mode="reset"),
        "term_3": EventTermCfg(func=reset_dummy2_to_zero_class, mode="reset"),
        "term_4": EventTermCfg(func=change_dummy1_by_value, mode="custom", params={"value": 10}),
        "term_5": EventTermCfg(func=change_dummy1_by_value, mode="custom", params={"value": 2}),
    }
    event_man = EventManager(cfg, env)

    assert event_man.active_terms == {
        "interval": ["term_1"],
        "reset": ["term_2", "term_3"],
        "custom": ["term_4", "term_5"],
    }
    assert event_man.available_modes == ["interval", "reset", "custom"]
    assert {mode: len(cfgs) for mode, cfgs in event_man._mode_class_term_cfgs.items()} == {
        "interval": 1,
        "reset": 1,
        "custom": 0,
    }
    event_man_str = str(event_man)
    assert "Active Event Terms in Mode: 'interval'" in event_man_str
    assert "term_5" in event_man_str

    empty_event_man = EventManager(None, env)
    assert len(empty_event_man.active_terms) == 0
    assert "contains 0 active terms" in str(empty_event_man)


def test_config_equivalence(env):
    """Dictionary, un-annotated and annotated configurations produce the same manager."""
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
        "term_2": EventTermCfg(func=reset_dummy1_to_zero, mode="reset"),
        "term_3": EventTermCfg(func=change_dummy1_by_value, mode="custom", params={"value": 10}),
    }

    @configclass
    class MyEventManagerCfg:
        term_1 = EventTermCfg(func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1))
        term_2 = EventTermCfg(func=reset_dummy1_to_zero, mode="reset")
        term_3 = EventTermCfg(func=change_dummy1_by_value, mode="custom", params={"value": 10})

    @configclass
    class MyEventManagerAnnotatedCfg:
        term_1: EventTermCfg = EventTermCfg(func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1))
        term_2: EventTermCfg = EventTermCfg(func=reset_dummy1_to_zero, mode="reset")
        term_3: EventTermCfg = EventTermCfg(func=change_dummy1_by_value, mode="custom", params={"value": 10})

    managers = [EventManager(c, env) for c in (cfg, MyEventManagerCfg(), MyEventManagerAnnotatedCfg())]
    for event_man in managers[1:]:
        assert event_man.active_terms == managers[0].active_terms
        assert event_man._mode_term_cfgs == managers[0]._mode_term_cfgs


@pytest.mark.parametrize(
    "term_2",
    [
        EventTermCfg(func="a:reset_dummy1_to_zero", mode="reset"),
        EventTermCfg(func=change_dummy1_by_value, mode="custom"),
    ],
    ids=["invalid_module", "missing_params"],
)
def test_invalid_event_config(env, term_2):
    """Unresolvable functions and unmatched parameters are rejected on construction."""
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy1_by_one, mode="interval", interval_range_s=(0.1, 0.1)),
        "term_2": term_2,
    }
    with pytest.raises(ValueError):
        EventManager(cfg, env)


@pytest.mark.parametrize("is_global_time", [False, True], ids=["per_env_time", "global_time"])
def test_apply_interval_mode(env, is_global_time):
    """Interval terms fire when their (fixed or random) interval elapses and are resampled afterwards.

    With global time all environments share one timer, otherwise each environment has its own.
    """
    term_1_interval_range_s = (10 * env.dt, 10 * env.dt)
    term_2_interval_range_s = (2 * env.dt, 10 * env.dt)
    cfg = {
        "term_1": EventTermCfg(
            func=increment_dummy1_by_one,
            mode="interval",
            interval_range_s=term_1_interval_range_s,
            is_global_time=is_global_time,
        ),
        "term_2": EventTermCfg(
            func=increment_dummy2_by_one,
            mode="interval",
            interval_range_s=term_2_interval_range_s,
            is_global_time=is_global_time,
        ),
    }
    event_man = EventManager(cfg, env)

    # track the random interval of term 2 manually
    term_2_interval_time = event_man._interval_term_time_left[1].clone()
    expected_dummy2_value = torch.zeros_like(env.dummy2)

    for count in range(50):
        event_man.apply("interval", dt=env.dt)
        term_2_interval_time -= env.dt

        # term 1 fires every 10 steps
        torch.testing.assert_close(env.dummy1, (count + 1) // 10 * torch.ones_like(env.dummy1))
        # term 2 fires every 2 to 10 steps based on the random interval
        fired = term_2_interval_time < 1e-6
        expected_dummy2_value += fired if is_global_time else fired.unsqueeze(1)
        torch.testing.assert_close(env.dummy2, expected_dummy2_value)

        # the fixed interval is resampled to the same value once it fires
        if (count + 1) % 10 == 0:
            torch.testing.assert_close(
                event_man._interval_term_time_left[0],
                torch.full_like(event_man._interval_term_time_left[0], term_1_interval_range_s[1]),
            )
        # pick up the resampled random interval
        if is_global_time:
            if fired:
                term_2_interval_time = event_man._interval_term_time_left[1].clone()
        else:
            term_2_interval_time[fired] = event_man._interval_term_time_left[1][fired]


def test_apply_interval_mode_resample_on_reset(env):
    """The interval timer is (not) resampled on reset based on ``resample_interval_on_reset``.

    A fixed interval makes the resampling deterministic: after one apply the timer reads ``interval - dt``,
    and on reset it is either restored to the interval or keeps counting down.
    """
    interval_s = 1.0  # large compared to env.dt so the terms do not fire during the test
    cfg = {
        "term_resample": EventTermCfg(
            func=increment_dummy1_by_one, mode="interval", interval_range_s=(interval_s, interval_s)
        ),
        "term_no_resample": EventTermCfg(
            func=increment_dummy2_by_one,
            mode="interval",
            interval_range_s=(interval_s, interval_s),
            resample_interval_on_reset=False,
        ),
    }
    event_man = EventManager(cfg, env)

    expected_init = torch.full((env.num_envs,), interval_s)
    expected_after_apply = expected_init - env.dt
    for time_left in event_man._interval_term_time_left:
        torch.testing.assert_close(time_left, expected_init)

    event_man.apply("interval", dt=env.dt)
    for time_left in event_man._interval_term_time_left:
        torch.testing.assert_close(time_left, expected_after_apply)

    event_man.reset(env_ids=torch.arange(env.num_envs))
    torch.testing.assert_close(event_man._interval_term_time_left[0], expected_init)
    torch.testing.assert_close(event_man._interval_term_time_left[1], expected_after_apply)


def test_apply_reset_mode(env):
    """Reset terms honor ``min_step_count_between_reset`` when applied to all environments."""
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy1_by_one, mode="reset"),
        "term_2": EventTermCfg(func=reset_dummy1_to_zero, mode="reset", min_step_count_between_reset=10),
    }
    event_man = EventManager(cfg, env)

    expected_dummy1_value = torch.zeros_like(env.dummy1)
    term_2_trigger_step_id = torch.zeros((env.num_envs,), dtype=torch.int32)

    for count in range(50):
        if count % 3 == 0:
            event_man.apply("reset", global_env_step_count=count)
            # term 1 increments on every reset call, term 2 zeroes every 10 steps (and on the first call)
            expected_dummy1_value += 1
            if (count - term_2_trigger_step_id[0]) >= 10 or count == 0:
                expected_dummy1_value = torch.zeros_like(env.dummy1)
                term_2_trigger_step_id[:] = count

        expected_trigger_count = torch.full((env.num_envs,), 3 * (count // 3), dtype=torch.int32)
        torch.testing.assert_close(event_man._reset_term_last_triggered_step_id[0], expected_trigger_count)
        torch.testing.assert_close(event_man._reset_term_last_triggered_step_id[1], term_2_trigger_step_id)
        torch.testing.assert_close(env.dummy1, expected_dummy1_value)


def test_apply_reset_mode_subset_env_ids(env):
    """Reset terms track the trigger step per environment when applied to a subset of environments."""
    cfg = {
        "term_1": EventTermCfg(func=increment_dummy1_by_one, mode="reset"),
        "term_2": EventTermCfg(func=reset_dummy1_to_zero, mode="reset", min_step_count_between_reset=10),
    }
    event_man = EventManager(cfg, env)

    term_2_trigger_step_id = torch.zeros((env.num_envs,), dtype=torch.int32)
    term_2_trigger_once = torch.zeros((env.num_envs,), dtype=torch.bool)
    expected_dummy1_value = torch.zeros_like(env.dummy1)

    for count in range(50):
        env_ids = (torch.rand(env.num_envs) < 0.5).nonzero().flatten()
        event_man.apply("reset", env_ids=env_ids, global_env_step_count=count)

        # term 2 triggers after 10 steps or if it never triggered before
        trigger_ids = (count - term_2_trigger_step_id[env_ids]) >= 10
        trigger_ids |= (term_2_trigger_step_id[env_ids] == 0) & ~term_2_trigger_once[env_ids]
        term_2_trigger_step_id[env_ids[trigger_ids]] = count
        term_2_trigger_once[env_ids[trigger_ids]] = True
        expected_dummy1_value[env_ids] += 1
        expected_dummy1_value[env_ids[trigger_ids]] = 0

        expected_trigger_count = torch.full((len(env_ids),), count, dtype=torch.int32)
        torch.testing.assert_close(event_man._reset_term_last_triggered_step_id[0][env_ids], expected_trigger_count)
        torch.testing.assert_close(event_man._reset_term_last_triggered_step_id[1], term_2_trigger_step_id)
        torch.testing.assert_close(env.dummy1, expected_dummy1_value)
