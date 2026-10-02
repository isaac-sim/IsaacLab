# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Warp event manager."""

from types import SimpleNamespace

import pytest
import warp as wp
from isaaclab_experimental.managers import EventManager, EventTermCfg
from isaaclab_experimental.utils import WarpGraphCache
from isaaclab_experimental.utils.warp import WarpCapturable

wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

DEVICE = "cuda:0"
NUM_ENVS = 4


@WarpCapturable(False, reason="synchronizes with the host")
def host_synchronizing_term(env, env_mask: wp.array):
    """Event term that cannot run inside a recorded stage."""


def device_term(env, env_mask: wp.array):
    """Event term whose work is entirely on the device."""


def select(env, env_mask: wp.array):
    """Event term that stores which environments it was applied to."""
    env.recordings += 1
    wp.copy(env.selected, env_mask)


def test_startup_only_term_does_not_keep_the_reset_stage_eager():
    """Startup terms run once while the environment is built, so they must not stop reset events from recording."""
    graph_cache = WarpGraphCache(DEVICE)
    env = SimpleNamespace(
        num_envs=NUM_ENVS, device=DEVICE, sim=SimpleNamespace(is_playing=lambda: True), _warp_graph_cache=graph_cache
    )
    manager = EventManager(
        {
            "startup_term": EventTermCfg(func=host_synchronizing_term, mode="startup"),
            "reset_term": EventTermCfg(func=device_term, mode="reset"),
        },
        env,
    )
    graph_cache.arm()

    graph_cache.call(
        "EventManager_apply_reset",
        manager.apply,
        mode="reset",
        env_mask_wp=wp.ones(NUM_ENVS, dtype=wp.bool, device=DEVICE),
        global_env_step_count=wp.zeros(1, dtype=wp.int32, device=DEVICE),
    )

    assert graph_cache.captured_stages == ("EventManager_apply_reset",)
    graph_cache.close()


def test_set_term_cfg_records_the_reset_events_again_only_when_the_trigger_changes():
    """A curriculum sets its event term on every reset; only a changed trigger spacing records the stage again.

    The spacing is a kernel argument of the recorded stage, so the changed spacing applies only after a new
    recording. Every recording runs the term's Python once.
    """
    graph_cache = WarpGraphCache(DEVICE)
    env = SimpleNamespace(
        num_envs=NUM_ENVS,
        device=DEVICE,
        sim=SimpleNamespace(is_playing=lambda: True),
        selected=wp.zeros(NUM_ENVS, dtype=wp.bool, device=DEVICE),
        recordings=0,
        _warp_graph_cache=graph_cache,
    )
    manager = EventManager({"select": EventTermCfg(func=select, mode="reset")}, env)
    env_mask = wp.ones(NUM_ENVS, dtype=wp.bool, device=DEVICE)
    step_count = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    term_cfg = manager.get_term_cfg("select")
    graph_cache.arm()

    def apply_reset(step: int, min_step_count_between_reset: int) -> list[bool]:
        term_cfg.min_step_count_between_reset = min_step_count_between_reset
        manager.set_term_cfg("select", term_cfg)
        step_count.fill_(step)
        graph_cache.call_steps(
            "EventManager_apply_reset",
            manager.stage_steps("apply_reset"),
            env_mask_wp=env_mask,
            global_env_step_count=step_count,
        )
        return env.selected.numpy().tolist()

    assert apply_reset(step=5, min_step_count_between_reset=0) == [True] * NUM_ENVS
    assert apply_reset(step=6, min_step_count_between_reset=0) == [True] * NUM_ENVS
    assert env.recordings == 1
    assert apply_reset(step=7, min_step_count_between_reset=10) == [False] * NUM_ENVS
    assert env.recordings == 2
    graph_cache.close()
