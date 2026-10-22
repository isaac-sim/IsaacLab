# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Warp event manager."""

from collections import Counter
from types import SimpleNamespace

import pytest
import warp as wp
from isaaclab_experimental.managers import EventManager, EventTermCfg, ManagerTermBase
from isaaclab_experimental.utils import CapturedStage
from isaaclab_experimental.utils.warp import WarpCapturable

wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

DEVICE = "cuda:0"
NUM_ENVS = 4


CALLS: Counter[str] = Counter()


@WarpCapturable(False, reason="synchronizes with the host")
def host_synchronizing_term(env, env_mask: wp.array):
    """Event term that cannot run inside a recorded stage."""


def device_term(env, env_mask: wp.array):
    """Event term whose work is entirely on the device."""
    CALLS["device_term"] += 1


@WarpCapturable(False, reason="synchronizes with the host")
class host_call_term(ManagerTermBase):
    """Class event term whose call cannot be recorded but whose reset can."""

    def __call__(self, env, env_mask: wp.array):
        pass

    def reset(self, env_mask: wp.array | None = None):
        CALLS["host_call_term.reset"] += 1


class host_reset_term(ManagerTermBase):
    """Class event term whose reset cannot be recorded."""

    def __call__(self, env, env_mask: wp.array):
        pass

    @WarpCapturable(False, reason="synchronizes with the host")
    def reset(self, env_mask: wp.array | None = None):
        CALLS["host_reset_term.reset"] += 1


def test_startup_only_term_does_not_keep_the_reset_stage_eager(monkeypatch):
    """Startup terms run once while the environment is built, so they must not stop reset events from recording."""
    monkeypatch.setattr(CapturedStage, "enabled", True)
    env = SimpleNamespace(num_envs=NUM_ENVS, device=DEVICE, sim=SimpleNamespace(is_playing=lambda: True))
    manager = EventManager(
        {
            "startup_term": EventTermCfg(func=host_synchronizing_term, mode="startup"),
            "reset_term": EventTermCfg(func=device_term, mode="reset"),
        },
        env,
    )
    env_mask = wp.ones(NUM_ENVS, dtype=wp.bool, device=DEVICE)
    step_count = wp.zeros(1, dtype=wp.int32, device=DEVICE)
    CALLS.clear()

    for _ in range(2):
        manager.apply(mode="reset", env_mask_wp=env_mask, global_env_step_count=step_count)

    assert CALLS["device_term"] == 1, "the second call must replay the recorded reset events"
    assert manager._captured_stages[EventManager._apply_reset].num_graphs == 1


def test_class_annotation_does_not_keep_the_reset_eager(monkeypatch):
    """A class annotation covers the term's call; only an annotation on ``reset`` keeps the reset out of the graph."""
    monkeypatch.setattr(CapturedStage, "enabled", True)
    env = SimpleNamespace(num_envs=NUM_ENVS, device=DEVICE, sim=SimpleNamespace(is_playing=lambda: True))
    manager = EventManager(
        {
            "host_call_term": EventTermCfg(func=host_call_term, mode="startup"),
            "host_reset_term": EventTermCfg(func=host_reset_term, mode="startup"),
        },
        env,
    )
    env_mask = wp.ones(NUM_ENVS, dtype=wp.bool, device=DEVICE)
    CALLS.clear()

    for _ in range(2):
        manager.reset(env_mask=env_mask)

    assert CALLS["host_call_term.reset"] == 1, "the second reset must replay the recorded reset"
    assert CALLS["host_reset_term.reset"] == 2, "an annotated reset must run on every reset"
