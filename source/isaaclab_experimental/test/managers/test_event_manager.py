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


@WarpCapturable(False, reason="synchronizes with the host")
def host_synchronizing_term(env, env_mask: wp.array):
    """Event term that cannot run inside a recorded stage."""


@pytest.mark.parametrize(("mode", "stages_capturable"), [("startup", True), ("reset", False)])
def test_non_capturable_term_keeps_event_stages_eager_only_when_it_runs_in_them(mode: str, stages_capturable: bool):
    """Startup terms run once while the environment is built, so they must not keep the reset stages eager."""
    graph_cache = WarpGraphCache("cpu")
    env = SimpleNamespace(
        num_envs=4, device="cpu", sim=SimpleNamespace(is_playing=lambda: True), _warp_graph_cache=graph_cache
    )

    EventManager({"term": EventTermCfg(func=host_synchronizing_term, mode=mode)}, env)

    assert graph_cache.is_capturable("EventManager") is stages_capturable
