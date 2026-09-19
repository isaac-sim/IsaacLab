# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the event manager."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import EventManager, EventTermCfg

pytestmark = pytest.mark.unit


def make_event_manager(terms: list[EventTermCfg]) -> EventManager:
    """Create an event manager with initialized reset bookkeeping."""
    manager = EventManager.__new__(EventManager)
    manager._resolve_terms_handle = None
    manager._is_scene_entities_resolved = True
    manager._mode_term_names = {"reset": [f"term_{index}" for index in range(len(terms))]}
    manager._mode_term_cfgs = {"reset": terms}
    manager._reset_term_last_triggered_step_id = [torch.zeros(3, dtype=torch.int32) for _ in terms]
    manager._reset_term_last_triggered_once = [torch.zeros(3, dtype=torch.bool) for _ in terms]
    manager._env = SimpleNamespace(num_envs=3)
    return manager


def test_apply_reset_mode_preserves_none_for_unfiltered_terms():
    """All-environment reset terms receive None while bookkeeping uses every row."""
    received_env_ids: list[torch.Tensor | None] = []

    def record_env_ids(env, env_ids: torch.Tensor | None):
        received_env_ids.append(env_ids)

    manager = make_event_manager(
        [
            EventTermCfg(func=record_env_ids, mode="reset"),
            EventTermCfg(func=record_env_ids, mode="reset"),
        ]
    )

    manager.apply("reset", env_ids=None, global_env_step_count=4)

    assert received_env_ids == [None, None]
    for triggered_steps, triggered_once in zip(
        manager._reset_term_last_triggered_step_id, manager._reset_term_last_triggered_once
    ):
        torch.testing.assert_close(triggered_steps, torch.full((3,), 4, dtype=torch.int32))
        assert torch.all(triggered_once)


def test_apply_reset_mode_resolves_none_for_filtered_terms():
    """Reset terms with trigger filtering receive the eligible environment IDs."""
    received_env_ids: list[torch.Tensor] = []

    def record_env_ids(env, env_ids: torch.Tensor):
        received_env_ids.append(env_ids)

    manager = make_event_manager([EventTermCfg(func=record_env_ids, mode="reset", min_step_count_between_reset=5)])

    manager.apply("reset", env_ids=None, global_env_step_count=4)

    assert len(received_env_ids) == 1
    torch.testing.assert_close(received_env_ids[0], torch.arange(3))
