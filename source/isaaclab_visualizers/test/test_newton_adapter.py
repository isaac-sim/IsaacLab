# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for viewer env resolution helpers."""

from __future__ import annotations

from isaaclab_visualizers.newton_adapter import apply_viewer_visible_worlds, resolve_visible_env_indices


def test_resolve_visible_env_indices_env_ids_win():
    assert resolve_visible_env_indices([1, 3], 1, 10) == [1, 3]


def test_resolve_visible_env_indices_cap_when_no_filter():
    # When _compute_visualized_env_ids is None (e.g. mode ``none``), cap is env_selection_max_visible, not random_count.
    assert resolve_visible_env_indices(None, 3, 10) == [0, 1, 2]


def test_resolve_visible_env_indices_all_when_no_cap():
    assert resolve_visible_env_indices(None, None, 10) is None


def test_resolve_visible_env_indices_num_envs_zero_falls_through_like_newton():
    assert resolve_visible_env_indices(None, 5, 0) is None


def test_apply_viewer_visible_worlds_delegates_to_resolved():
    calls: list = []

    class _V:
        def set_visible_worlds(self, worlds):
            calls.append(worlds)

    apply_viewer_visible_worlds(_V(), env_ids=None, env_selection_max_visible=2, num_envs=5)
    assert calls == [[0, 1]]

    apply_viewer_visible_worlds(_V(), env_ids=[2], env_selection_max_visible=99, num_envs=5)
    assert calls[-1] == [2]

    apply_viewer_visible_worlds(_V(), env_ids=None, env_selection_max_visible=None, num_envs=3)
    assert calls[-1] is None
