# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for viewer env resolution helpers."""

from __future__ import annotations

from isaaclab_visualizers.newton_adapter import apply_viewer_visible_worlds, resolve_visible_env_indices


def test_resolve_visible_env_indices_truncates_explicit_list():
    assert resolve_visible_env_indices([1, 3, 5], 2, 10) == [1, 3]
    assert resolve_visible_env_indices([1, 3], 1, 10) == [1]


def test_resolve_visible_env_indices_explicit_full_list_when_no_cap():
    assert resolve_visible_env_indices([1, 3], None, 10) == [1, 3]


def test_resolve_visible_env_indices_cap_when_no_filter():
    # When _compute_visualized_env_ids is None, cap is max_visible_envs.
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

    apply_viewer_visible_worlds(_V(), env_ids=None, max_visible_envs=2, num_envs=5)
    assert calls == [[0, 1]]

    apply_viewer_visible_worlds(_V(), env_ids=[2], max_visible_envs=99, num_envs=5)
    assert calls[-1] == [2]

    apply_viewer_visible_worlds(_V(), env_ids=None, max_visible_envs=None, num_envs=3)
    assert calls[-1] is None
