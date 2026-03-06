# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for PhysxSceneDataProvider visualizer-facing contracts."""

from __future__ import annotations

from isaaclab_physx.scene_data_providers import PhysxSceneDataProvider


def _make_provider():
    provider = object.__new__(PhysxSceneDataProvider)
    return provider


def test_get_newton_model_for_env_ids_builds_and_caches_sorted_keys():
    provider = _make_provider()
    provider._needs_newton_sync = True
    provider._newton_model = "full-model"
    provider._filtered_newton_model = None
    provider._filtered_env_ids_key = None

    build_calls = []

    def _fake_build(env_ids):
        build_calls.append(env_ids)
        provider._filtered_newton_model = f"filtered-{env_ids}"

    provider._build_filtered_newton_model = _fake_build

    # None asks for the full model.
    assert provider.get_newton_model_for_env_ids(None) == "full-model"

    # First subset request builds using sorted env id key.
    model_a = provider.get_newton_model_for_env_ids([3, 1])
    assert model_a == "filtered-[1, 3]"
    assert build_calls == [[1, 3]]

    # Equivalent request should use cache and not rebuild.
    model_b = provider.get_newton_model_for_env_ids([1, 3])
    assert model_b == "filtered-[1, 3]"
    assert build_calls == [[1, 3]]

    # Different subset rebuilds.
    model_c = provider.get_newton_model_for_env_ids([2])
    assert model_c == "filtered-[2]"
    assert build_calls == [[1, 3], [2]]
