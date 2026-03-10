# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for PhysxSceneDataProvider visualizer-facing contracts."""

from __future__ import annotations

from types import SimpleNamespace

from isaaclab_physx.scene_data_providers import PhysxSceneDataProvider

from isaaclab.physics.scene_data_requirements import VisualizerPrebuiltArtifacts


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


def test_try_use_prebuilt_artifact_populates_provider_state():
    """Provider should consume scene-time prebuilt artifact as fast path."""
    provider = _make_provider()
    artifact = VisualizerPrebuiltArtifacts(
        model="prebuilt-model",
        state="prebuilt-state",
        rigid_body_paths=["/World/envs/env_0/A"],
        articulation_paths=["/World/envs/env_0/Robot"],
        num_envs=4,
    )
    provider._simulation_context = SimpleNamespace(get_scene_data_visualizer_prebuilt_artifact=lambda: artifact)

    provider._xform_views = {"old": object()}
    provider._view_body_index_map = {"old": [1]}
    provider._view_order_tensors = {"old": object()}
    provider._pose_buf_num_bodies = 7
    provider._positions_buf = object()
    provider._orientations_buf = object()
    provider._covered_buf = object()
    provider._xform_mask_buf = object()
    provider._env_id_to_body_indices = {0: [0]}
    provider._filtered_newton_model = "old-filtered-model"
    provider._filtered_newton_state = "old-filtered-state"
    provider._filtered_env_ids_key = (0,)
    provider._filtered_body_indices = [0]

    assert provider._try_use_prebuilt_newton_artifact() is True
    assert provider._newton_model == "prebuilt-model"
    assert provider._newton_state == "prebuilt-state"
    assert provider._rigid_body_paths == ["/World/envs/env_0/A"]
    assert provider._articulation_paths == ["/World/envs/env_0/Robot"]
    assert provider._num_envs_at_last_newton_build == 4
    assert provider._xform_views == {}
    assert provider._view_body_index_map == {}
    assert provider._view_order_tensors == {}
    assert provider._pose_buf_num_bodies == 0
    assert provider._positions_buf is None
    assert provider._orientations_buf is None
    assert provider._covered_buf is None
    assert provider._xform_mask_buf is None
    assert provider._env_id_to_body_indices == {}
    assert provider._filtered_newton_model is None
    assert provider._filtered_newton_state is None
    assert provider._filtered_env_ids_key is None
    assert provider._filtered_body_indices == []


def test_try_use_prebuilt_artifact_respects_force_usd_fallback_flag():
    """Force flag should disable prebuilt fast path even when artifact is available."""
    provider = _make_provider()
    provider._force_usd_fallback = True
    artifact = VisualizerPrebuiltArtifacts(
        model="prebuilt-model",
        state="prebuilt-state",
        rigid_body_paths=["/World/envs/env_0/A"],
        articulation_paths=["/World/envs/env_0/Robot"],
        num_envs=4,
    )
    provider._simulation_context = SimpleNamespace(get_scene_data_visualizer_prebuilt_artifact=lambda: artifact)

    assert provider._try_use_prebuilt_newton_artifact() is False


def test_build_newton_model_from_usd_short_circuits_when_prebuilt_available():
    """If prebuilt artifact is available, USD fallback should not run."""
    provider = _make_provider()
    provider._try_use_prebuilt_newton_artifact = lambda: True
    provider._build_newton_model_from_usd()
    assert provider._last_newton_model_build_source == "prebuilt"
