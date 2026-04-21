# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for PhysxSceneDataProvider visualizer-facing contracts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from isaaclab_physx.scene_data_providers import PhysxSceneDataProvider

from isaaclab.physics.scene_data_requirements import VisualizerPrebuiltArtifacts


def _make_provider():
    return object.__new__(PhysxSceneDataProvider)


def test_get_newton_model_returns_model_when_sync_enabled():
    """Callers receive the full Newton model from :meth:`get_newton_model`."""
    provider = _make_provider()
    provider._needs_newton_sync = True
    provider._newton_model = "full-model"

    assert provider.get_newton_model() == "full-model"


@patch("isaaclab_physx.scene_data_providers.physx_scene_data_provider.replace_newton_shape_colors", lambda m, s: None)
def test_load_prebuilt_artifact_populates_provider_state():
    """Loading the prebuilt artifact sets model, state, and rigid-body paths."""
    provider = _make_provider()
    artifact = VisualizerPrebuiltArtifacts(
        model="prebuilt-model",
        state="prebuilt-state",
        rigid_body_paths=["/World/envs/env_0/A"],
        articulation_paths=["/World/envs/env_0/Robot"],
        num_envs=4,
    )
    provider._simulation_context = SimpleNamespace(get_scene_data_visualizer_prebuilt_artifact=lambda: artifact)
    provider._stage = None

    provider._xform_views = {"old": object()}
    provider._view_body_index_map = {"old": [1]}
    provider._view_order_tensors = {"old": object()}
    provider._pose_buf_num_bodies = 7
    provider._positions_buf = object()
    provider._orientations_buf = object()
    provider._covered_buf = object()
    provider._xform_mask_buf = object()
    provider._load_newton_model_from_prebuilt_artifact()
    assert provider._newton_model == "prebuilt-model"
    assert provider._newton_state == "prebuilt-state"
    assert provider._rigid_body_paths == ["/World/envs/env_0/A"]
    assert provider._rigid_body_view_paths == ["/World/envs/env_0/A", "/World/envs/env_0/Robot"]
    assert provider._num_envs_at_last_newton_build == 4
    assert provider._last_newton_model_build_source == "prebuilt"
    assert provider._xform_views == {}
    assert provider._view_body_index_map == {}
    assert provider._view_order_tensors == {}
    assert provider._pose_buf_num_bodies == 0
    assert provider._positions_buf is None
    assert provider._orientations_buf is None
    assert provider._covered_buf is None
    assert provider._xform_mask_buf is None


def test_load_prebuilt_artifact_missing_sets_error_state():
    """When no artifact is registered, model/state stay unset."""
    provider = _make_provider()
    provider._simulation_context = SimpleNamespace(get_scene_data_visualizer_prebuilt_artifact=lambda: None)
    provider._load_newton_model_from_prebuilt_artifact()
    assert provider._last_newton_model_build_source == "missing"
    assert provider._newton_model is None
    assert provider._newton_state is None
