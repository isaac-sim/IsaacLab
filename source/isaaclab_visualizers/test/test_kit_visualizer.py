# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Kit frame-boundary physics synchronization."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import Mock, call, patch

import numpy as np
import pytest
from isaaclab_visualizers.kit import KitVisualizer, KitVisualizerCfg
from isaaclab_visualizers.kit import kit_visualizer as kit_visualizer_module

import omni

from isaaclab.sim import SimulationContext

_PLAY_SIMULATIONS_SETTING = "/app/player/playSimulations"


def _make_visualizer() -> KitVisualizer:
    visualizer = KitVisualizer(KitVisualizerCfg(window_width=2, window_height=1))
    visualizer._is_initialized = True
    visualizer._update_camera_image_panel = Mock()
    visualizer._refresh_partial_viz_point_instancers_if_needed = Mock()
    visualizer.is_training_paused = Mock(return_value=False)
    return visualizer


@contextmanager
def _mock_kit_app(app: Mock) -> Iterator[None]:
    kit_module = ModuleType("omni.kit")
    kit_module.__path__ = []
    app_module = ModuleType("omni.kit.app")
    app_module.get_app = Mock(return_value=app)
    kit_module.app = app_module
    with (
        patch.object(omni, "kit", kit_module, create=True),
        patch.dict(sys.modules, {"omni.kit": kit_module, "omni.kit.app": app_module}),
    ):
        yield


@contextmanager
def _mock_replicator() -> Iterator[None]:
    replicator_module = ModuleType("omni.replicator")
    replicator_core_module = ModuleType("omni.replicator.core")
    replicator_module.core = replicator_core_module
    with patch.dict(
        sys.modules,
        {"omni.replicator": replicator_module, "omni.replicator.core": replicator_core_module},
    ):
        yield


def test_step_prepares_physics_immediately_before_app_update():
    visualizer = _make_visualizer()
    events = []
    sim = Mock()
    sim.physics_manager.before_kit_app_update.side_effect = lambda: events.append("prepare") or True
    sim.physics_manager.forward.side_effect = lambda: events.append("forward")
    app = Mock()
    app.is_running.return_value = True
    app.update.side_effect = lambda: events.append("update")
    settings = Mock()
    settings.get.return_value = True

    with (
        patch.object(SimulationContext, "instance", return_value=sim),
        patch.object(kit_visualizer_module, "get_settings_manager", return_value=settings),
        _mock_kit_app(app),
    ):
        visualizer.step(0.01)

    assert events == ["prepare", "forward", "update"]
    assert visualizer._app_pumped_this_step is True
    settings.set_bool.assert_has_calls([call(_PLAY_SIMULATIONS_SETTING, False), call(_PLAY_SIMULATIONS_SETTING, True)])


def test_render_rgb_array_prepares_physics_for_on_demand_update():
    visualizer = _make_visualizer()
    visualizer._app_pumped_this_step = False
    visualizer._rgb_annotator = Mock()
    visualizer._rgb_annotator.get_data.return_value = np.full((1, 2, 4), 255, dtype=np.uint8)
    events = []
    sim = Mock()
    sim.physics_manager.before_kit_app_update.side_effect = lambda: events.append("prepare") or True
    sim.physics_manager.forward.side_effect = lambda: events.append("forward")
    app = Mock()
    app.update.side_effect = lambda: events.append("update")
    settings = Mock()
    settings.get.return_value = True

    with (
        patch.object(SimulationContext, "instance", return_value=sim),
        patch.object(kit_visualizer_module, "get_settings_manager", return_value=settings),
        _mock_kit_app(app),
        _mock_replicator(),
    ):
        image = visualizer.render_rgb_array()

    assert events == ["prepare", "forward", "update"]
    assert image.shape == (1, 2, 3)
    settings.set_bool.assert_has_calls([call(_PLAY_SIMULATIONS_SETTING, False), call(_PLAY_SIMULATIONS_SETTING, True)])


def test_render_rgb_array_repumps_after_post_step_pose_write():
    visualizer = _make_visualizer()
    visualizer._app_pumped_this_step = True
    visualizer._rgb_annotator = Mock()
    visualizer._rgb_annotator.get_data.return_value = np.full((1, 2, 4), 255, dtype=np.uint8)
    events = []
    sim = Mock()
    sim.physics_manager.has_pending_kit_app_update.side_effect = [False, True]
    sim.physics_manager.before_kit_app_update.side_effect = lambda: events.append("prepare") or True
    sim.physics_manager.forward.side_effect = lambda: events.append("forward")
    app = Mock()
    app.update.side_effect = lambda: events.append("update")
    settings = Mock()
    settings.get.return_value = True

    with (
        patch.object(SimulationContext, "instance", return_value=sim),
        patch.object(kit_visualizer_module, "get_settings_manager", return_value=settings),
        _mock_kit_app(app),
        _mock_replicator(),
    ):
        clean_image = visualizer.render_rgb_array()
        assert events == []
        dirty_image = visualizer.render_rgb_array()

    assert clean_image.shape == (1, 2, 3)
    assert dirty_image.shape == (1, 2, 3)
    assert events == ["prepare", "forward", "update"]
    assert sim.physics_manager.has_pending_kit_app_update.call_count == 2


def test_kit_app_update_does_not_repeat_forward_after_clean_prepare():
    visualizer = _make_visualizer()
    sim = Mock()
    sim.physics_manager.before_kit_app_update.return_value = False
    app = Mock()
    settings = Mock()
    settings.get.return_value = True

    with (
        patch.object(SimulationContext, "instance", return_value=sim),
        patch.object(kit_visualizer_module, "get_settings_manager", return_value=settings),
    ):
        visualizer._update_kit_app(app)

    sim.physics_manager.before_kit_app_update.assert_called_once_with()
    sim.physics_manager.forward.assert_not_called()
    app.update.assert_called_once_with()


def test_camera_pose_sync_does_not_consume_request_without_running_app():
    visualizer = _make_visualizer()
    sim = Mock()
    app = Mock()
    app.is_running.return_value = False

    with patch.object(SimulationContext, "instance", return_value=sim), _mock_kit_app(app):
        visualizer._sync_camera_pose_updates_to_kit()

    sim.physics_manager.before_kit_app_update.assert_not_called()


def test_kit_app_update_restores_play_state_on_failure():
    visualizer = _make_visualizer()
    sim = Mock()
    sim.physics_manager.before_kit_app_update.return_value = False
    app = Mock()
    app.update.side_effect = RuntimeError("update failed")
    settings = Mock()
    settings.get.return_value = False

    with (
        patch.object(SimulationContext, "instance", return_value=sim),
        patch.object(kit_visualizer_module, "get_settings_manager", return_value=settings),
        pytest.raises(RuntimeError, match="update failed"),
    ):
        visualizer._update_kit_app(app)

    sim.physics_manager.before_kit_app_update.assert_called_once_with()
    sim.physics_manager.forward.assert_not_called()
    assert settings.set_bool.call_args_list == [
        call(_PLAY_SIMULATIONS_SETTING, False),
        call(_PLAY_SIMULATIONS_SETTING, False),
    ]
