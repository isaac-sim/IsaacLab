# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regressions for viewport frame reuse after scene-data mutations."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from isaaclab_visualizers.kit import KitBackendCfg, KitVisualizer
from isaaclab_visualizers.kit import kit_visualizer as kit_visualizer_module

from isaaclab.visualizers import VisualizerCfg, WindowCfg

pytestmark = pytest.mark.unit


@pytest.fixture
def viewport(monkeypatch):
    """Model a renderer whose annotator only changes when Kit produces a frame."""
    backend = SimpleNamespace(transforms_timestamp=0)
    provider = SimpleNamespace(backend=backend)
    sim = Mock(is_rendering=True, render_generation=1)
    sim.get_scene_data_provider.return_value = provider
    visualizer = KitVisualizer(VisualizerCfg(backend=KitBackendCfg(), window=WindowCfg(size=(2, 1))))
    visualizer._is_initialized = True
    visualizer._sim = sim
    fabric = SimpleNamespace(
        rendered_transforms_timestamp=-1,
        update_transforms=Mock(side_effect=lambda scene_data: scene_data.backend.transforms_timestamp),
        update_geometries=Mock(),
    )
    visualizer._fabric = fabric
    monkeypatch.setattr(visualizer, "_refresh_partial_viz_point_instancers_if_needed", Mock())
    frame = np.zeros((1, 2, 4), dtype=np.uint8)
    visualizer._rgb_annotator = Mock()
    visualizer._rgb_annotator.get_data.side_effect = lambda: frame.copy()

    app = Mock()
    app.is_running.return_value = True
    app.update.side_effect = lambda: frame.fill(backend.transforms_timestamp)
    omni = sys.modules.get("omni")
    if omni is None:
        omni = ModuleType("omni")
        monkeypatch.setitem(sys.modules, "omni", omni)
    kit_module = ModuleType("omni.kit")
    app_module = ModuleType("omni.kit.app")
    app_module.get_app = lambda: app
    kit_module.app = app_module
    monkeypatch.setitem(sys.modules, "omni.kit", kit_module)
    monkeypatch.setitem(sys.modules, "omni.kit.app", app_module)
    monkeypatch.setattr(omni, "kit", kit_module, raising=False)
    replicator_module = ModuleType("omni.replicator")
    replicator_module.core = ModuleType("omni.replicator.core")
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator_module)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", replicator_module.core)
    monkeypatch.setattr(omni, "replicator", replicator_module, raising=False)
    settings = Mock()
    settings.get.return_value = True
    monkeypatch.setattr(kit_visualizer_module, "get_settings_manager", lambda: settings)
    return SimpleNamespace(visualizer=visualizer, backend=backend, app=app, frame=frame)


def test_capture_reuses_clean_frame_but_refreshes_same_step_pose_write(viewport):
    """A pose write after step() must appear without another simulation or UI step."""
    viewport.visualizer.step(0.01)
    np.testing.assert_array_equal(viewport.visualizer.render_rgb_array(), np.zeros((1, 2, 3), dtype=np.uint8))
    viewport.app.update.assert_called_once_with()

    viewport.backend.transforms_timestamp += 1
    np.testing.assert_array_equal(viewport.visualizer.render_rgb_array(), np.ones((1, 2, 3), dtype=np.uint8))
    assert viewport.app.update.call_count == 2
    np.testing.assert_array_equal(viewport.visualizer.render_rgb_array(), np.ones((1, 2, 3), dtype=np.uint8))
    assert viewport.app.update.call_count == 2


def test_capture_retries_same_step_pose_publication_after_failed_update(viewport):
    """A failed app update must not make the old annotator buffer look current."""
    viewport.visualizer.step(0.01)
    viewport.backend.transforms_timestamp += 1
    viewport.app.update.side_effect = RuntimeError("frame update failed")

    with pytest.raises(RuntimeError, match="frame update failed"):
        viewport.visualizer.render_rgb_array()

    viewport.app.update.side_effect = lambda: viewport.frame.fill(viewport.backend.transforms_timestamp)
    np.testing.assert_array_equal(viewport.visualizer.render_rgb_array(), np.ones((1, 2, 3), dtype=np.uint8))
    assert viewport.app.update.call_count == 3


def test_capture_does_not_consume_pose_write_from_inside_app_update(viewport):
    """Writes made after a frame starts remain visible to the next capture request."""
    viewport.visualizer.step(0.01)
    viewport.backend.transforms_timestamp += 1

    def update_then_write():
        viewport.frame.fill(viewport.backend.transforms_timestamp)
        viewport.backend.transforms_timestamp += 1

    viewport.app.update.side_effect = update_then_write
    np.testing.assert_array_equal(viewport.visualizer.render_rgb_array(), np.ones((1, 2, 3), dtype=np.uint8))
    viewport.app.update.side_effect = lambda: viewport.frame.fill(viewport.backend.transforms_timestamp)
    np.testing.assert_array_equal(viewport.visualizer.render_rgb_array(), np.full((1, 2, 3), 2, dtype=np.uint8))
    assert viewport.app.update.call_count == 3


def test_headless_capture_repumps_paused_product_after_camera_change(viewport):
    """A shared scene stamp cannot make a paused, camera-specific render product current."""
    viewport.visualizer._runtime_headless = True
    render_product = viewport.visualizer._rgb_render_product = Mock()
    camera_value = 7
    viewport.app.update.side_effect = lambda: viewport.frame.fill(camera_value)
    viewport.visualizer.step(0.01)
    viewport.app.update.assert_not_called()

    np.testing.assert_array_equal(viewport.visualizer.render_rgb_array(), np.full((1, 2, 3), 7, dtype=np.uint8))
    render_product.pause.assert_called_once_with()

    # Another RTX consumer rendered the same body poses while this product was paused.
    viewport.visualizer._fabric.rendered_transforms_timestamp = viewport.backend.transforms_timestamp
    camera_value = 19
    np.testing.assert_array_equal(viewport.visualizer.render_rgb_array(), np.full((1, 2, 3), 19, dtype=np.uint8))

    assert viewport.app.update.call_count == 2
    assert render_product.resume.call_count == 2
    assert render_product.pause.call_count == 2


@pytest.mark.parametrize("failure", ["exception", "not_running"])
def test_initial_annotator_capture_retries_after_failed_frame(viewport, monkeypatch, failure):
    """Creating an annotator leaves a frame obligation until its own update succeeds."""
    viewport.visualizer.step(0.01)
    annotator = viewport.visualizer._rgb_annotator
    viewport.visualizer._rgb_annotator = None
    render_product = Mock(path="/Render/ViewportCapture")
    replicator = sys.modules["omni.replicator.core"]
    monkeypatch.setattr(
        replicator, "create", SimpleNamespace(render_product=Mock(return_value=render_product)), raising=False
    )
    monkeypatch.setattr(
        replicator, "AnnotatorRegistry", SimpleNamespace(get_annotator=Mock(return_value=annotator)), raising=False
    )
    monkeypatch.setattr(viewport.visualizer, "_apply_render_product_background", Mock())

    if failure == "exception":
        viewport.app.update.side_effect = RuntimeError("initial frame failed")
        with pytest.raises(RuntimeError, match="initial frame failed"):
            viewport.visualizer.render_rgb_array()
    else:
        viewport.app.is_running.return_value = False
        np.testing.assert_array_equal(viewport.visualizer.render_rgb_array(), np.zeros((1, 2, 3), dtype=np.uint8))

    updates_before_retry = viewport.app.update.call_count
    viewport.app.is_running.return_value = True
    viewport.app.update.side_effect = lambda: viewport.frame.fill(7)
    np.testing.assert_array_equal(viewport.visualizer.render_rgb_array(), np.full((1, 2, 3), 7, dtype=np.uint8))
    assert viewport.app.update.call_count == updates_before_retry + 1

    np.testing.assert_array_equal(viewport.visualizer.render_rgb_array(), np.full((1, 2, 3), 7, dtype=np.uint8))
    assert viewport.app.update.call_count == updates_before_retry + 1
    annotator.attach.assert_called_once_with([render_product])
