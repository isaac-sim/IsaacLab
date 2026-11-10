# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Rerun/Viser construction and camera controls."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_visualizers.rerun import RerunBackendCfg
from isaaclab_visualizers.rerun.rerun_visualizer import NewtonViewerRerun
from isaaclab_visualizers.viser import ViserBackendCfg
from isaaclab_visualizers.viser.viser_visualizer import NewtonViewerViser
from newton.viewer import ViewerRerun

from isaaclab.visualizers import VisualizerCfg


def test_viser_missing_dependency_propagates_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "viser", None)
    monkeypatch.setattr(NewtonViewerViser, "_viser_module", None)
    with pytest.raises(ModuleNotFoundError) as error:
        NewtonViewerViser()
    assert error.value.name == "viser"


def test_rerun_visualizer_set_camera_view():
    # No viewer: must not raise.
    cfg = VisualizerCfg(backend=RerunBackendCfg())
    visualizer = cfg.backend.class_type(cfg)
    visualizer.set_camera_view((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

    # A selected scene camera short-circuits _apply_camera_pose before it calls into the
    # real rerun SDK (rr.send_blueprint), letting this test exercise the pose-conversion and
    # viewer-attribute-assignment logic without a live rerun session.
    visualizer._viewer = SimpleNamespace(_camera_pose=None)
    visualizer.current_image_view = SimpleNamespace(camera=object())
    visualizer.set_camera_view([1, 2, 3], [4, 5, 6])

    assert visualizer._viewer._camera_pose == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))


def test_rerun_viewer_skips_unchanged_instance_appearance(monkeypatch):
    viewer = NewtonViewerRerun.__new__(NewtonViewerRerun)
    viewer._instances = {}
    viewer._instance_appearance = {}
    viewer._qualify = lambda name: name
    calls = []

    def _log_instances(_self, name, mesh, xforms, scales, colors, materials, hidden=False, opacities=None):
        calls.append(SimpleNamespace(colors=colors, opacities=opacities))
        if hidden:
            viewer._instances.pop(name, None)
        else:
            viewer._instances[name] = {}

    monkeypatch.setattr(ViewerRerun, "log_instances", _log_instances)
    colors = wp.array([(0.2, 0.4, 0.6)], dtype=wp.vec3, device="cpu")
    opacities = wp.array([0.8], dtype=wp.float32, device="cpu")

    viewer.log_instances("/robot", "/mesh", None, None, colors, None, opacities=opacities)
    viewer.log_instances("/robot", "/mesh", None, None, colors, None, opacities=opacities)

    assert calls[0].colors is colors
    assert calls[0].opacities is opacities
    assert calls[1].colors is None
    assert calls[1].opacities is None

    changed_colors = wp.array([(0.7, 0.4, 0.6)], dtype=wp.vec3, device="cpu")
    viewer.log_instances("/robot", "/mesh", None, None, changed_colors, None, opacities=opacities)
    np.testing.assert_array_equal(calls[2].colors.numpy(), changed_colors.numpy())
    assert calls[2].opacities is None


def test_viser_visualizer_set_camera_view():
    cfg = VisualizerCfg(backend=ViserBackendCfg())
    visualizer = cfg.backend.class_type(cfg)
    visualizer.set_camera_view([1, 2, 3], [0, 0, 0])
    assert visualizer._pending_camera_pose == ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))

    clients = {}
    visualizer._viewer = SimpleNamespace(_server=SimpleNamespace(get_clients=lambda: clients.copy()))
    visualizer.set_camera_view([4, 5, 6], [0, 0, 0])
    assert visualizer._pending_camera_pose == ((4.0, 5.0, 6.0), (0.0, 0.0, 0.0))

    # A pose requested before clients connect reaches their cameras on the next step, even while paused.
    clients.update({index: SimpleNamespace(camera=SimpleNamespace()) for index in range(2)})
    visualizer._is_initialized = True
    visualizer._sim = object()
    visualizer.backend = SimpleNamespace(model=SimpleNamespace(world_count=1))
    visualizer._paused_rendering = True
    visualizer.step(0.1)
    assert visualizer._pending_camera_pose is None
    for client in clients.values():
        assert client.camera.position == (4.0, 5.0, 6.0)
        assert client.camera.look_at == (0.0, 0.0, 0.0)
        assert 0 < client.camera.fov < 3.1416

    visualizer.set_camera_view([7, 8, 9], [1, 2, 3])
    for client in clients.values():
        assert client.camera.position == (7.0, 8.0, 9.0)
        assert client.camera.look_at == (1.0, 2.0, 3.0)
