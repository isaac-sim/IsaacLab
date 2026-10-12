# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Rerun/Viser construction and camera controls."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from isaaclab_visualizers.rerun import RerunVisualizer, RerunVisualizerCfg
from isaaclab_visualizers.viser import ViserVisualizer, ViserVisualizerCfg
from isaaclab_visualizers.viser.viser_visualizer import NewtonViewerViser


def test_viser_missing_dependency_propagates_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "viser", None)
    monkeypatch.setattr(NewtonViewerViser, "_viser_module", None)
    with pytest.raises(ModuleNotFoundError) as error:
        NewtonViewerViser()
    assert error.value.name == "viser"


def test_rerun_visualizer_set_camera_view():
    # No viewer: must not raise.
    visualizer = RerunVisualizer(RerunVisualizerCfg())
    visualizer.set_camera_view((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

    # A selected scene camera short-circuits _apply_camera_pose before it calls into the
    # real rerun SDK (rr.send_blueprint), letting this test exercise the pose-conversion and
    # viewer-attribute-assignment logic without a live rerun session.
    visualizer._viewer = SimpleNamespace(_camera_pose=None)
    visualizer.current_image_view = SimpleNamespace(camera=object())
    visualizer.set_camera_view([1, 2, 3], [4, 5, 6])

    assert visualizer._viewer._camera_pose == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))


def test_viser_visualizer_set_camera_view():
    visualizer = ViserVisualizer(ViserVisualizerCfg())
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
    visualizer.backend = SimpleNamespace(model=SimpleNamespace(num_envs=1))
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
