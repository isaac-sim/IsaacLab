# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for RerunVisualizer/ViserVisualizer set_camera_view()."""

from __future__ import annotations

from types import SimpleNamespace

from isaaclab_visualizers.rerun import RerunVisualizer, RerunVisualizerCfg
from isaaclab_visualizers.viser import ViserVisualizer, ViserVisualizerCfg


def test_rerun_visualizer_set_camera_view_updates_viewer_pose():
    visualizer = RerunVisualizer(RerunVisualizerCfg())
    # _streaming_view_active=True short-circuits _apply_camera_pose before it calls into the
    # real rerun SDK (rr.send_blueprint), letting this test exercise the pose-conversion and
    # viewer-attribute-assignment logic without a live rerun session.
    visualizer._viewer = SimpleNamespace(_camera_pose=None)
    visualizer._streaming_view_active = True

    visualizer.set_camera_view([1, 2, 3], [4, 5, 6])

    assert visualizer._viewer._camera_pose == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))


def test_rerun_visualizer_set_camera_view_no_viewer_is_noop():
    visualizer = RerunVisualizer(RerunVisualizerCfg())
    assert visualizer._viewer is None

    visualizer.set_camera_view((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))  # must not raise


def test_viser_visualizer_set_camera_view_applies_immediately_when_client_ready(monkeypatch):
    visualizer = ViserVisualizer(ViserVisualizerCfg())
    visualizer._viewer = SimpleNamespace()
    monkeypatch.setattr(visualizer, "_try_apply_viser_camera_view", lambda pose: True)

    visualizer.set_camera_view([1, 2, 3], [0, 0, 0])

    assert visualizer._last_camera_pose == ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    assert visualizer._pending_camera_pose is None


def test_viser_visualizer_set_camera_view_defers_when_no_client_ready(monkeypatch):
    visualizer = ViserVisualizer(ViserVisualizerCfg())
    visualizer._viewer = SimpleNamespace()
    monkeypatch.setattr(visualizer, "_try_apply_viser_camera_view", lambda pose: False)

    visualizer.set_camera_view([1, 2, 3], [0, 0, 0])

    assert visualizer._pending_camera_pose == ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    assert visualizer._last_camera_pose is None
