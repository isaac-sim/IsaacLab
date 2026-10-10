# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for RerunVisualizer/ViserVisualizer set_camera_view()."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import rerun as rr
import rerun.blueprint as rrb
import rerun_bindings
from isaaclab_visualizers.rerun import RerunVisualizer, RerunVisualizerCfg
from isaaclab_visualizers.rerun import rerun_visualizer as rerun_visualizer_module
from isaaclab_visualizers.viser import ViserVisualizer, ViserVisualizerCfg


def test_rerun_visualizer_set_camera_view():
    # No viewer: must not raise.
    visualizer = RerunVisualizer(RerunVisualizerCfg())
    visualizer.set_camera_view((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

    # A selected scene camera short-circuits _apply_camera_pose before it calls into the
    # real rerun SDK (rr.send_blueprint), letting this test exercise the pose-conversion and
    # viewer-attribute-assignment logic without a live rerun session.
    visualizer._viewer = SimpleNamespace(_camera_pose=None)
    visualizer._camera_sensor = object()
    visualizer.set_camera_view([1, 2, 3], [4, 5, 6])

    assert visualizer._viewer._camera_pose == ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))


def test_rerun_visualizer_caps_updates_by_wall_time(monkeypatch):
    visualizer = RerunVisualizer(RerunVisualizerCfg(max_fps=20.0))
    times = iter((10.0, 10.01, 10.05))
    monkeypatch.setattr(rerun_visualizer_module.time, "monotonic", lambda: next(times))

    assert visualizer._is_render_due()
    assert not visualizer._is_render_due()
    assert visualizer._is_render_due()


@pytest.mark.parametrize("max_fps", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_rerun_visualizer_rejects_invalid_max_fps(max_fps):
    assert RerunVisualizerCfg().max_fps == 60.0

    with pytest.raises(ValueError, match="max_fps must be finite and positive or None"):
        RerunVisualizerCfg(max_fps=max_fps)


def test_rerun_rate_limit_preserves_streaming_and_plot_cadence(monkeypatch):
    visualizer = _initialized_rerun_visualizer(RerunVisualizerCfg(max_fps=60.0, live_plots_update_interval=1))
    visualizer._live_plot_sources = [object()]
    render_live_plots = MagicMock()
    monkeypatch.setattr(visualizer, "_render_live_plots", render_live_plots)
    publication_due = iter((False, True))
    monkeypatch.setattr(visualizer, "_is_render_due", lambda: next(publication_due))

    visualizer.step(0.01)
    visualizer._viewer.begin_frame.assert_not_called()
    assert visualizer._live_plots_pending

    visualizer.step(0.01)
    assert visualizer.render_tiled_rgb_array.call_count == 2
    visualizer._viewer.begin_frame.assert_called_once()
    render_live_plots.assert_called_once_with(visualizer._viewer)
    assert not visualizer._live_plots_pending


def test_rerun_recording_preserves_frames_without_bypassing_live_rate_limit(tmp_path, monkeypatch):
    recording_path = tmp_path / "recording.rrd"
    viewers = []

    class _RecordingViewer:
        def __init__(self, *, app_id, rec_id=None, record_to_rrd=None, **_kwargs):
            self.stream = rr.RecordingStream(app_id, recording_id=rec_id)
            rr.set_global_data_recording(self.stream)
            if record_to_rrd is not None:
                self.stream.save(record_to_rrd)
            self.live_sink = self.stream.binary_stream()
            self.begin_frame_calls = 0
            self._camera_pose = None
            viewers.append(self)

        def _get_blueprint(self):
            return rrb.Blueprint(rrb.Spatial3DView())

        def begin_frame(self, sim_time):
            self.begin_frame_calls += 1
            rr.set_time("time", timestamp=sim_time)

        def log_state(self, _state):
            rr.log("test/frame", rr.Scalars(self.begin_frame_calls))

        def end_frame(self):
            pass

        def is_paused(self):
            return False

        def set_model(self, _model):
            pass

        def set_visible_worlds(self, _worlds):
            pass

        def set_world_offsets(self, _offsets):
            pass

        def close(self):
            self.live_sink.read()
            self.stream.disconnect()

    monkeypatch.setattr(rerun_visualizer_module, "NewtonViewerRerun", _RecordingViewer)
    monkeypatch.setattr(
        rerun_visualizer_module,
        "_ensure_rerun_server",
        lambda **_kwargs: ("rerun+http://127.0.0.1:9876/proxy", False),
    )
    provider = MagicMock(num_envs=1)
    provider.create_mapping.return_value = ()
    provider.get_transforms.return_value = False
    backend = SimpleNamespace(
        model=SimpleNamespace(num_envs=1, body_count=0, body_label=()),
        state_0=SimpleNamespace(body_q=None, particle_q=None),
        geometry_offsets=(),
    )
    sim = SimpleNamespace(
        cfg=SimpleNamespace(physics=object()),
        device="cpu",
        get_scene_data_provider=lambda: provider,
        get_or_create_backend=lambda _cfg: backend,
        vis_marker_registry=SimpleNamespace(get_groups=lambda: {}),
    )
    visualizer = RerunVisualizer(
        RerunVisualizerCfg(max_fps=60.0, record_to_rrd=str(recording_path), enable_markers=False)
    )
    publication_due = iter((False, False, True))
    monkeypatch.setattr(visualizer, "_is_render_due", lambda: next(publication_due))

    try:
        visualizer.initialize(sim, cameras=[])
        for _ in range(3):
            visualizer.step(0.01)
    finally:
        visualizer.close()

    recording = rerun_bindings.load_recording(recording_path)
    recorded_frame_rows = sum(chunk.num_rows for chunk in recording.chunks() if chunk.entity_path == "/test/frame")
    assert recorded_frame_rows == 3
    assert len(viewers) == 2
    assert viewers[0].begin_frame_calls == 1
    assert viewers[1].begin_frame_calls == 3


def test_viser_visualizer_set_camera_view(monkeypatch):
    visualizer = ViserVisualizer(ViserVisualizerCfg())
    visualizer._viewer = SimpleNamespace()

    # Client ready: pose applies immediately, nothing left pending.
    monkeypatch.setattr(visualizer, "_try_apply_viser_camera_view", lambda pose: True)
    visualizer.set_camera_view([1, 2, 3], [0, 0, 0])
    assert visualizer._last_camera_pose == ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    assert visualizer._pending_camera_pose is None

    # Client not ready: pose is deferred instead of applied.
    monkeypatch.setattr(visualizer, "_try_apply_viser_camera_view", lambda pose: False)
    visualizer.set_camera_view([4, 5, 6], [0, 0, 0])
    assert visualizer._pending_camera_pose == ((4.0, 5.0, 6.0), (0.0, 0.0, 0.0))


def _initialized_rerun_visualizer(cfg: RerunVisualizerCfg) -> RerunVisualizer:
    visualizer = RerunVisualizer(cfg)
    visualizer._is_initialized = True
    visualizer._viewer = MagicMock()
    visualizer._viewer.is_paused.return_value = False
    visualizer.backend = SimpleNamespace(
        model=SimpleNamespace(num_envs=1, body_count=0),
        state_0=SimpleNamespace(body_q=None, particle_q=None),
        geometry_offsets=(),
    )
    provider = MagicMock()
    provider.get_transforms.return_value = False
    visualizer._sim = SimpleNamespace(get_scene_data_provider=lambda: provider)
    visualizer._transform_mapping = ()
    visualizer.render_tiled_rgb_array = MagicMock(return_value=None)
    return visualizer
