# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Newton GL HUD recording panel without a window."""

from types import SimpleNamespace

import numpy as np
import pytest
from isaaclab_visualizers.newton.newton_recording import NewtonRecordingPanel, _ffmpeg_executable


class _Array:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def numpy(self):
        return self._values


def _owner(frames=()):
    state = SimpleNamespace(body_q=_Array([[0, 0, 1, 0, 0, 0, 1]]), joint_q=_Array([0.5, -0.5]))
    frames = iter(frames)
    viewer = SimpleNamespace(get_frame=lambda render_ui=False: _Array(next(frames)))
    return SimpleNamespace(
        _update_state=lambda: state,
        _viewer=viewer,
        _sim_time=0.0,
        cfg=SimpleNamespace(window=SimpleNamespace(fps=10.0)),
    )


def test_checkpoint_saves_only_present_state_arrays(tmp_path):
    panel = NewtonRecordingPanel(_owner())
    panel.checkpoint_path = str(tmp_path / "nested" / "state.npz")

    panel.save_checkpoint()

    with np.load(panel.checkpoint_path) as saved:
        assert sorted(saved.files) == ["body_q", "joint_q"]
        np.testing.assert_allclose(saved["joint_q"], [0.5, -0.5])


@pytest.mark.skipif(_ffmpeg_executable() is None, reason="ffmpeg is not installed")
def test_video_streams_frames_and_ignores_resized_ones(tmp_path):
    frames = [np.full((16, 24, 3), 10 * i, np.uint8) for i in range(3)] + [np.zeros((8, 8, 3), np.uint8)]
    panel = NewtonRecordingPanel(_owner(frames))
    panel.video_path = str(tmp_path / "clip.mp4")

    panel.start_video()
    for _ in frames:
        panel.capture()
    panel.stop_video()

    assert panel._video_frames == 3
    assert (tmp_path / "clip.mp4").stat().st_size > 0
    assert not panel.recording_video
