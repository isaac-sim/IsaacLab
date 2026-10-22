# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Newton GL HUD recording panel without a window."""

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from isaaclab_visualizers.newton import NewtonRecordingCfg
from isaaclab_visualizers.newton.newton_recording import NewtonRecordingPanel, _ffmpeg_executable


class _Array:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def numpy(self):
        return self._values


def _owner(frames=(), **recording):
    state = SimpleNamespace(body_q=_Array([[0, 0, 1, 0, 0, 0, 1]]), joint_q=_Array([0.5, -0.5]))
    frames = iter(frames)
    viewer = SimpleNamespace(get_frame=lambda render_ui=False: _Array(next(frames)))
    return SimpleNamespace(
        _update_state=lambda: state,
        _viewer=viewer,
        _sim_time=0.0,
        cfg=SimpleNamespace(
            window=SimpleNamespace(fps=10.0), backend=SimpleNamespace(recording=NewtonRecordingCfg(**recording))
        ),
    )


def test_checkpoint_writes_through_the_runs_saver_into_numbered_files(tmp_path):
    panel = NewtonRecordingPanel(_owner(output_dir=str(tmp_path)))
    panel.save_checkpoint()
    assert "no policy" in panel._message  # Nothing is written until the run offers a saver.

    panel.set_checkpoint_saver(lambda path: Path(path).write_text("policy"))
    panel.save_checkpoint()
    panel.save_checkpoint()

    folder = tmp_path / "checkpoint"
    assert sorted(file.name for file in folder.iterdir()) == ["newton_gl_0001.pt", "newton_gl_0002.pt"]
    assert panel.checkpoint_path == str(folder / "newton_gl_0002.pt")


def test_checkpoint_failure_is_reported_without_raising(tmp_path):
    def broken(path):
        raise OSError("disk full")

    panel = NewtonRecordingPanel(_owner(output_dir=str(tmp_path)))
    panel.set_checkpoint_saver(broken)

    panel.save_checkpoint()

    assert "disk full" in panel._message and not panel.checkpoint_path


def test_output_paths_number_stamp_or_overwrite(tmp_path):
    panel = NewtonRecordingPanel(_owner(output_dir=str(tmp_path), filename_prefix="run"))

    first = panel._output_path("video", ".mp4")
    first.touch()
    assert [first.name, panel._output_path("video", ".mp4").name] == ["run_0001.mp4", "run_0002.mp4"]

    panel.auto_increment = False
    assert panel._output_path("video", ".mp4").name == "run.mp4"

    panel.timestamp = True
    assert re.fullmatch(r"run_\d{8}-\d{6}\.mp4", panel._output_path("scene", ".mp4").name)
    assert (tmp_path / "scene").is_dir()


@pytest.mark.skipif(_ffmpeg_executable() is None, reason="ffmpeg is not installed")
def test_video_streams_frames_and_ignores_resized_ones(tmp_path):
    frames = [np.full((16, 24, 3), 10 * i, np.uint8) for i in range(3)] + [np.zeros((8, 8, 3), np.uint8)]
    panel = NewtonRecordingPanel(_owner(frames, output_dir=str(tmp_path)))

    panel.start_video()
    for _ in frames:
        panel.capture()
    panel.stop_video()
    panel.save_video()

    assert panel._video_frames == 3
    assert (tmp_path / "video" / "newton_gl_0001.mp4").stat().st_size > 0
    assert not panel.recording_video


@pytest.mark.skipif(_ffmpeg_executable() is None, reason="ffmpeg is not installed")
def test_stopped_video_is_kept_only_when_saved(tmp_path):
    frames = [np.zeros((16, 24, 3), np.uint8)] * 6
    panel = NewtonRecordingPanel(_owner(frames, output_dir=str(tmp_path)))
    video_dir = tmp_path / "video"

    panel.start_video()
    panel.capture()
    panel.capture()
    panel.stop_video()

    assert panel.unsaved_video and not panel.recording_video
    assert [file.name for file in video_dir.iterdir()] == [".unsaved_video.mp4"]  # Nothing numbered yet.

    panel.start_video()  # Starting again replaces the unsaved clip.
    assert not panel.unsaved_video
    panel.capture()
    panel.stop_video()
    panel.save_video()

    assert not panel.unsaved_video
    assert [file.name for file in video_dir.iterdir()] == ["newton_gl_0001.mp4"]


@pytest.mark.skipif(_ffmpeg_executable() is None, reason="ffmpeg is not installed")
def test_video_with_a_length_saves_itself(tmp_path):
    frames = [np.zeros((16, 24, 3), np.uint8)] * 10
    panel = NewtonRecordingPanel(_owner(frames, output_dir=str(tmp_path), video_fps=4, video_length=1.0))

    panel.start_video()
    for _ in frames:
        panel.capture()

    assert not panel.recording_video and not panel.unsaved_video
    assert panel._video_frames == 4  # 1 s at 4 Hz
    assert "Saved 4 frames" in panel._message
    assert (tmp_path / "video" / "newton_gl_0001.mp4").exists()


@pytest.mark.skipif(_ffmpeg_executable() is None, reason="ffmpeg is not installed")
def test_closing_keeps_a_running_video(tmp_path):
    panel = NewtonRecordingPanel(_owner([np.zeros((16, 24, 3), np.uint8)] * 2, output_dir=str(tmp_path)))

    panel.start_video()
    panel.capture()
    panel.close()

    assert (tmp_path / "video" / "newton_gl_0001.mp4").exists()
