# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Newton GL HUD recording panel without a window."""

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from isaaclab_visualizers.newton import NewtonRecordingCfg, newton_recording
from isaaclab_visualizers.newton.newton_recording import NewtonRecordingPanel, _ffmpeg_executable, resolve_output_dir

pytestmark = pytest.mark.kitless


class _Array:
    def __init__(self, values):
        self._values = np.asarray(values)

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
    assert sorted(file.name for file in folder.iterdir()) == ["newton_0001.pt", "newton_0002.pt"]
    assert panel.checkpoint_path == str(folder / "newton_0002.pt")


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
    assert (tmp_path / "video" / "newton_0001.mp4").stat().st_size > 0
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
    assert [file.name for file in video_dir.iterdir()] == [panel._video_temp.name]  # Nothing numbered yet.

    panel.start_video()  # Starting again replaces the unsaved clip.
    assert not panel.unsaved_video
    panel.capture()
    panel.stop_video()
    panel.save_video()

    assert not panel.unsaved_video
    assert [file.name for file in video_dir.iterdir()] == ["newton_0001.mp4"]


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
    assert (tmp_path / "video" / "newton_0001.mp4").exists()


@pytest.mark.skipif(_ffmpeg_executable() is None, reason="ffmpeg is not installed")
def test_closing_keeps_a_running_video(tmp_path):
    panel = NewtonRecordingPanel(_owner([np.zeros((16, 24, 3), np.uint8)] * 2, output_dir=str(tmp_path)))

    panel.start_video()
    panel.capture()
    panel.close()

    assert (tmp_path / "video" / "newton_0001.mp4").exists()


def test_relative_output_folder_lands_in_the_repository_not_the_working_directory(tmp_path, monkeypatch):
    repo, elsewhere = tmp_path / "repo", tmp_path / "elsewhere"
    (repo / "source").mkdir(parents=True)
    elsewhere.mkdir()
    monkeypatch.setattr(newton_recording, "ISAACLAB_ROOT", repo)
    monkeypatch.chdir(elsewhere)

    assert resolve_output_dir("recording") == repo / "recording"
    assert resolve_output_dir(str(tmp_path / "absolute")) == tmp_path / "absolute"


def test_relative_output_folder_uses_the_working_directory_without_a_source_checkout(tmp_path, monkeypatch):
    package, cwd = tmp_path / "site-packages" / "isaaclab", tmp_path / "cwd"
    package.mkdir(parents=True)
    cwd.mkdir()
    monkeypatch.setattr(newton_recording, "ISAACLAB_ROOT", package)
    monkeypatch.chdir(cwd)

    assert resolve_output_dir("recording") == cwd / "recording"


class _StrictImgui:
    """Imgui double that clicks one button and fails like imgui does on an unbalanced disabled scope."""

    class HoveredFlags_:
        allow_when_disabled = 1

    def __init__(self, click: str):
        self.click, self.disabled_depth = click, 0

    def ImVec4(self, *values):  # noqa: N802
        return values

    def button(self, label):
        return label == self.click

    def checkbox(self, _label, value):
        return False, value

    def begin_disabled(self):
        self.disabled_depth += 1

    def end_disabled(self):
        assert self.disabled_depth > 0, "Calling EndDisabled() too many times!"
        self.disabled_depth -= 1

    def __getattr__(self, _name):
        return lambda *args, **kwargs: False


def test_saving_a_video_keeps_the_disabled_scopes_balanced(tmp_path):
    frames = [np.zeros((16, 24, 3), np.uint8)] * 3
    panel = NewtonRecordingPanel(_owner(frames, output_dir=str(tmp_path)))
    panel.start_video()
    panel.capture()
    panel.stop_video()
    imgui = _StrictImgui(click="Save##video")

    panel._draw_video(imgui)  # Saving clears the unsaved video partway through the frame.

    assert not panel.unsaved_video and imgui.disabled_depth == 0


def test_screenshot_writes_the_next_frame_as_a_numbered_png(tmp_path):
    frame = np.zeros((4, 6, 3), np.uint8)
    frame[..., 0] = 200
    panel = NewtonRecordingPanel(_owner(frames=[frame, frame], output_dir=str(tmp_path)))

    panel.request_screenshot()
    panel.capture()
    panel.capture()  # Nothing is pending, so no second file.
    panel.request_screenshot()
    panel.capture()

    from PIL import Image

    folder = tmp_path / "screenshot"
    assert sorted(file.name for file in folder.iterdir()) == ["newton_0001.png", "newton_0002.png"]
    assert np.array(Image.open(folder / "newton_0001.png"))[0, 0].tolist() == [200, 0, 0]
    assert panel.screenshot_path == str(folder / "newton_0002.png")


class _Encoder:
    """Stand-in for the ffmpeg process whose input pipe can break."""

    def __init__(self, broken=False, returncode=0):
        self.broken, self.returncode, self.waited, self.written = broken, returncode, False, 0
        self.stdin = SimpleNamespace(write=self._write, close=self._close)

    def _write(self, data):
        if self.broken:
            raise BrokenPipeError
        self.written += 1

    def _close(self):
        if self.broken:
            raise BrokenPipeError

    def wait(self):
        self.waited = True
        return self.returncode


def _recording_to(tmp_path, encoder, **recording):
    panel = NewtonRecordingPanel(_owner(output_dir=str(tmp_path), **recording))
    panel._video, panel._video_shape, panel._video_limit = encoder, (4, 4), None
    panel._video_temp = tmp_path / ".unsaved_video.mp4"
    panel._video_started = None
    panel._video_temp.write_bytes(b"partial")
    return panel


def test_a_broken_encoder_pipe_ends_the_recording_and_reaps_the_process(tmp_path):
    encoder = _Encoder(broken=True, returncode=1)
    panel = _recording_to(tmp_path, encoder)

    panel._write_video_frame(np.zeros((4, 4, 3), np.uint8))

    assert not panel.recording_video and encoder.waited  # The dead process is reaped, not left as a zombie.
    assert not panel.unsaved_video and not panel._video_temp.exists()  # A failed encode is not offered for saving.
    assert "ffmpeg stopped unexpectedly" in panel._message


def test_an_encoder_that_exits_with_an_error_leaves_nothing_to_save(tmp_path):
    panel = _recording_to(tmp_path, _Encoder(returncode=1))

    panel.stop_video()

    assert not panel.unsaved_video and "ffmpeg failed" in panel._message


def test_each_panel_encodes_to_its_own_temporary_file_and_a_prefix_cannot_leave_the_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(newton_recording, "_ffmpeg_executable", lambda: "ffmpeg")
    first = NewtonRecordingPanel(_owner(output_dir=str(tmp_path), filename_prefix="../outside"))
    second = NewtonRecordingPanel(_owner(output_dir=str(tmp_path)))
    first.start_video()
    second.start_video()

    assert first._video_temp != second._video_temp
    assert first._output_path("video", ".mp4").parent == tmp_path / "video"
    assert first._output_path("video", ".mp4").name == "outside_0001.mp4"


def test_a_screenshot_is_taken_while_rendering_is_paused_but_a_video_gets_no_frame(tmp_path):
    panel = NewtonRecordingPanel(_owner(frames=[np.zeros((4, 6, 3), np.uint8)], output_dir=str(tmp_path)))
    encoder = _Encoder()
    panel._video, panel._video_shape = encoder, (4, 6)

    panel.request_screenshot()
    panel.capture(paused=True)

    assert encoder.written == 0 and (tmp_path / "screenshot" / "newton_0001.png").exists()


def test_a_failed_save_is_reported_as_a_save_error_and_can_be_retried(tmp_path, monkeypatch):
    panel = _recording_to(tmp_path, _Encoder(), video_length=0.0)
    panel.stop_video()
    assert panel.unsaved_video

    monkeypatch.setattr("os.replace", Mock(side_effect=OSError("disk full")))
    panel.save_video()

    assert panel.unsaved_video and "disk full" in panel._message and "ffmpeg" not in panel._message
    monkeypatch.undo()
    panel.save_video()
    assert not panel.unsaved_video and (tmp_path / "video" / "newton_0001.mp4").exists()


def test_hud_is_captured_only_when_the_toggle_asks_for_it(tmp_path):
    requested = []
    owner = _owner(frames=[np.zeros((4, 6, 3), np.uint8)] * 2, output_dir=str(tmp_path))
    owner._viewer.get_frame = lambda render_ui=False: (
        requested.append(render_ui) or _Array(np.zeros((4, 6, 3), np.uint8))
    )
    panel = NewtonRecordingPanel(owner)

    panel.request_screenshot()
    panel.capture()
    panel.screenshot_include_hud = True
    panel.request_screenshot()
    panel.capture()

    assert requested == [False, True]
