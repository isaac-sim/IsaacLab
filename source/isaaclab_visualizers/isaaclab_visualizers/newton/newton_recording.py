# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recording section of the Newton GL HUD: video, ``.viser`` scene recordings, and state checkpoints."""

from __future__ import annotations

import logging
import re
import shutil
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .newton_visualizer import NewtonGLVisualizer

logger = logging.getLogger(__name__)

_STATE_FIELDS = ("body_q", "body_qd", "joint_q", "joint_qd", "particle_q", "particle_qd")
"""Newton state arrays written by a checkpoint."""


def _ffmpeg_executable() -> str | None:
    """Return an ffmpeg binary, preferring the one bundled with ``imageio-ffmpeg`` (the ``video`` extra)."""
    try:
        import imageio_ffmpeg  # noqa: PLC0415

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class NewtonRecordingPanel:
    """Records the Newton GL window while the HUD asks for it.

    * **Video** streams the displayed frames (OpenGL or RTX) to an ``mp4`` through ffmpeg, like ``--video``.
    * **Scene** writes a ``.viser`` recording of the visible environments that the Viser player opens.
    * **Checkpoint** saves the current simulation state arrays to an ``npz`` file with one button press.
    """

    def __init__(self, owner: NewtonGLVisualizer):
        """Create the panel.

        Args:
            owner: Visualizer whose window and simulation state are recorded.
        """
        self._owner = owner
        cfg = owner.cfg.recording
        self.output_dir = cfg.output_dir
        self.filename_prefix = cfg.filename_prefix
        self.auto_increment = cfg.auto_increment
        self.timestamp = cfg.timestamp
        self.video_fps = cfg.video_fps or 0
        """Encoded frame rate [Hz]; ``0`` follows the window update rate."""
        self.video_length = cfg.video_length or 0.0
        """Video duration [s] after which recording stops by itself; ``0`` records until stopped."""
        self.video_path = self.scene_path = self.checkpoint_path = ""
        """Files written by the latest recording of each kind."""
        self._video_rate = 0
        self._video_limit: int | None = None
        self._video: subprocess.Popen | None = None
        self._video_frames = 0
        self._video_shape: tuple[int, int] | None = None
        self._scene: Any = None
        self._scene_frames = 0
        self._message = ""

    @property
    def recording_video(self) -> bool:
        """Whether video frames are being written."""
        return self._video is not None

    @property
    def recording_scene(self) -> bool:
        """Whether a ``.viser`` recording is open."""
        return self._scene is not None

    def draw(self, imgui: Any) -> None:
        """Draw the Recording section."""
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Recording"):
            return
        imgui.separator()
        imgui.text("Output")
        _c, self.output_dir = imgui.input_text("Folder##recording", self.output_dir)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Root folder; files go to <folder>/video, <folder>/scene, and <folder>/checkpoint")
        _c, self.filename_prefix = imgui.input_text("Prefix##recording", self.filename_prefix)
        _c, self.auto_increment = imgui.checkbox("Auto-number", self.auto_increment)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Add _0001, _0002, ... so a new recording never overwrites an earlier one")
        imgui.same_line()
        _c, self.timestamp = imgui.checkbox("Timestamp", self.timestamp)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Add the start date and time to the file name")
        imgui.separator()

        imgui.text("Video")
        if self.recording_video:
            imgui.begin_disabled()
        _c, fps = imgui.input_int("FPS##video", self.video_fps)
        self.video_fps = max(0, fps)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Frame rate of the video; 0 uses the window update rate")
        _c, length = imgui.input_float("Length (s)##video", self.video_length, 0.0, 0.0, "%.1f")
        self.video_length = max(0.0, length)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Stop automatically after this many seconds of video; 0 records until stopped")
        if self.recording_video:
            imgui.end_disabled()
        if self.recording_video:
            if imgui.button("Stop##video"):
                self.stop_video()
            imgui.same_line()
            rate = max(self._video_rate, 1)
            imgui.text(f"{self._video_frames / rate:.1f} s ({self._video_frames} frames)")
        elif imgui.button("Start##video"):
            self.start_video()
        if imgui.is_item_hovered() and not self.recording_video:
            imgui.set_tooltip("Record the window like --video; needs ffmpeg")
        imgui.separator()

        imgui.text("Scene")
        if self.recording_scene:
            if imgui.button("Stop##scene"):
                self.stop_scene()
            imgui.same_line()
            imgui.text(f"{self._scene_frames} frames")
        elif imgui.button("Start##scene"):
            self.start_scene()
        if imgui.is_item_hovered() and not self.recording_scene:
            imgui.set_tooltip("Record the visible environments to a .viser file for the Viser player")
        imgui.separator()

        imgui.text("Checkpoint")
        if imgui.button("Save##checkpoint"):
            self.save_checkpoint()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Save the current simulation state to an npz file")
        if self._message:
            imgui.separator()
            imgui.text_wrapped(self._message)

    def _output_path(self, kind: str, suffix: str) -> Path:
        """Return the next file of ``kind`` from the output settings, creating its folder."""
        folder = Path(self.output_dir) / kind
        folder.mkdir(parents=True, exist_ok=True)
        name = self.filename_prefix or "newton_gl"
        if self.timestamp:
            name = f"{name}_{datetime.now():%Y%m%d-%H%M%S}"
        if not self.auto_increment:
            return folder / f"{name}{suffix}"
        pattern = re.compile(rf"^{re.escape(name)}_(\d+){re.escape(suffix)}$")
        indices = [int(match.group(1)) for file in folder.iterdir() if (match := pattern.match(file.name))]
        return folder / f"{name}_{max(indices, default=0) + 1:04d}{suffix}"

    # ------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------

    def start_video(self) -> None:
        """Start writing the displayed frames to :attr:`video_path`; the size is fixed by the first frame."""
        if self._video is not None:
            return
        if _ffmpeg_executable() is None:
            self._message = "Video needs ffmpeg: install it or the 'video' extra."
            return
        self._video_frames, self._video_shape = 0, None
        self._video_rate = self.video_fps or max(1, round(self._owner.cfg.window.fps))
        self._video_limit = round(self.video_length * self._video_rate) if self.video_length > 0 else None
        self.video_path = str(self._output_path("video", ".mp4"))
        self._video = True  # Opened on the first frame, once the framebuffer size is known.
        self._message = f"Recording video to {self.video_path}"

    def _write_video_frame(self, frame: np.ndarray) -> None:
        if self._video is True:
            height, width = frame.shape[:2]
            path = Path(self.video_path)
            fps = self._video_rate
            self._video_shape = (height, width)
            self._video = subprocess.Popen(  # noqa: S603
                [
                    _ffmpeg_executable(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
                ],
                stdin=subprocess.PIPE,
            )  # fmt: skip
        if frame.shape[:2] != self._video_shape:
            return  # The window was resized mid-recording; skip frames the encoder cannot accept.
        try:
            self._video.stdin.write(np.ascontiguousarray(frame).tobytes())
            self._video_frames += 1
            if self._video_limit is not None and self._video_frames >= self._video_limit:
                self.stop_video()
        except (BrokenPipeError, OSError):
            self._message = "ffmpeg stopped unexpectedly; video recording ended."
            self._video = None

    def stop_video(self) -> None:
        """Finish the video file."""
        video, self._video = self._video, None
        if video is None:
            return
        if video is not True:
            video.stdin.close()
            video.wait()
            self._message = f"Saved {self._video_frames} frames to {self.video_path}"
            logger.info("[NewtonRecording] Wrote %d video frames to %s", self._video_frames, self.video_path)
        else:
            self._message = "No frame was recorded."

    # ------------------------------------------------------------------
    # Scene (.viser)
    # ------------------------------------------------------------------

    def start_scene(self) -> None:
        """Start a ``.viser`` recording of the visible environments to :attr:`scene_path`."""
        if self._scene is not None:
            return
        from newton.viewer import ViewerViser  # noqa: PLC0415

        owner = self._owner
        path = self._output_path("scene", ".viser")
        self.scene_path = str(path)
        try:
            viewer = ViewerViser(port=_free_port(), label="Isaac Lab", verbose=False, record_to_viser=str(path))
            viewer.set_model(owner.backend.model)
            viewer.set_visible_worlds(owner._env_ids)
            viewer.set_world_offsets((0.0, 0.0, 0.0))
        except Exception as exc:
            logger.exception("[NewtonRecording] Could not start the .viser recording.")
            self._message = f"Scene recording failed: {exc}"
            return
        self._scene, self._scene_frames = viewer, 0
        self._message = f"Recording scene to {self.scene_path}"

    def stop_scene(self) -> None:
        """Write the ``.viser`` file."""
        viewer, self._scene = self._scene, None
        if viewer is None:
            return
        try:
            viewer.close()
            self._message = f"Saved {self._scene_frames} frames to {self.scene_path}"
        except Exception as exc:
            logger.exception("[NewtonRecording] Could not finish the .viser recording.")
            self._message = f"Scene recording failed: {exc}"

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self) -> None:
        """Write the simulation state arrays to :attr:`checkpoint_path`."""
        state = self._owner._update_state()
        arrays = {name: array.numpy() for name in _STATE_FIELDS if (array := getattr(state, name, None)) is not None}
        path = self._output_path("checkpoint", ".npz")
        self.checkpoint_path = str(path)
        np.savez(path, **arrays)
        self._message = f"Saved checkpoint ({', '.join(arrays)}) to {self.checkpoint_path}"

    # ------------------------------------------------------------------
    # Per-frame capture
    # ------------------------------------------------------------------

    def capture(self) -> None:
        """Record the frame that was just rendered; call once per window update, outside the frame."""
        owner = self._owner
        if self._scene is not None:
            self._scene.begin_frame(owner._sim_time)
            try:
                self._scene.log_state(owner._update_state())
            finally:
                self._scene.end_frame()
            self._scene_frames += 1
        if self._video is not None:
            self._write_video_frame(owner._viewer.get_frame(render_ui=False).numpy())

    def close(self) -> None:
        """Finish all open recordings."""
        self.stop_video()
        self.stop_scene()
