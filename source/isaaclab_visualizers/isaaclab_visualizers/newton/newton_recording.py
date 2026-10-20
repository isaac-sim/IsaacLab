# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recording section of the Newton GL HUD: video, ``.viser`` scene recordings, and state checkpoints."""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
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
        self.video_path = "videos/newton_gl.mp4"
        self.scene_path = "recordings/newton_gl.viser"
        self.checkpoint_path = "checkpoints/newton_state.npz"
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
        self._draw_row(
            imgui, "Video", "video_path", self.recording_video, self.start_video, self.stop_video,
            f"{self._video_frames} frames", "Record the window like --video; needs ffmpeg",
        )  # fmt: skip
        self._draw_row(
            imgui, "Scene", "scene_path", self.recording_scene, self.start_scene, self.stop_scene,
            f"{self._scene_frames} frames", "Record the visible environments to a .viser file for the Viser player",
        )  # fmt: skip
        self._draw_row(
            imgui, "Checkpoint", "checkpoint_path", False, self.save_checkpoint, None, "",
            "Save the current simulation state to an npz file",
        )  # fmt: skip
        if self._message:
            imgui.text_wrapped(self._message)

    def _draw_row(self, imgui, label, attribute, active, start, stop, status, tooltip) -> None:
        imgui.text(label)
        if imgui.is_item_hovered():
            imgui.set_tooltip(tooltip)
        if active:
            imgui.begin_disabled()
        _changed, value = imgui.input_text(f"##{attribute}", getattr(self, attribute))
        setattr(self, attribute, value)
        if active:
            imgui.end_disabled()
        if active and stop is not None:
            if imgui.button(f"Stop##{attribute}"):
                stop()
            imgui.same_line()
            imgui.text(status)
        elif imgui.button(f"{'Save' if stop is None else 'Start'}##{attribute}"):
            start()

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
        self._video = True  # Opened on the first frame, once the framebuffer size is known.
        self._message = f"Recording video to {self.video_path}"

    def _write_video_frame(self, frame: np.ndarray) -> None:
        if self._video is True:
            height, width = frame.shape[:2]
            path = Path(self.video_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fps = max(1, round(self._owner.cfg.window.fps))
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
        path = Path(self.scene_path)
        path.parent.mkdir(parents=True, exist_ok=True)
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
        path = Path(self.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
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
