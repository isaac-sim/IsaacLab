# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recording section of the Newton GL HUD: video, screenshots, and policy checkpoints."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from isaaclab.paths import ISAACLAB_ROOT

if TYPE_CHECKING:
    from .newton_visualizer import NewtonGLVisualizer

logger = logging.getLogger(__name__)


def resolve_output_dir(path: str) -> Path:
    """Return the absolute recording folder for ``path``.

    A relative path lands in the repository when running from a source checkout, wherever the run was started, and
    in the working directory otherwise, such as with an installed wheel.
    """
    folder = Path(path).expanduser()
    if folder.is_absolute():
        return folder
    return (ISAACLAB_ROOT if (ISAACLAB_ROOT / "source").is_dir() else Path.cwd()) / folder


def hover_tooltip(imgui: Any, text: str) -> None:
    """Show ``text`` while the last item is hovered, including when it is disabled."""
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
        imgui.set_tooltip(text)


_RED = (1.0, 0.35, 0.35, 1.0)
_YELLOW = (1.0, 0.8, 0.2, 1.0)
_GREEN = (0.45, 0.9, 0.5, 1.0)


def _ffmpeg_executable() -> str | None:
    """Return an ffmpeg binary, preferring the one bundled with ``imageio-ffmpeg`` (the ``video`` extra)."""
    try:
        import imageio_ffmpeg  # noqa: PLC0415

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


class NewtonRecordingPanel:
    """Records the Newton GL window while the HUD asks for it.

    * **Video** streams the displayed frames (OpenGL or RTX) to an ``mp4`` through ffmpeg, like ``--video``.
    * **Checkpoint** saves the policy being played or trained to a ``.pt`` file with one button press.
    """

    def __init__(self, owner: NewtonGLVisualizer):
        """Create the panel.

        Args:
            owner: Visualizer whose window and simulation state are recorded.
        """
        self._owner = owner
        cfg = owner.cfg.backend.recording
        self.output_dir = cfg.output_dir
        self.filename_prefix = cfg.filename_prefix
        self.auto_increment = cfg.auto_increment
        self.timestamp = cfg.timestamp
        self.video_fps = cfg.video_fps or 0
        """Encoded frame rate [Hz]; ``0`` follows the window update rate."""
        self.video_length = cfg.video_length or 0.0
        """Video duration [s] after which recording stops by itself; ``0`` records until stopped."""
        self.video_path = self.checkpoint_path = ""
        """Files written by the latest recording of each kind."""
        self._video_rate = 0
        self._video_limit: int | None = None
        self._checkpoint_saver: Callable[[str], None] | None = None
        self._video_temp: Path | None = None
        self._video_unsaved: Path | None = None
        """Finished clip that stays in a temporary file until :meth:`save_video` keeps it."""
        self._video: subprocess.Popen | bool | None = None
        """The encoder, ``True`` while recording has started but the first frame has not arrived, or None."""
        self._video_started: datetime | None = None
        self._video_frames = 0
        self._video_shape: tuple[int, int] | None = None
        self.screenshot_path = ""
        """File written by the latest screenshot."""
        self.video_include_hud = False
        """Whether recorded video frames include the HUD panels."""
        self.screenshot_include_hud = False
        """Whether screenshots include the HUD panels."""
        self._screenshot_pending = False
        self._message = ""

    @property
    def recording_video(self) -> bool:
        """Whether video frames are being written."""
        return self._video is not None

    @property
    def unsaved_video(self) -> bool:
        """Whether a stopped video is waiting for :meth:`save_video`."""
        return self._video_unsaved is not None

    def draw(self, imgui: Any) -> None:
        """Draw the Recording section."""
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        active = self.recording_video
        if active:  # Keep the state visible when the section is collapsed.
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*_RED))
        expanded = imgui.collapsing_header("Recording  (REC)###recording" if active else "Recording###recording")
        if active:
            imgui.pop_style_color()
        hover_tooltip(
            imgui,
            "Record a video, take a screenshot, or save a policy checkpoint",
        )
        if not expanded:
            return
        imgui.separator()
        self._draw_video(imgui)
        self._draw_screenshot(imgui)
        self._draw_checkpoint(imgui)
        self._draw_output(imgui)
        self._draw_message(imgui)

    def _draw_video(self, imgui: Any) -> None:
        """Draw the video group: Start, Stop, Save, and status."""
        imgui.text("Video (.mp4)")
        if self.recording_video:
            if imgui.button("Stop##video"):
                self.stop_video()
            hover_tooltip(imgui, "Stop recording. Press Save to keep the video")
        else:
            if imgui.button("Start##video"):
                self.start_video()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Record the window like --video; needs ffmpeg. Starting replaces an unsaved video.")
            imgui.same_line()
            saveable = self.unsaved_video  # Saving clears it mid-frame, so the disabled scope must not re-read it.
            if not saveable:
                imgui.begin_disabled()
            if imgui.button("Save##video"):
                self.save_video()
            if not saveable:
                imgui.end_disabled()
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip("Keep the stopped video in the output folder")
        _c, self.video_include_hud = imgui.checkbox("Include HUD##video", self.video_include_hud)
        hover_tooltip(imgui, "Draw the HUD panels into the recorded frames")
        elapsed = f"{self._video_frames / max(self._video_rate, 1):.2f} s ({self._video_frames} frames)"
        if self.recording_video:
            imgui.text_colored(imgui.ImVec4(*_RED), f"REC  {elapsed}")
        elif self.unsaved_video:
            imgui.text_colored(imgui.ImVec4(*_YELLOW), f"Unsaved  {elapsed}")
        imgui.separator()

    def _draw_screenshot(self, imgui: Any) -> None:
        """Draw the screenshot group."""
        imgui.text("Screenshot (.png)")
        if imgui.button("Capture##screenshot"):
            self.request_screenshot()
        hover_tooltip(imgui, "Save the window image to the output folder")
        imgui.same_line()
        _c, self.screenshot_include_hud = imgui.checkbox("Include HUD##screenshot", self.screenshot_include_hud)
        hover_tooltip(imgui, "Draw the HUD panels into the screenshot")
        imgui.separator()

    def _draw_checkpoint(self, imgui: Any) -> None:
        """Draw the policy checkpoint group."""
        imgui.text("Checkpoint (.pt)")
        available = self._checkpoint_saver is not None
        if not available:
            imgui.begin_disabled()
        if imgui.button("Save##checkpoint"):
            self.save_checkpoint()
        if not available:
            imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "Save the policy being played or trained to the output folder"
                if available
                else "Not available in this run: it has no policy to save"
            )
        imgui.separator()

    def _draw_output(self, imgui: Any) -> None:
        """Draw the output folder, file prefix, and numbering options."""
        imgui.text("Output")
        _c, self.output_dir = imgui.input_text("Folder##recording", self.output_dir)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                f"Root folder, currently {resolve_output_dir(self.output_dir)}."
                " Files go to <folder>/video, <folder>/screenshot, and <folder>/checkpoint"
            )
        _c, self.filename_prefix = imgui.input_text("Prefix##recording", self.filename_prefix)
        _c, self.auto_increment = imgui.checkbox("Auto-number", self.auto_increment)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Add _0001, _0002, ... so a new recording never overwrites an earlier one")
        imgui.same_line()
        _c, self.timestamp = imgui.checkbox("Timestamp", self.timestamp)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Add the start date and time to the file name")
        imgui.separator()

    def _draw_message(self, imgui: Any) -> None:
        """Draw the latest result message, green when saved and red when something failed."""
        if self._message:
            imgui.separator()
            failed = any(word in self._message for word in ("failed", "Could not", "needs ffmpeg", "no policy"))
            color = _GREEN if self._message.startswith("Saved") else _RED if failed else None
            if color is None:
                imgui.text_wrapped(self._message)
            else:
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*color))
                imgui.text_wrapped(self._message)
                imgui.pop_style_color()

    def _output_path(self, kind: str, suffix: str, when: datetime | None = None) -> Path:
        """Return the next file of ``kind`` from the output settings, creating its folder."""
        folder = resolve_output_dir(self.output_dir) / kind
        folder.mkdir(parents=True, exist_ok=True)
        name = Path(self.filename_prefix).name or "newton"  # The prefix names a file, never a folder.
        if self.timestamp:
            name = f"{name}_{(when or datetime.now()):%Y%m%d-%H%M%S}"
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
        self._discard_unsaved_video()
        folder = resolve_output_dir(self.output_dir) / "video"
        folder.mkdir(parents=True, exist_ok=True)
        self._video_temp = folder / f".unsaved_video_{os.getpid()}_{id(self):x}.mp4"  # One file per panel.
        self._video_started = datetime.now()
        self._video = True  # Opened on the first frame, once the framebuffer size is known.
        self._message = "Recording video. Stop, then Save to keep it."

    def _write_video_frame(self, frame: np.ndarray) -> None:
        if self._video is True:
            height, width = frame.shape[:2]
            path = self._video_temp
            fps = self._video_rate
            self._video_shape = (height, width)
            try:
                self._video = subprocess.Popen(  # noqa: S603
                    [
                        _ffmpeg_executable(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
                    ],
                    stdin=subprocess.PIPE,
                )  # fmt: skip
            except OSError as exc:
                self._video = None
                self._message = f"Could not start ffmpeg: {exc}"
                return
        if frame.shape[:2] != self._video_shape:
            return  # The window was resized mid-recording; skip frames the encoder cannot accept.
        video = self._video
        try:
            video.stdin.write(np.ascontiguousarray(frame).tobytes())
        except OSError:  # Includes a broken pipe, which means ffmpeg has exited.
            self._video = None
            self._finish_encoder(video)
            self._message = "ffmpeg stopped unexpectedly; video recording ended."
            return
        self._video_frames += 1
        if self._video_limit is not None and self._video_frames >= self._video_limit:
            self.stop_video()
            self.save_video()  # A fixed length is meant to need no further clicks.

    def _finish_encoder(self, video: subprocess.Popen) -> bool:
        """Close the encoder, wait for it, and keep its clip only when it exited cleanly; returns whether it did."""
        with contextlib.suppress(OSError):  # A pipe the encoder already closed.
            video.stdin.close()
        succeeded = video.wait() == 0
        self._video_unsaved = self._video_temp if succeeded and self._video_temp.exists() else None
        if not succeeded:
            self._video_temp.unlink(missing_ok=True)
        return succeeded

    def stop_video(self) -> None:
        """End the capture. The clip stays in a temporary file until :meth:`save_video` keeps it."""
        video, self._video = self._video, None
        if video is None:
            return
        if video is True:
            self._message = "No frame was recorded."
            return
        if not self._finish_encoder(video):
            self._message = "ffmpeg failed; the video was discarded."
            return
        self._message = f"Stopped after {self._video_frames} frames. Press Save to keep the video."

    def save_video(self) -> None:
        """Move the stopped video to its numbered file in the output folder."""
        temp = self._video_unsaved
        if temp is None:
            return
        path = self._output_path("video", ".mp4", self._video_started)
        try:
            os.replace(temp, path)
        except OSError as exc:  # The clip stays in its temporary file, so Save can be pressed again.
            self._message = f"Could not save the video: {exc}"
            return
        self._video_unsaved = None
        self.video_path = str(path)
        self._message = f"Saved {self._video_frames} frames to {self.video_path}"
        logger.info("[NewtonRecording] Wrote %d video frames to %s", self._video_frames, self.video_path)

    def _discard_unsaved_video(self) -> None:
        temp, self._video_unsaved = self._video_unsaved, None
        if temp is not None:
            temp.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def set_checkpoint_saver(self, save: Callable[[str], None] | None) -> None:
        """Offer the policy of this run to the Checkpoint button.

        Args:
            save: Callable that writes the policy checkpoint to the given path, or None when the run has no policy.
        """
        self._checkpoint_saver = save

    def save_checkpoint(self) -> None:
        """Write the run's policy checkpoint to a numbered file in the output folder."""
        if self._checkpoint_saver is None:
            self._message = "This run has no policy to save."
            return
        path = self._output_path("checkpoint", ".pt")
        try:
            self._checkpoint_saver(str(path))
        except Exception as exc:
            logger.exception("[NewtonRecording] Could not save the policy checkpoint.")
            self._message = f"Checkpoint failed: {exc}"
            return
        self.checkpoint_path = str(path)
        self._message = f"Saved policy checkpoint to {self.checkpoint_path}"

    # ------------------------------------------------------------------
    # Per-frame capture
    # ------------------------------------------------------------------

    def capture(self, paused: bool = False) -> None:
        """Record the frame that was just rendered; call once per window update, outside the frame.

        Args:
            paused: Whether rendering is paused, when a video gets no new frames but a screenshot is still taken.
        """
        owner = self._owner
        if self._video is not None and not paused:
            self._write_video_frame(owner._viewer.get_frame(render_ui=self.video_include_hud).numpy())
        if self._screenshot_pending:
            self._screenshot_pending = False
            self._write_screenshot(owner._viewer.get_frame(render_ui=self.screenshot_include_hud).numpy())

    def request_screenshot(self) -> None:
        """Save the next rendered window frame, as a numbered PNG in the output folder."""
        self._screenshot_pending = True

    def _write_screenshot(self, frame: np.ndarray) -> None:
        """Write one frame to the next screenshot file."""
        from PIL import Image  # noqa: PLC0415

        path = self._output_path("screenshot", ".png")
        try:
            Image.fromarray(np.ascontiguousarray(frame[..., :3], dtype=np.uint8)).save(path)
        except Exception as exc:
            logger.exception("[NewtonRecording] Could not save the screenshot.")
            self._message = f"Screenshot failed: {exc}"
            return
        self.screenshot_path = str(path)
        self._message = f"Saved screenshot to {self.screenshot_path}"

    def close(self) -> None:
        """Finish an open video recording, keeping a clip that was running or stopped but not yet saved."""
        self.stop_video()
        self.save_video()
