# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recording section of the Newton GL HUD: video, ``.viser`` scene recordings, and state checkpoints."""

from __future__ import annotations

import logging
import os
import re
import shutil
import socket
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from isaaclab.paths import ISAACLAB_ROOT

from .newton_animation_panel import NewtonAnimationPanel
from .newton_replay import RecordedSlot, write_slot_sidecar
from .newton_visualization_markers import render_newton_visualization_markers

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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class NewtonRecordingPanel:
    """Records the Newton GL window while the HUD asks for it.

    * **Video** streams the displayed frames (OpenGL or RTX) to an ``mp4`` through ffmpeg, like ``--video``.
    * **Scene** writes a ``.viser`` recording of the visible environments that the Viser player opens.
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
        self.video_path = self.scene_path = self.checkpoint_path = ""
        """Files written by the latest recording of each kind."""
        self._video_rate = 0
        self._video_limit: int | None = None
        self._checkpoint_saver: Callable[[str], None] | None = None
        self._video_temp: Path | None = None
        self._video_unsaved: Path | None = None
        """Finished clip that stays in a temporary file until :meth:`save_video` keeps it."""
        self.animation = NewtonAnimationPanel(owner, self.scene_folder)
        self._video: subprocess.Popen | None = None
        self._video_frames = 0
        self._video_shape: tuple[int, int] | None = None
        self._scene: Any = None
        self._scene_frames = 0
        self._markers_failed = False
        self._message = ""

    @property
    def recording_video(self) -> bool:
        """Whether video frames are being written."""
        return self._video is not None

    @property
    def unsaved_video(self) -> bool:
        """Whether a stopped video is waiting for :meth:`save_video`."""
        return self._video_unsaved is not None

    @property
    def recording_scene(self) -> bool:
        """Whether a ``.viser`` recording is open."""
        return self._scene is not None

    def draw(self, imgui: Any) -> None:
        """Draw the Recording section."""
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        active = self.recording_video or self.recording_scene
        if active:  # Keep the state visible when the section is collapsed.
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*_RED))
        expanded = imgui.collapsing_header("Recording  (REC)###recording" if active else "Recording###recording")
        if active:
            imgui.pop_style_color()
        hover_tooltip(imgui, "Record a video, record an animation (and play it back), or save a policy checkpoint")
        self.animation.draw_window(imgui)
        if not expanded:
            return
        imgui.separator()
        self._draw_video(imgui)
        self._draw_animation(imgui)
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
        elapsed = f"{self._video_frames / max(self._video_rate, 1):.2f} s ({self._video_frames} frames)"
        if self.recording_video:
            imgui.text_colored(imgui.ImVec4(*_RED), f"REC  {elapsed}")
        elif self.unsaved_video:
            imgui.text_colored(imgui.ImVec4(*_YELLOW), f"Unsaved  {elapsed}")
        imgui.separator()

    def _draw_animation(self, imgui: Any) -> None:
        """Draw the animation group: Record and its status."""
        imgui.text("Animation (.viser)")
        if self.recording_scene:
            if imgui.button("Stop##scene"):
                self.stop_scene()
            hover_tooltip(imgui, "Stop recording and write the .viser file")
        elif imgui.button("Record##scene"):
            self.start_scene()
        if imgui.is_item_hovered() and not self.recording_scene:
            imgui.set_tooltip("Record the visible environments to a .viser file")
        imgui.same_line()
        self.animation.draw_open_button(imgui)
        if self.recording_scene:
            imgui.text_colored(imgui.ImVec4(*_RED), f"REC  {self._scene_frames} frames")
        self.animation.draw(imgui)
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
                " Files go to <folder>/video, <folder>/scene, and <folder>/checkpoint"
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

    def scene_folder(self) -> str:
        """Folder the Add Animation window lists by default: the ``scene`` subfolder of the output folder."""
        return str(resolve_output_dir(self.output_dir) / "scene")

    def _output_path(self, kind: str, suffix: str) -> Path:
        """Return the next file of ``kind`` from the output settings, creating its folder."""
        folder = resolve_output_dir(self.output_dir) / kind
        folder.mkdir(parents=True, exist_ok=True)
        name = self.filename_prefix or "newton"
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
        self._discard_unsaved_video()
        folder = resolve_output_dir(self.output_dir) / "video"
        folder.mkdir(parents=True, exist_ok=True)
        self._video_temp = folder / ".unsaved_video.mp4"
        self._video = True  # Opened on the first frame, once the framebuffer size is known.
        self._message = "Recording video. Stop, then Save to keep it."

    def _write_video_frame(self, frame: np.ndarray) -> None:
        if self._video is True:
            height, width = frame.shape[:2]
            path = self._video_temp
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
                self.save_video()  # A fixed length is meant to need no further clicks.
        except (BrokenPipeError, OSError):
            self._message = "ffmpeg stopped unexpectedly; video recording ended."
            self._video = None

    def stop_video(self) -> None:
        """End the capture. The clip stays in a temporary file until :meth:`save_video` keeps it."""
        video, self._video = self._video, None
        if video is None:
            return
        if video is True:
            self._message = "No frame was recorded."
            return
        video.stdin.close()
        video.wait()
        self._video_unsaved = self._video_temp
        self._message = f"Stopped after {self._video_frames} frames. Press Save to keep the video."

    def save_video(self) -> None:
        """Move the stopped video to its numbered file in the output folder."""
        temp, self._video_unsaved = self._video_unsaved, None
        if temp is None:
            return
        path = self._output_path("video", ".mp4")
        os.replace(temp, path)
        self.video_path = str(path)
        self._message = f"Saved {self._video_frames} frames to {self.video_path}"
        logger.info("[NewtonRecording] Wrote %d video frames to %s", self._video_frames, self.video_path)

    def _discard_unsaved_video(self) -> None:
        temp, self._video_unsaved = self._video_unsaved, None
        if temp is not None:
            temp.unlink(missing_ok=True)

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
            offsets = getattr(owner._viewer, "world_offsets", None)
            if owner._policy_ghost_panel is not None and offsets is not None:
                viewer.world_offsets = offsets  # Ghost slots sit on the reference, as in the window.
            else:
                viewer.set_world_offsets((0.0, 0.0, 0.0))
            viewer.marker_groups = owner._viewer.marker_groups  # The same markers the window draws.
        except Exception as exc:
            logger.exception("[NewtonRecording] Could not start the .viser recording.")
            self._message = f"Scene recording failed: {exc}"
            return
        self._scene, self._scene_frames, self._markers_failed = viewer, 0, False
        self._message = f"Recording scene to {self.scene_path}"

    def stop_scene(self) -> None:
        """Write the ``.viser`` file."""
        viewer, self._scene = self._scene, None
        if viewer is None:
            return
        try:
            slots = self._describe_slots(viewer)
            viewer.close()
            if slots is not None:
                write_slot_sidecar(self.scene_path, *slots)
            self._message = f"Saved {self._scene_frames} frames to {self.scene_path}"
        except Exception as exc:
            logger.exception("[NewtonRecording] Could not finish the .viser recording.")
            self._message = f"Scene recording failed: {exc}"

    def _describe_slots(self, viewer: Any):
        """Return what a replay needs to tint each Policy Ghost slot of a recorded ghost session, or None."""
        panel = self._owner._policy_ghost_panel
        if panel is None:
            return None
        layout = panel.layout
        slots = [
            RecordedSlot(slot.index, slot.label, None if slot.index == 0 else tuple(slot.color or (1.0, 1.0, 1.0)))
            for slot in layout.slots
            if slot.active
        ]
        worlds = {}
        for batch in viewer._shape_instances.values():
            ids = batch.worlds.numpy() if hasattr(batch.worlds, "numpy") else batch.worlds
            worlds[batch.name] = [int(world) for world in ids]
        env_ids = list(self._owner._env_ids) if self._owner._env_ids is not None else None
        return layout.slice_size, slots, worlds, env_ids

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

    def capture(self) -> None:
        """Record the frame that was just rendered; call once per window update, outside the frame."""
        owner = self._owner
        if self._scene is not None:
            self._scene.begin_frame(owner._sim_time)
            try:
                self._scene.log_state(owner._update_state())
                self._record_markers(owner)
            finally:
                self._scene.end_frame()
            self._scene_frames += 1
        if self._video is not None:
            self._write_video_frame(owner._viewer.get_frame(render_ui=False).numpy())

    def _record_markers(self, owner: Any) -> None:
        """Add the visualization markers, such as velocity arrows, to the animation frame being recorded."""
        if not owner.cfg.enable_markers or self._markers_failed:
            return
        try:
            render_newton_visualization_markers(self._scene, owner._env_ids, num_envs=owner.backend.model.num_envs)
        except Exception:
            logger.exception("[NewtonRecording] Markers could not be recorded; continuing without them.")
            self._markers_failed = True

    def close(self) -> None:
        """Finish all open recordings, keeping a video that was running or stopped but not yet saved."""
        self.stop_video()
        self.save_video()
        self.stop_scene()
