# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Step-driven internal video recorder.

Recording is triggered by env.step() calls, not by the Gym render loop.
Frames are sourced from the configured visualizer or scene sensor and written
to mp4 files via moviepy.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ...visualizers.visualizer_cfg import ImageViewCfg
from .video_recorder_cfg import CAPTURE_VISUALIZER_TYPES, parse_video_source

if TYPE_CHECKING:
    from .video_recorder_cfg import VideoRecorderCfg

logger = logging.getLogger(__name__)


class VideoRecorder:
    """Records one video stream per :class:`VideoRecorderCfg` entry.

    Instantiated by the env base class; ``step()`` is called once per env step
    after physics and rendering have completed.

    Raises:
        ImportError: If ``moviepy`` is not installed.
        ValueError: If :attr:`~VideoRecorderCfg.source` does not follow the source grammar.
    """

    def __init__(self, cfg: VideoRecorderCfg, env: object):
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        self._clip_class = ImageSequenceClip
        self._source = parse_video_source(cfg.source) if cfg.view is None else None
        self._capture: Callable[[], np.ndarray | None] | None = None
        view_cfg = cfg.view
        if self._source is not None and self._source[0] == "sensor":
            _, name, channel = self._source
            view_cfg = ImageViewCfg(
                source=name, channels=(channel or "rgb",),
                depth_range=(cfg.depth_colormap_min, cfg.depth_colormap_max),
            )  # fmt: skip
        if view_cfg is not None:
            camera = env.scene.sensors.get(view_cfg.source) if isinstance(view_cfg.source, str) else None
            view = env.sim.get_or_create_backend(view_cfg, camera=camera)
            self._capture = lambda: view.read_rgb(env.sim.get_physics_step_count())

        self.cfg = cfg
        self._env = env
        self._frames: list[np.ndarray] = []
        self._step_count = 0
        self._clip_index = max(self._existing_clip_indices(), default=-1) + 1
        self._recording = False

    def step(self) -> None:
        """Advance the recorder by one env step."""
        self._step_count += 1

        # Skip steps before the configured offset.
        if self._step_count <= self.cfg.step_offset:
            return

        effective_step = self._step_count - self.cfg.step_offset
        interval = self.cfg.video_interval
        clip_step = (effective_step - 1) % interval + 1 if interval else effective_step

        if clip_step == 1:
            if self._recording:
                self._close_clip()
            self._recording = True

        if self._recording:
            if clip_step % self.cfg.frame_stride == 0:
                frame = self._get_frame()
                if frame is not None:
                    self._frames.append(frame)
            if clip_step >= self.cfg.video_length:
                self._close_clip()

    def close(self) -> None:
        """Flush any buffered frames and close the current clip."""
        self._close_clip()
        self._capture = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_frame(self) -> np.ndarray | None:
        if self._capture is not None:
            return self._capture()
        # Recorders are constructed before visualizers initialize; bind their capture on first use.
        sim = self._env.sim
        _, name, channel = self._source
        for viz in sim.visualizers:
            if viz.cfg.visualizer_type in CAPTURE_VISUALIZER_TYPES and (not name or viz.cfg.visualizer_type == name):
                break
        else:
            active = [viz.cfg.visualizer_type for viz in sim.visualizers]
            raise RuntimeError(
                f"Source {self.cfg.source!r} has no recording-capable visualizer (active: {active}). "
                "Use --viz kit, --viz newton_gl, --viz newton_rtx, or source='sensor:<name>'."
            )
        if channel == "streaming_view" and not viz.cfg.streaming_view:
            raise RuntimeError(f"Enable streaming_view on {viz.cfg.visualizer_type!r} to record its sensor view.")
        if viz.cfg.visualizer_type == "kit" and sim.physics_manager.video_capture_backend() == "newton_gl":
            logger.warning(
                "[VideoRecorder] Kit with Newton physics requires cubric to propagate transforms to RTX. "
                "Use source='viz:newton_gl' if cubric is unavailable."
            )
        # Legacy visualizer recording follows camera selection in the window.
        self._capture = viz.render_tiled_rgb_array if channel == "streaming_view" else viz.render_rgb_array
        return self._capture()

    def _clip_path(self, index: int) -> str:
        return os.path.join(self.cfg.output_dir or "videos", f"{self.cfg.output_filename_prefix}_{index:04d}.mp4")

    def _existing_clip_indices(self) -> list[int]:
        output_dir = self.cfg.output_dir or "videos"
        if not os.path.isdir(output_dir):
            return []

        pattern = re.compile(rf"^{re.escape(self.cfg.output_filename_prefix)}_(?P<index>\d+)\.mp4$")
        return [
            int(match.group("index"))
            for filename in os.listdir(output_dir)
            if (match := pattern.match(filename)) is not None
        ]

    def _close_clip(self) -> None:
        if not self._frames:
            self._recording = False
            return
        try:
            os.makedirs(self.cfg.output_dir or "videos", exist_ok=True)
            path = self._clip_path(self._clip_index)
            fps = self.cfg.fps
            if fps is None:
                base_fps = self._env.metadata.get("render_fps")
                if base_fps is None:
                    base_fps = 1.0 / self._env.step_dt
                # frame_stride subsamples: one frame every N steps, so playback fps scales down.
                fps = max(1, round(base_fps / self.cfg.frame_stride))
            clip = self._clip_class(self._frames, fps=fps)
            clip.write_videofile(path, codec="libx264", audio=False, logger=None)
            logger.info("[VideoRecorder] Wrote %d frames to %s", len(self._frames), path)
            self._clip_index += 1
            if self.cfg.keep_last_n_clips is not None:
                cutoff = self._clip_index - self.cfg.keep_last_n_clips
                for index in self._existing_clip_indices():
                    if index >= cutoff:
                        continue
                    path = self._clip_path(index)
                    try:
                        os.remove(path)
                        logger.debug("[VideoRecorder] Deleted old clip %s", path)
                    except FileNotFoundError:
                        pass
        except Exception:
            logger.exception("[VideoRecorder] Failed to write clip.")
        finally:
            self._frames = []
            self._recording = False
