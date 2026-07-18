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
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .video_recorder_cfg import VideoRecorderCfg

logger = logging.getLogger(__name__)


def _parse_source(source: str) -> tuple[str, str, str]:
    """Parse a source string into (kind, type_or_name, sub).

    Returns:
        ``(kind, type_or_name, sub)`` where ``kind`` is ``"visualizer"`` or
        ``"sensor"``, ``type_or_name`` is the visualizer type or sensor name,
        and ``sub`` is the sub-channel (``"interactive"``, ``"tiled"``, data type, …).
    """
    source = source.strip()
    if ":" in source:
        kind, rest = source.split(":", 1)
    else:
        kind, rest = source, ""

    if "/" in rest:
        type_or_name, sub = rest.split("/", 1)
    else:
        type_or_name, sub = rest, ""

    return kind.lower(), type_or_name.lower(), sub.lower()


class VideoRecorder:
    """Records one video stream per :class:`VideoRecorderCfg` entry.

    Instantiated by the env base class; ``step()`` is called once per env step
    after physics and rendering have completed.
    """

    def __init__(self, cfg: VideoRecorderCfg, env: object):
        self.cfg = cfg
        self._env = env
        self._frames: list[np.ndarray] = []
        self._step_count = 0
        self._clip_index = 0
        self._recording = False

    def step(self) -> None:
        """Advance the recorder by one env step."""
        self._step_count += 1
        should_trigger = self._check_trigger()

        if should_trigger:
            if self._recording:
                self._close_clip()
            self._recording = True

        if self._recording:
            frame = self._get_frame()
            if frame is not None:
                self._frames.append(frame)
                if len(self._frames) >= self.cfg.clip_length:
                    self._close_clip()

    def close(self) -> None:
        """Flush any buffered frames and close the current clip."""
        if self._recording and self._frames:
            self._close_clip()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_trigger(self) -> bool:
        if self.cfg.clip_trigger_step <= 0:
            return self._step_count == 1
        return self._step_count % self.cfg.clip_trigger_step == 0

    def _get_frame(self) -> np.ndarray | None:
        kind, type_or_name, sub = _parse_source(self.cfg.source)
        try:
            if kind == "visualizer":
                return self._frame_from_visualizer(type_or_name, sub)
            if kind == "sensor":
                return self._frame_from_sensor(type_or_name, sub or "rgb")
        except Exception:
            logger.debug("[VideoRecorder] Frame capture failed.", exc_info=True)
        return None

    def _frame_from_visualizer(self, viz_type: str, sub: str) -> np.ndarray | None:
        sim = getattr(self._env, "sim", None)
        if sim is None:
            return None
        visualizers = getattr(sim, "visualizers", [])

        if viz_type:
            candidates = [v for v in visualizers if getattr(v.cfg, "visualizer_type", None) == viz_type]
        else:
            candidates = [v for v in visualizers if hasattr(v, "render_rgb_array")]

        if not candidates:
            logger.warning(
                "[VideoRecorder] No visualizer of type '%s' is active. "
                "Add the appropriate VisualizerCfg to sim.visualizer_cfgs.",
                viz_type or "any",
            )
            return None

        viz = candidates[0]
        if sub == "tiled":
            if hasattr(viz, "render_tiled_rgb_array"):
                return viz.render_tiled_rgb_array()
            logger.warning("[VideoRecorder] Visualizer '%s' does not support tiled capture.", viz_type)
            return None
        if not hasattr(viz, "render_rgb_array"):
            logger.warning("[VideoRecorder] Visualizer '%s' does not support render_rgb_array().", viz_type)
            return None
        return viz.render_rgb_array()

    def _frame_from_sensor(self, name: str, data_type: str) -> np.ndarray | None:
        scene = getattr(self._env, "scene", None)
        if scene is None:
            return None
        sensors = getattr(scene, "sensors", {})
        sensor = sensors.get(name)
        if sensor is None:
            logger.warning("[VideoRecorder] Sensor '%s' not found in env.scene.sensors.", name)
            return None
        output = getattr(getattr(sensor, "data", None), "output", None)
        if output is None or data_type not in output:
            return None
        data = output[data_type]
        # ProxyArray or torch.Tensor: shape (N, H, W, C)
        if hasattr(data, "torch"):
            tensor = data.torch
        else:
            tensor = data
        frame = tensor[0].cpu().numpy().astype(np.uint8)
        return frame[:, :, :3] if frame.ndim == 3 and frame.shape[2] >= 3 else frame

    def _close_clip(self) -> None:
        if not self._frames:
            self._recording = False
            return
        try:
            from moviepy.editor import ImageSequenceClip

            os.makedirs(self.cfg.output_dir, exist_ok=True)
            path = os.path.join(self.cfg.output_dir, f"clip_{self._clip_index:04d}.mp4")
            clip = ImageSequenceClip(self._frames, fps=self.cfg.fps)
            clip.write_videofile(path, codec="libx264", audio=False, logger=None)
            logger.info("[VideoRecorder] Wrote %d frames to %s", len(self._frames), path)
            self._clip_index += 1
        except Exception:
            logger.exception("[VideoRecorder] Failed to write clip.")
        finally:
            self._frames = []
            self._recording = False
