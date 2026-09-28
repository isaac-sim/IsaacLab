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
from typing import TYPE_CHECKING

import numpy as np

try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
except ImportError:
    ImageSequenceClip = None  # type: ignore[assignment,misc]

from .video_recorder_cfg import parse_video_source

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
        RuntimeError: On the first recording step if the requested visualizer or
            sensor cannot be found or does not support frame capture.
    """

    def __init__(self, cfg: VideoRecorderCfg, env: object):
        self._source = parse_video_source(cfg.source)

        if ImageSequenceClip is None:
            raise ImportError("moviepy is required for video recording. Install it with: pip install 'moviepy<2'")

        self.cfg = cfg
        self._env = env
        self._frames: list[np.ndarray] = []
        self._step_count = 0
        self._frames_step_count = 0
        self._clip_index = self._next_clip_index()
        self._recording = False
        # Set to True after the first unrecoverable frame-capture error so that
        # subsequent steps do not propagate the exception or repeat the log message.
        self._frame_error_logged: bool = False
        # Set to True after the Kit/Newton cubric warning is emitted. The condition it
        # reports is fixed configuration state, so warning once per recorder is enough;
        # without this the message repeats on every captured frame.
        self._cubric_warning_logged: bool = False

    def step(self) -> None:
        """Advance the recorder by one env step."""
        self._step_count += 1

        # Skip steps before the configured offset.
        if self._step_count <= self.cfg.step_offset:
            return

        effective_step = self._step_count - self.cfg.step_offset
        should_trigger = self._check_trigger(effective_step)

        if should_trigger:
            if self._recording:
                self._close_clip()
            self._recording = True
            self._frames_step_count = 0

        if self._recording:
            self._frames_step_count += 1
            if self._frames_step_count % self.cfg.frame_stride == 0:
                frame = self._get_frame()
                if frame is not None:
                    self._frames.append(frame)
            if self._frames_step_count >= self.cfg.video_length:
                self._close_clip()

    def close(self) -> None:
        """Flush any buffered frames and close the current clip."""
        if self._recording and self._frames:
            self._close_clip()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_trigger(self, effective_step: int) -> bool:
        if self.cfg.video_interval <= 0:
            return effective_step == 1
        return (effective_step - 1) % self.cfg.video_interval == 0

    def _get_frame(self) -> np.ndarray | None:
        if self._frame_error_logged:
            return None
        try:
            kind, type_or_name, sub = self._source
            if kind == "viz":
                return self._frame_from_visualizer(type_or_name, sub)
            return self._frame_from_sensor(type_or_name, gt_type=sub)
        except RuntimeError as exc:
            logger.error(
                "[VideoRecorder] Frame capture failed for source=%r: %s  "
                "Further capture attempts for this stream will be suppressed.",
                self.cfg.source,
                exc,
            )
            self._frame_error_logged = True
            return None

    def _frame_from_visualizer(self, viz_type: str, sub: str) -> np.ndarray | None:
        sim = getattr(self._env, "sim", None)
        if sim is None:
            raise RuntimeError(
                "[VideoRecorder] env.sim is not available; cannot capture frames. "
                "Ensure the environment is fully initialized before recording starts."
            )
        visualizers = getattr(sim, "visualizers", [])

        if viz_type:
            candidates = [v for v in visualizers if getattr(v.cfg, "visualizer_type", None) == viz_type]
            if not candidates:
                active = [getattr(v.cfg, "visualizer_type", "unknown") for v in visualizers]
                raise RuntimeError(
                    f"[VideoRecorder] source='viz:{viz_type}' requested but no '{viz_type}' "
                    f"visualizer is active (active: {active or ['none']}). "
                    "Launch the simulation with isaaclab.app.launch_simulation, which adds the visualizer a "
                    "recorder needs."
                )
        else:
            # Auto: pick the first active visualizer that supports frame capture.
            candidates = [v for v in visualizers if hasattr(v, "render_rgb_array")]
            if not candidates:
                active = [getattr(v.cfg, "visualizer_type", "unknown") for v in visualizers]
                raise RuntimeError(
                    "[VideoRecorder] source='viz' found no recording-capable visualizer "
                    f"(active: {active or ['none']}). "
                    "Pass --viz kit, --viz newton_gl, or --viz newton_rtx, or use "
                    "source='sensor:<name>' to record from a scene sensor."
                )

        # Kit Replicator requires cubric to propagate Newton Fabric transforms to RTX's
        # scene delegate. Without cubric, frames will be black. Log a warning but allow
        # the capture to proceed — users with cubric available will get correct frames.
        if viz_type == "kit" and not self._cubric_warning_logged:
            physics_backend = getattr(getattr(sim, "physics_manager", None), "video_capture_backend", lambda: None)()
            if physics_backend == "newton_gl":
                self._cubric_warning_logged = True
                logger.warning(
                    "[VideoRecorder] source='viz:kit' with Newton physics requires cubric "
                    "to propagate Fabric transforms to RTX. Frames may be black if cubric is "
                    "unavailable. Use source='viz:newton_gl' for guaranteed capture."
                )

        viz = candidates[0]
        if not sim.is_rendering:
            sim.forward()
        if sub == "streaming_view":
            if not hasattr(viz, "render_tiled_rgb_array"):
                raise RuntimeError(
                    f"[VideoRecorder] source='viz:{viz_type}:streaming_view' requested but the "
                    f"'{viz_type}' visualizer does not support streaming view capture."
                )
            if not getattr(getattr(viz, "cfg", None), "streaming_view", False):
                cfg_name = {"kit": "KitVisualizerCfg", "newton_gl": "NewtonGLVisualizerCfg"}.get(
                    viz_type, "VisualizerCfg"
                )
                raise RuntimeError(
                    f"[VideoRecorder] source='viz:{viz_type}:streaming_view' requested but "
                    f"streaming_view is not enabled on the '{viz_type}' visualizer. "
                    f"Enable it by setting streaming_view=True on the visualizer config:\n\n"
                    f"    {cfg_name}(streaming_view=True, ...)\n\n"
                    "Declare a CameraCfg in the scene and select it with streaming_sensor_prim_path."
                )
            return viz.render_tiled_rgb_array()

        if not hasattr(viz, "render_rgb_array"):
            raise RuntimeError(
                f"[VideoRecorder] source='viz:{viz_type or '<auto>'}' does not support frame "
                "capture: the visualizer has no render_rgb_array() implementation."
            )

        frame = viz.render_rgb_array()
        if frame is None:
            viz_type_name = getattr(getattr(viz, "cfg", None), "visualizer_type", "unknown")
            raise RuntimeError(
                f"[VideoRecorder] render_rgb_array() returned None for '{viz_type_name}' visualizer. "
                "Use a capture-capable visualizer or source='sensor:<name>' instead."
            )
        return frame

    def _frame_from_sensor(self, name: str, gt_type: str = "rgb") -> np.ndarray | None:
        from .camera_colorizer import (
            SUPPORTED_GT_TYPES,
            CameraFrameColorizer,
            sensor_key_for_gt_type,
        )

        gt_type = gt_type or "rgb"
        if gt_type not in SUPPORTED_GT_TYPES:
            raise RuntimeError(
                f"[VideoRecorder] Unsupported GT type '{gt_type}' in sensor source. "
                f"Valid types: {sorted(SUPPORTED_GT_TYPES)}. "
                f"Use source='sensor:{name}:<type>' where <type> is one of the valid types."
            )

        scene = getattr(self._env, "scene", None)
        if scene is None:
            raise RuntimeError(
                "[VideoRecorder] env.scene is not available; cannot capture sensor frames. "
                "Ensure the environment is fully initialized before recording starts."
            )
        sensors = getattr(scene, "sensors", {})
        sensor = sensors.get(name)
        if sensor is None:
            available = sorted(sensors.keys())
            truncated = available[:8]
            suffix = f" … and {len(available) - 8} more" if len(available) > 8 else ""
            hint = (
                " Add a CameraCfg to your scene (e.g. InteractiveSceneCfg.tiled_camera) "
                "to enable sensor-based recording."
                if not available
                else ""
            )
            raise RuntimeError(
                f"[VideoRecorder] Sensor '{name}' not found in env.scene.sensors "
                f"(available: [{', '.join(repr(s) for s in truncated)}{suffix}]).{hint}"
            )
        output = getattr(getattr(sensor, "data", None), "output", None)
        if output is None:
            raise RuntimeError(
                f"[VideoRecorder] Sensor '{name}' has no data output. "
                "Ensure the sensor is initialized and has been stepped at least once."
            )
        available_keys = frozenset(output.keys())
        try:
            sensor_key = sensor_key_for_gt_type(gt_type, available_keys)
        except (ValueError, KeyError):
            raise RuntimeError(
                f"[VideoRecorder] Sensor '{name}' has no '{gt_type}' output "
                f"(available: {sorted(available_keys)}). "
                f"Ensure the sensor's data_types includes '{gt_type}'."
            )
        data = output[sensor_key]
        # ProxyArray or torch.Tensor: shape (N, H, W, C)
        raw = data.torch if hasattr(data, "torch") else data
        return CameraFrameColorizer.colorize(
            raw[0],
            gt_type,
            depth_min=self.cfg.depth_colormap_min,
            depth_max=self.cfg.depth_colormap_max,
        )

    def _effective_output_dir(self) -> str:
        return self.cfg.output_dir or "videos"

    def _clip_path(self, index: int) -> str:
        return os.path.join(self._effective_output_dir(), f"{self.cfg.output_filename_prefix}_{index:04d}.mp4")

    def _next_clip_index(self) -> int:
        return max(self._existing_clip_indices(), default=-1) + 1

    def _existing_clip_indices(self) -> list[int]:
        output_dir = self._effective_output_dir()
        if not os.path.isdir(output_dir):
            return []

        pattern = re.compile(rf"^{re.escape(str(self.cfg.output_filename_prefix))}_(?P<index>\d+)\.mp4$")
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
            os.makedirs(self._effective_output_dir(), exist_ok=True)
            path = self._clip_path(self._clip_index)
            fps = self.cfg.fps
            if fps is None:
                base_fps = self._env.metadata.get("render_fps") if hasattr(self._env, "metadata") else None
                if base_fps is None:
                    step_dt = getattr(self._env, "step_dt", None)
                    base_fps = round(1.0 / step_dt) if step_dt else 30
                # frame_stride subsamples: one frame every N steps, so playback fps scales down.
                fps = max(1, round(base_fps / self.cfg.frame_stride))
            # Warn if the clip appears to be all-black (mean pixel < 2/255).
            # This can happen with Kit+Newton when cubric is unavailable.
            sample = self._frames[len(self._frames) // 2]
            mean_pixel = float(np.mean(sample))
            if mean_pixel < 2.0:
                logger.warning(
                    "[VideoRecorder] source=%r: sampled frame appears mostly black "
                    "(mean pixel value %.1f/255). For Kit+Newton, ensure cubric is available "
                    "to propagate Fabric transforms to the RTX renderer, or switch to "
                    "source='viz:newton_gl' for guaranteed capture.",
                    self.cfg.source,
                    mean_pixel,
                )
            clip = ImageSequenceClip(self._frames, fps=fps)
            clip.write_videofile(path, codec="libx264", audio=False, logger=None)
            logger.info("[VideoRecorder] Wrote %d frames to %s", len(self._frames), path)
            self._clip_index += 1
            self._maybe_delete_old_clips()
        except Exception:
            logger.exception("[VideoRecorder] Failed to write clip.")
        finally:
            self._frames = []
            self._recording = False

    def _maybe_delete_old_clips(self) -> None:
        if self.cfg.keep_last_n_clips is None:
            return
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
