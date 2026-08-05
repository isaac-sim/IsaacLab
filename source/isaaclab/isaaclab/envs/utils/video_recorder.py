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

try:
    from moviepy.editor import ImageSequenceClip
except ImportError:
    ImageSequenceClip = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from .video_recorder_cfg import VideoRecorderCfg

logger = logging.getLogger(__name__)

_VALID_SOURCE_KINDS = ("visualizer", "sensor")


def _parse_source(source: str) -> tuple[str, str, str]:
    """Parse a source string into (kind, type_or_name, sub).

    Source strings use ``:`` as the sole delimiter: ``"<kind>:<type>:<sub>"``.

    Returns:
        ``(kind, type_or_name, sub)`` where ``kind`` is ``"visualizer"`` or
        ``"sensor"``, ``type_or_name`` is the visualizer type or sensor name,
        and ``sub`` is the sub-channel (e.g. ``"tiled"``).
    """
    # Only lowercase the kind (parts[0]); preserve original casing for sensor names and visualizer types.
    parts = source.strip().split(":")
    kind = parts[0].lower() if parts else ""
    type_or_name = parts[1] if len(parts) > 1 else ""
    sub = parts[2].lower() if len(parts) > 2 else ""
    return kind, type_or_name, sub


class VideoRecorder:
    """Records one video stream per :class:`VideoRecorderCfg` entry.

    Instantiated by the env base class; ``step()`` is called once per env step
    after physics and rendering have completed.

    Raises:
        ImportError: If ``moviepy`` is not installed.
        ValueError: If :attr:`~VideoRecorderCfg.source` has an unrecognized kind.
        RuntimeError: On the first recording step if the requested visualizer or
            sensor cannot be found or does not support frame capture.
    """

    def __init__(self, cfg: VideoRecorderCfg, env: object):
        kind, _, _ = _parse_source(cfg.source)
        if kind not in _VALID_SOURCE_KINDS:
            raise ValueError(
                f"[VideoRecorder] Unrecognized source kind '{kind}' in source='{cfg.source}'. "
                f"Expected one of: {_VALID_SOURCE_KINDS}."
            )

        if ImageSequenceClip is None:
            raise ImportError("moviepy is required for video recording. Install it with: pip install 'moviepy<2'")

        self.cfg = cfg
        self._env = env
        self._frames: list[np.ndarray] = []
        self._step_count = 0
        self._frames_step_count = 0
        self._clip_index = 0
        self._recording = False
        # Set to True after the first unrecoverable frame-capture error so that
        # subsequent steps do not propagate the exception or repeat the log message.
        self._frame_error_logged: bool = False

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
            kind, type_or_name, sub = _parse_source(self.cfg.source)
            if kind == "visualizer":
                return self._frame_from_visualizer(type_or_name, sub)
            if kind == "sensor":
                return self._frame_from_sensor(type_or_name)
            # Unreachable: kind was validated in __init__, but keeps type checkers happy.
            return None  # pragma: no cover
        except RuntimeError as exc:
            logger.error(
                "[VideoRecorder] Frame capture failed; recording will be disabled for this stream. "
                "source=%r  error: %s",
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
                    f"[VideoRecorder] source='visualizer:{viz_type}' requested but no '{viz_type}' "
                    f"visualizer is active (active: {active or ['none']}). "
                    f"Pass --viz {viz_type} or add the corresponding VisualizerCfg to sim.visualizer_cfgs."
                )
        else:
            # Auto: pick the first active visualizer that supports frame capture.
            candidates = [v for v in visualizers if hasattr(v, "render_rgb_array")]
            if not candidates:
                active = [getattr(v.cfg, "visualizer_type", "unknown") for v in visualizers]
                raise RuntimeError(
                    "[VideoRecorder] source='visualizer' found no recording-capable visualizer "
                    f"(active: {active or ['none']}). "
                    "Pass --viz kit or --viz newton, or use source='sensor:<name>' to record from a scene sensor."
                )

        # Kit Replicator requires cubric to propagate Newton Fabric transforms to RTX's
        # scene delegate. Without cubric, frames will be black. Log a warning but allow
        # the capture to proceed — users with cubric available will get correct frames.
        if viz_type == "kit":
            physics_backend = getattr(getattr(sim, "physics_manager", None), "video_capture_backend", lambda: None)()
            if physics_backend == "newton_gl":
                logger.warning(
                    "[VideoRecorder] source='visualizer:kit' with Newton physics requires cubric "
                    "to propagate Fabric transforms to RTX. Frames may be black if cubric is "
                    "unavailable. Use source='visualizer:newton' for guaranteed capture."
                )

        viz = candidates[0]
        if sub == "tiled":
            if viz_type == "newton":
                # Newton captures its entire GL viewport (which shows the tiled camera panel
                # when tiled_cam_view=True on NewtonVisualizerCfg).  There is no separate
                # render_tiled_rgb_array() path for Newton.
                return viz.render_rgb_array()
            if not hasattr(viz, "render_tiled_rgb_array"):
                raise RuntimeError(
                    f"[VideoRecorder] source='visualizer:{viz_type}:tiled' requested but the "
                    f"'{viz_type}' visualizer does not support tiled capture."
                )
            return viz.render_tiled_rgb_array()

        return viz.render_rgb_array()

    def _frame_from_sensor(self, name: str) -> np.ndarray | None:
        # Note: only the "rgb" channel is currently supported for sensor sources.
        # Sub-channel specifiers such as "sensor:<name>:depth" are parsed but the sub
        # field is ignored — depth colorization is not implemented.  To record depth,
        # retrieve the depth tensor manually in a custom recorder subclass.
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
        if output is None or "rgb" not in output:
            raise RuntimeError(
                f"[VideoRecorder] Sensor '{name}' has no 'rgb' output. Ensure the sensor's data_types includes 'rgb'."
            )
        data = output["rgb"]
        # ProxyArray or torch.Tensor: shape (N, H, W, C)
        if hasattr(data, "torch"):
            tensor = data.torch
        else:
            tensor = data
        frame = tensor[0].cpu().numpy().astype(np.uint8)
        return frame[:, :, :3] if frame.ndim == 3 and frame.shape[2] >= 3 else frame

    def _effective_output_dir(self) -> str:
        return self.cfg.output_dir or "videos"

    def _clip_path(self, index: int) -> str:
        return os.path.join(self._effective_output_dir(), f"{self.cfg.output_filename_prefix}_{index:04d}.mp4")

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
        for index in range(max(0, cutoff)):
            path = self._clip_path(index)
            try:
                os.remove(path)
                logger.debug("[VideoRecorder] Deleted old clip %s", path)
            except FileNotFoundError:
                pass
