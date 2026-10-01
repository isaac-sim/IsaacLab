# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for video recording from visualizers and scene sensors."""

from __future__ import annotations

import warnings

from ...utils import configclass
from ...visualizers.visualizer_cfg import VISUALIZER_ALIASES, VISUALIZER_TYPES

CAPTURE_VISUALIZER_TYPES = ("kit", "newton_gl", "newton_rtx")
"""Visualizer types a video can be recorded from; the streaming ``rerun`` and ``viser`` cannot."""

SENSOR_CHANNELS = ("rgb", "depth", "segmentation", "normals")
"""Channels a ``sensor:<name>:<channel>`` source can record."""


def parse_video_source(source: str) -> tuple[str, str, str]:
    """Split a :attr:`VideoRecorderCfg.source` into ``(kind, name, sub)``.

    ``kind`` is ``"viz"`` or ``"sensor"``; ``name`` is the canonical visualizer type (``""`` for a bare
    ``"viz"``) or the sensor name; ``sub`` is ``"streaming_view"``, a sensor channel, or ``""``. The prefix
    ``visualizer`` is the long form of ``viz``, as ``--visualizer`` is of ``--viz``; the deprecated ``newton``
    type is mapped to ``newton_gl`` with a :class:`DeprecationWarning`.
    A bare visualizer type is shorthand for ``viz:<type>``.

    Raises:
        ValueError: If *source* does not follow the source grammar of :class:`VideoRecorderCfg`.
    """
    if source in (*VISUALIZER_TYPES, *VISUALIZER_ALIASES):
        source = f"viz:{source}"
    kind, *parts = source.split(":")
    kind = "viz" if kind == "visualizer" else kind
    name, sub = (*parts, "", "")[:2]
    if kind == "viz" and not parts:
        return kind, "", ""
    if kind == "viz" and parts[1:] in ([], ["streaming_view"]) and name in (*VISUALIZER_TYPES, *VISUALIZER_ALIASES):
        if name in VISUALIZER_ALIASES:
            warnings.warn(
                f"Video source type {name!r} is deprecated. Use {VISUALIZER_ALIASES[name]!r} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return kind, VISUALIZER_ALIASES.get(name, name), sub
    if kind == "sensor" and 1 <= len(parts) <= 2 and name and sub in ("", *SENSOR_CHANNELS):
        return kind, name, sub
    raise ValueError(
        f"Invalid video source {source!r}: expected '<type>', 'viz', 'viz:<type>', 'viz:<type>:streaming_view' or "
        f"'sensor:<name>[:<channel>]', with <type> one of {', '.join(VISUALIZER_TYPES)} and <channel> one of "
        f"{', '.join(SENSOR_CHANNELS)}."
    )


@configclass
class VideoRecorderCfg:
    """Configuration for one video recording stream.

    A recording stream captures frames from a *source* — a visualizer or a named scene sensor — and writes
    them to an mp4 clip file.  Multiple ``VideoRecorderCfg`` entries on an env cfg produce independent
    simultaneous streams.

    Source string format
    --------------------
    Fields are colon-separated: ``"<kind>:<type>:<sub>"``.

    * ``"viz"``                        – the first capture-capable visualizer ``--viz`` selected, else a
      headless Newton GL visualizer added for the recording.
    * ``"viz:<type>"``                 – a ``kit``, ``newton_gl`` or ``newton_rtx`` visualizer, interactive
      camera: the one ``--viz`` selected, else a headless one added for the recording.
    * ``"viz:<type>:streaming_view"``  – the streaming camera panel of that visualizer (requires
      ``streaming_view=True`` on its config, e.g. :class:`~isaaclab_visualizers.kit.KitVisualizerCfg`).
    * ``"sensor:<name>"``              – scene sensor, RGB channel (default); no visualizer is added.
    * ``"sensor:<name>:<channel>"``    – scene sensor channel: ``rgb``, ``depth`` (colorized via the turbo
      colormap), ``segmentation`` or ``normals`` (colorized).

    The streaming ``rerun`` and ``viser`` visualizers cannot be recorded from. A visualizer added for the
    recording reuses the configured visualizer of its type in
    :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs`, e.g. its camera pose, else the type's default config.
    :func:`~isaaclab.app.launch_simulation` resolves ``"viz"`` and ``"viz:<type>"`` to the visualizer the
    recording uses. The camera position and resolution are configured on the visualizer cfg, not here.

    ``visualizer`` is the long form of the ``viz`` prefix (``"visualizer:kit"`` is ``"viz:kit"``). The deprecated
    ``newton`` type still works with a warning; use ``newton_gl``. A bare visualizer type is shorthand for
    ``viz:<type>`` (e.g. ``"newton_gl"`` is ``"viz:newton_gl"``).
    """

    source: str = "viz"
    """Recording source.  See class docstring for the source string format."""

    output_dir: str | None = None
    """Directory for output mp4 files (created on demand).

    ``None`` (default): when recording is enabled via ``--video``, the RL entrypoint sets
    this to ``<log_dir>/videos/<subdir>`` automatically.  Set an explicit path to override.
    """

    fps: int | None = None
    """Output video frame rate in frames per second.

    ``None`` (default): resolved automatically from the environment at recording time
    using ``env.metadata["render_fps"]`` when available, falling back to
    ``round(1.0 / env.step_dt)``.  Set an explicit integer to override.
    """

    video_length: int = 200
    """Number of env steps captured per clip.  Must be positive."""

    video_interval: int = 0
    """Start a new clip every ``video_interval`` env steps after :attr:`step_offset`.

    ``0`` means a single clip starts at :attr:`step_offset` and the recorder is inactive
    afterwards.  Set to a positive integer to record recurring clips at that cadence.
    Must be non-negative.
    """

    step_offset: int = 0
    """Number of env steps to skip before the first clip starts.  Defaults to 0 (record
    from the very first step).  Applies to both one-shot and recurring recordings.
    Must be non-negative.
    """

    frame_stride: int = 1
    """Capture one frame every ``frame_stride`` env steps within a clip.  Defaults to 1
    (capture every step).  Increase to sub-sample the recording — e.g. ``frame_stride=2``
    records half as many frames, halving file size at the cost of temporal resolution.
    A clip that captures ``video_length // frame_stride`` unique frames is still triggered
    and closed after ``video_length`` env steps.  Must be positive.
    """

    output_filename_prefix: str = "clip"
    """Prefix for output clip filenames.  Each clip is written as
    ``<output_dir>/<output_filename_prefix>_<index>.mp4``.

    Defaults to ``"clip"`` → ``clip_0000.mp4``, ``clip_0001.mp4``, …

    Set a descriptive prefix when multiple recorders share the same ``output_dir`` so their
    clips do not overwrite each other.  For example, with two recorders::

        VideoRecorderCfg(source="viz:kit",           output_dir="videos", output_filename_prefix="viewport"),
        VideoRecorderCfg(source="sensor:wrist_cam",  output_dir="videos", output_filename_prefix="wrist"),

    produces ``videos/viewport_0000.mp4`` and ``videos/wrist_0000.mp4`` side-by-side.
    """

    depth_colormap_min: float = 0.1
    """Near-clip [m] for the turbo depth colormap used when ``source`` ends with ``:depth``.
    Values closer than this are clamped to the minimum color."""

    depth_colormap_max: float = 10.0
    """Far-clip [m] for the turbo depth colormap used when ``source`` ends with ``:depth``.
    Values farther than this are clamped to the maximum color."""

    keep_last_n_clips: int | None = None
    """If set, delete older clips so that at most this many clips are kept on disk at any
    time.  Older clips (by index) are removed immediately after a new one is written.

    Defaults to ``None`` (keep all clips).

    Useful during long training runs with a recurring ``video_interval`` where retaining
    every clip would fill the disk.  For example, ``keep_last_n_clips=3`` with
    ``video_interval=1000`` keeps only the three most recently recorded clips::

        VideoRecorderCfg(
            source="viz:newton_gl",
            video_interval=1000,
            video_length=200,
            keep_last_n_clips=3,
        )
    """

    def validate_config(self) -> None:
        """Reject clip schedules the recorder cannot honor.

        Raises:
            ValueError: If a clip timing field is outside its documented range.
        """
        minimums = {"video_length": 1, "frame_stride": 1, "video_interval": 0, "step_offset": 0}
        invalid = [f"{name}={getattr(self, name)!r}" for name, low in minimums.items() if getattr(self, name) < low]
        if invalid:
            raise ValueError(
                f"Invalid VideoRecorderCfg for source={self.source!r}: {', '.join(invalid)}. "
                "video_length and frame_stride must be positive; video_interval and step_offset must be non-negative."
            )
