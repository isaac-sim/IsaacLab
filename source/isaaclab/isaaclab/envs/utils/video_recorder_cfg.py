# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for video recording from visualizers and scene sensors."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass


@configclass
class VideoRecorderCfg:
    """Configuration for one video recording stream.

    A recording stream captures frames from a *source* — either an active visualizer
    (interactive or tiled camera) or a named scene sensor — and writes them to an mp4
    clip file.  Multiple ``VideoRecorderCfg`` entries on an env cfg produce independent
    simultaneous streams.

    Source string format
    --------------------
    Fields are colon-separated: ``"<kind>:<type>:<sub>"``.

    * ``"visualizer"``                        – first active recording-capable visualizer.
    * ``"visualizer:kit"``                    – Kit visualizer, interactive viewport camera.
    * ``"visualizer:newton"``                 – Newton GL visualizer, interactive camera.
    * ``"visualizer:newton_rtx"``             – Newton OVRTX path-traced interactive camera.
    * ``"visualizer:newton:streaming_view"``  – Newton GL streaming camera panel (requires
      ``streaming_view=True`` on :class:`~isaaclab_visualizers.newton.NewtonGLVisualizerCfg`).
    * ``"visualizer:kit:streaming_view"``     – Kit streaming camera panel (requires
      ``streaming_view=True`` on :class:`~isaaclab_visualizers.kit.KitVisualizerCfg`).
    * ``"sensor:<name>"``                     – scene sensor, RGB channel (default).
    * ``"sensor:<name>:rgb"``                 – scene sensor, RGB.
    * ``"sensor:<name>:depth"``               – scene sensor, depth colorized via turbo colormap.
    * ``"sensor:<name>:segmentation"``        – scene sensor, segmentation colorized.
    * ``"sensor:<name>:normals"``             – scene sensor, surface normals colorized.

    The camera position and resolution are configured on the visualizer cfg
    (e.g. :class:`~isaaclab_visualizers.kit.KitVisualizerCfg`), not here.
    """

    source: str = "visualizer"
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
    """Number of env steps captured per clip."""

    video_interval: int = 0
    """Start a new clip every ``video_interval`` env steps after :attr:`step_offset`.

    ``0`` means a single clip starts at :attr:`step_offset` and the recorder is inactive
    afterwards.  Set to a positive integer to record recurring clips at that cadence.
    """

    step_offset: int = 0
    """Number of env steps to skip before the first clip starts.  Defaults to 0 (record
    from the very first step).  Applies to both one-shot and recurring recordings.
    """

    frame_stride: int = 1
    """Capture one frame every ``frame_stride`` env steps within a clip.  Defaults to 1
    (capture every step).  Increase to sub-sample the recording — e.g. ``frame_stride=2``
    records half as many frames, halving file size at the cost of temporal resolution.
    A clip that captures ``video_length // frame_stride`` unique frames is still triggered
    and closed after ``video_length`` env steps.
    """

    output_filename_prefix: str = "clip"
    """Prefix for output clip filenames.  Each clip is written as
    ``<output_dir>/<output_filename_prefix>_<index>.mp4``.

    Defaults to ``"clip"`` → ``clip_0000.mp4``, ``clip_0001.mp4``, …

    Set a descriptive prefix when multiple recorders share the same ``output_dir`` so their
    clips do not overwrite each other.  For example, with two recorders::

        VideoRecorderCfg(source="visualizer:kit",    output_dir="videos", output_filename_prefix="viewport"),
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
            source="visualizer:newton",
            video_interval=1000,
            video_length=200,
            keep_last_n_clips=3,
        )
    """
