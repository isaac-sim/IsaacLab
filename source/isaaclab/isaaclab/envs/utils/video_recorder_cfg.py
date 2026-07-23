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

    * ``"visualizer"``              – first active recording-capable visualizer, interactive camera.
    * ``"visualizer:kit"``          – Kit visualizer, interactive viewport camera.
    * ``"visualizer:newton"``       – Newton GL visualizer, interactive camera.
    * ``"visualizer:newton:tiled"`` – Newton GL visualizer, tiled camera panel.
    * ``"sensor:<name>"``           – ``env.scene.sensors[<name>]``, rgb channel.

    The camera position and window resolution are configured on the visualizer cfg
    (e.g. :class:`~isaaclab_visualizers.kit.KitVisualizerCfg`), not here.
    """

    source: str = "visualizer"
    """Recording source.  See class docstring for the source string format."""

    output_dir: str = "videos"
    """Directory for output mp4 files (created on demand, relative to the working directory)."""

    fps: int = 30
    """Output video frame rate in frames per second."""

    video_length: int = 200
    """Number of env steps captured per clip."""

    video_interval: int = 0
    """Start a new clip every ``video_interval`` env steps.

    ``0`` means a single clip is started at the first step and the recorder is inactive
    afterwards (useful for fixed-length episode captures).  Set to a positive integer to
    record recurring clips spaced by that many env steps.
    """
