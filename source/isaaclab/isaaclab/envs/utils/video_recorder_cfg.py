# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder`."""

from __future__ import annotations

from isaaclab.utils import configclass

from .video_recorder import VideoRecorder


@configclass
class VideoRecorderCfg:
    """Configuration for :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder`.

    Set :attr:`class_type` to a custom subclass of
    :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder` to swap the
    video-capture implementation (e.g. an Option-B pipeline that only renders
    ``video_num_tiles`` cameras on the GPU) without modifying any environment code.
    """

    class_type: type = VideoRecorder
    """The recorder class to instantiate.  Must accept ``(cfg, scene)`` as constructor arguments.
    Defaults to :class:`~isaaclab.envs.utils.video_recorder.VideoRecorder`.
    """

    video_num_tiles: int = -1
    """Number of environment tiles to include in each video frame when using ``render_mode="rgb_array"``.
    Defaults to -1, which renders all environments.

    Environments are arranged into a square grid of size
    ``ceil(sqrt(video_num_tiles)) * ceil(sqrt(video_num_tiles))``, with unused slots filled with
    black. For example:

    * ``-1``: all environments (default)
    * ``1``: single environment (1*1)
    * ``4``: first 4 environments (2*2 grid)
    * ``9``: first 9 environments (3*3 grid)

    CLI example: ``env.video_recorder.video_num_tiles=9``
    """
