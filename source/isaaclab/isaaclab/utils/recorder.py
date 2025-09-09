# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mokey-patch function to fix memory leak in gymnasium video recording wrapper.

Should replace the stop_recording function in :class:`gymnasium.wrappers.RecordVideo` class."""

from __future__ import annotations

import gc
import os
from gymnasium import error, logger


def stop_recording(self):
    """Stop current recording and saves the video."""
    assert self.recording, "stop_recording was called, but no recording was started"

    if len(self.recorded_frames) == 0:
        logger.warn("Ignored saving a video as there were zero frames to save.")
    else:
        try:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        except ImportError as e:
            raise error.DependencyNotInstalled('MoviePy is not installed, run `pip install "gymnasium[other]"`') from e

        clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
        moviepy_logger = None if self.disable_logger else "bar"
        path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
        clip.write_videofile(path, logger=moviepy_logger)

        del clip

    del self.recorded_frames
    self.recorded_frames = []
    self.recording = False
    self._video_name = None

    if self.gc_trigger and self.gc_trigger(self.episode_id):
        gc.collect()
