# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extended :class:`gymnasium.wrappers.RecordVideo` with a video retention policy.

Use :class:`RecordVideoWrapper` as a drop-in replacement for
:class:`gymnasium.wrappers.RecordVideo` when ``--video_keep_last`` is required.
"""

from __future__ import annotations

import glob
import logging
import os

import gymnasium as gym

logger = logging.getLogger(__name__)


class RecordVideoWrapper(gym.wrappers.RecordVideo):
    """Drop-in replacement for :class:`gymnasium.wrappers.RecordVideo` with a video retention policy.

    Identical to the base wrapper in every respect except that, when *video_keep_last* is
    set, the output folder is pruned after each clip finishes so that at most
    ``video_keep_last`` ``.mp4`` files are kept (oldest files are removed first).

    Args:
        env: The environment to wrap.
        video_folder: Directory where video files are written.
        video_keep_last: Maximum number of ``.mp4`` files to retain in *video_folder*.
            When ``None`` (default) no pruning is performed and all videos are kept.
        **kwargs: All remaining keyword arguments are forwarded unchanged to
            :class:`gymnasium.wrappers.RecordVideo`.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        *,
        video_keep_last: int | None = None,
        **kwargs,
    ):
        if video_keep_last is not None and video_keep_last < 1:
            raise ValueError(f"video_keep_last must be >= 1 or None, got {video_keep_last}.")
        super().__init__(env, video_folder, **kwargs)
        self.video_keep_last = video_keep_last

    def stop_recording(self):
        """Stop current recording, save the video, then prune old clips."""
        super().stop_recording()
        if self.video_keep_last is not None:
            self._prune_old_videos()

    def start_recording(self, video_name: str):
        """Start a new recording, resetting the frame-skip counter on the underlying VideoRecorder."""
        super().start_recording(video_name)
        # Reset the frame-skip counter so the first step of every new clip always
        # triggers a fresh GPU render, regardless of where the previous clip ended.
        video_recorder = getattr(self.env, "video_recorder", None)
        if video_recorder is not None and hasattr(video_recorder, "reset_frame_counter"):
            video_recorder.reset_frame_counter()

    def _prune_old_videos(self) -> None:
        """Remove the oldest ``.mp4`` files so that at most ``video_keep_last`` remain."""
        all_files = sorted(glob.glob(os.path.join(self.video_folder, "*.mp4")), key=os.path.getmtime)
        files_to_delete = all_files[: -self.video_keep_last]
        for path in files_to_delete:
            try:
                os.remove(path)
                logger.info(f"Removed old video (keep_last={self.video_keep_last}): {path}")
            except OSError as exc:
                logger.warning(f"Could not remove old video {path}: {exc}")
