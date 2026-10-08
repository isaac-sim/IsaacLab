# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record rendered frames to an H.264 MP4 file with ffmpeg."""

import shutil
import subprocess
from pathlib import Path

import numpy as np


class VideoWriter:
    """Encode RGB(A) frames into an MP4 file at a fixed frame rate.

    Args:
        path: Output file; it must not exist.
        width: Frame width [px], even.
        height: Frame height [px], even.
        fps: Playback frame rate [frames/s].
    """

    def __init__(self, path: Path, width: int, height: int, fps: int = 30):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Recording a video requires ffmpeg on the PATH")
        if width % 2 or height % 2:
            raise ValueError("Video dimensions must be even")
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Main profile without B-frames plays in every common player.
        self._process = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps)]
            + ["-i", "pipe:0", "-an", "-c:v", "libx264", "-profile:v", "main", "-bf", "0", "-crf", "16"]
            + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE,
        )

    def write(self, image: np.ndarray) -> None:
        """Append one (height, width, 3 or 4) uint8 frame; an alpha channel is dropped."""
        self._process.stdin.write(np.ascontiguousarray(image[..., :3]).tobytes())

    def close(self) -> None:
        """Finish the file."""
        self._process.stdin.close()
        if self._process.wait() != 0:
            raise RuntimeError("ffmpeg failed; the video is incomplete")
