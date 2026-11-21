# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The frames a world model predicted for one output of one view, and the cameras that read them."""

from __future__ import annotations

import torch


class FrameBuffer:
    """The newest chunk of frames of one output of one view, for every environment.

    The world-model manager keeps one per view and output and publishes into it after a chunk or a reset. A camera
    reads it through its own :class:`FrameReader`, which hands over only the environments whose frames changed.
    """

    def __init__(self, num_envs: int, chunk_length: int, frame_shape: tuple[int, ...], dtype: torch.dtype, device: str):
        self.frames = torch.zeros((num_envs, chunk_length, *frame_shape), dtype=dtype, device=device)
        """Frames of the newest chunk, shape ``(N, T, H, W, C)``. After a reset, every frame is the start frame."""
        # Per environment, on the device: increases whenever the environment's frames change.
        self._serial = torch.zeros(num_envs, dtype=torch.int64, device=device)
        # On the host: increases whenever any environment's frames change, so readers skip unchanged steps without
        # reading the device.
        self._version = 0

    def publish(self, frames: torch.Tensor, env_mask: torch.Tensor | None = None) -> None:
        """Replace the frames of the selected environments, on the device.

        Args:
            frames: Frames ``(N, T, H, W, C)``, or one frame per environment ``(N, 1, H, W, C)`` that fills the
                chunk.
            env_mask: Environments to replace, shape ``(N,)``, ``bool``; all of them if ``None``.
        """
        if env_mask is None:
            self.frames.copy_(frames)
            self._serial += 1
        else:
            torch.where(env_mask.view(-1, *[1] * (self.frames.dim() - 1)), frames, self.frames, out=self.frames)
            self._serial += env_mask
        self._version += 1

    def reader(self) -> FrameReader:
        """Return a new reader for a camera that shows these frames."""
        return FrameReader(self)


class FrameReader:
    """A camera's view of a :class:`FrameBuffer`: what changed since the camera last read it."""

    def __init__(self, buffer: FrameBuffer):
        self._buffer = buffer
        self._seen = torch.full_like(buffer._serial, -1)
        self._version = -1

    def read_into(self, history: torch.Tensor, mask: torch.Tensor) -> bool:
        """Copy the frames of the environments that changed since the last call, in place on the device.

        Args:
            history: The camera's frame history of this output, shape ``(N, T, H, W, C)``.
            mask: Set to the environments whose frames were copied, shape ``(N,)``, ``bool``.

        Returns:
            Whether any frames changed. If not, nothing is copied, the mask is cleared and the device is not read.
        """
        buffer = self._buffer
        if buffer._version == self._version:
            mask.zero_()
            return False
        self._version = buffer._version
        torch.ne(buffer._serial, self._seen, out=mask)
        self._seen.copy_(buffer._serial)
        torch.where(mask.view(-1, *[1] * (history.dim() - 1)), buffer.frames, history, out=history)
        return True
