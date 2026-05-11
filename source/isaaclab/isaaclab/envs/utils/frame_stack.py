# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frame-stacking helper for camera-based RL tasks.

Provides :class:`FrameStackBuffer`, a ring buffer over the last ``N`` rendered frames
that tasks can use to supply explicit temporal observations to a policy.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class FrameStackBuffer:
    """Ring buffer that stacks the last ``frame_stack`` rendered frames along the channel dim.

    Example::

        self._stack = FrameStackBuffer(
            single_frame_shape=(self.num_envs, H, W, C),
            frame_stack=self.cfg.frame_stack,
            device=self.device,
        )
        # in _get_observations:
        stacked = self._stack.update(rgb)
        # in _reset_idx:
        self._stack.reset(env_ids)

    Args:
        single_frame_shape: Shape of one rendered frame, ``(num_envs, H, W, C)``.
        frame_stack: Number of frames to keep. Must be ``>= 1``; ``1`` is a passthrough.
        device: Torch device for the internal buffers.
        dtype: Torch dtype for the internal buffers. Defaults to :obj:`torch.uint8`.
    """

    def __init__(
        self,
        single_frame_shape: tuple[int, ...],
        frame_stack: int,
        device: str | torch.device,
        dtype: torch.dtype = torch.uint8,
    ):
        if frame_stack < 1:
            raise ValueError(f"frame_stack must be >= 1, got {frame_stack}.")
        if len(single_frame_shape) < 2:
            raise ValueError(
                f"single_frame_shape must have at least 2 dims (envs + channels), got {single_frame_shape}."
            )
        self.frame_stack: int = frame_stack
        self._single_shape: tuple[int, ...] = tuple(int(d) for d in single_frame_shape)
        self._num_envs: int = self._single_shape[0]
        self._channels: int = self._single_shape[-1]
        self._device = torch.device(device) if isinstance(device, str) else device
        self._dtype = dtype

        self._history: torch.Tensor = torch.zeros((frame_stack, *self._single_shape), device=self._device, dtype=dtype)
        self._stacked: torch.Tensor = torch.zeros(
            (*self._single_shape[:-1], self._channels * frame_stack), device=self._device, dtype=dtype
        )
        self._frame_idx: int = 0
        self._needs_init: torch.Tensor = torch.ones(self._num_envs, device=self._device, dtype=torch.bool)
        # CPU-side mirror of _needs_init.any() — avoids a GPU→CPU sync on the steady-state path.
        self._needs_init_cpu: bool = True

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Shape of the tensor returned by :meth:`update`, ``(num_envs, H, W, C * frame_stack)``."""
        return (*self._single_shape[:-1], self._channels * self.frame_stack)

    @property
    def output_channels(self) -> int:
        """Channel count of the stacked output (``= single_channels * frame_stack``)."""
        return self._channels * self.frame_stack

    def update(self, single_frame: torch.Tensor) -> torch.Tensor:
        """Push a new frame and return the stacked output.

        On the first :meth:`update` after construction or :meth:`reset` for an env, all
        history slots for that env are filled with ``single_frame`` so the policy never
        sees zero-padded warmup data.

        Args:
            single_frame: New rendered frame, shape ``(num_envs, H, W, C)``.

        Returns:
            Stacked tensor ``(num_envs, H, W, C * frame_stack)`` in oldest-to-newest
            channel order. This is the buffer's own storage — do not mutate it.
        """
        if single_frame.shape != self._single_shape:
            raise ValueError(
                f"single_frame shape {tuple(single_frame.shape)} does not match expected "
                f"{self._single_shape} (set at construction)."
            )

        if self._needs_init_cpu:
            init_ids = self._needs_init.nonzero(as_tuple=False).squeeze(-1)
            if init_ids.numel() > 0:
                for i in range(self.frame_stack):
                    self._history[i, init_ids] = single_frame[init_ids]
            self._needs_init.zero_()
            self._needs_init_cpu = False

        self._history[self._frame_idx].copy_(single_frame)

        # narrow + copy_ rebuild avoids per-frame torch.cat allocations.
        for i in range(self.frame_stack):
            src_slot = (self._frame_idx + 1 + i) % self.frame_stack
            self._stacked.narrow(-1, i * self._channels, self._channels).copy_(self._history[src_slot])

        self._frame_idx = (self._frame_idx + 1) % self.frame_stack
        return self._stacked

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Mark envs for history re-initialization on the next :meth:`update`.

        Args:
            env_ids: Indices of envs to reset. ``None`` resets all envs.
        """
        if env_ids is None:
            self._needs_init.fill_(True)
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long)
            self._needs_init[env_ids] = True
        self._needs_init_cpu = True
