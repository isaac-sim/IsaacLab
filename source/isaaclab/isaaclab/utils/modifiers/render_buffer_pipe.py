# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render buffer processing for an incoming observation tensor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from ..images import RenderBufferPipe
from ..warp import ProxyArray
from .modifier_base import ModifierBase

if TYPE_CHECKING:
    from .modifier_cfg import RenderBufferPipeModifierCfg


class RenderBufferPipeModifier(ModifierBase):
    """Apply a render buffer pipe to the tensor received from the preceding modifier.

    The modifier owns a pipe and returns a snapshot. Its output must preserve the input
    shape, as required by :class:`ModifierBase`. The first incoming tensor prepares the pipe;
    subsequent calls reuse its output storage. Multiple camera inputs or shape changes belong
    in :class:`~isaaclab.envs.mdp.processed_image` instead.
    """

    def __init__(self, cfg: RenderBufferPipeModifierCfg, data_dim: tuple[int, ...], device: str):
        super().__init__(cfg, data_dim, device)
        self._pipe: RenderBufferPipe | None = None
        self._closed = False

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("The render buffer pipe modifier is closed.")
        if tuple(data.shape) != tuple(self._data_dim):
            raise ValueError(
                f"Observation modifier input shape must be {tuple(self._data_dim)}, got {tuple(data.shape)}."
            )
        if self._pipe is None:
            stream = wp.stream_from_torch(torch.cuda.current_stream(data.device)) if data.is_cuda else None
            with wp.ScopedStream(stream, sync_enter=True, sync_exit=True):
                pipe = RenderBufferPipe({self._cfg.input_name: ProxyArray(wp.from_torch(data))}, self._cfg.pipeline)
                try:
                    if self._cfg.output_name not in pipe.outputs:
                        raise ValueError(f"Image processing does not provide output {self._cfg.output_name!r}.")
                    if pipe.outputs[self._cfg.output_name].shape != tuple(self._data_dim):
                        raise ValueError(
                            "Observation modifiers must preserve shape; use mdp.processed_image for shape changes."
                        )
                except Exception:
                    pipe.close()
                    raise
                self._pipe = pipe
        return self._pipe.apply_to_tensor(data, input_name=self._cfg.input_name, output_name=self._cfg.output_name)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if self._pipe is not None:
            self._pipe.reset(env_ids)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pipe is not None:
            self._pipe.close()
