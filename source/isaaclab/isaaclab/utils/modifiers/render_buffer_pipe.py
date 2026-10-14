# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render buffer processing for an incoming observation tensor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ..images import ImageBufferSpec, RenderBufferPipe
from .modifier_base import ModifierBase

if TYPE_CHECKING:
    from .modifier_cfg import RenderBufferPipeModifierCfg


class RenderBufferPipeModifier(ModifierBase):
    """Apply a render buffer pipe to the tensor received from the preceding modifier.

    The modifier owns a pipe and returns a snapshot. Its output must preserve the input
    shape, as required by :class:`ModifierBase`. Multiple camera inputs or shape changes
    belong in :class:`~isaaclab.envs.mdp.processed_image` instead.
    """

    def __init__(self, cfg: RenderBufferPipeModifierCfg, data_dim: tuple[int, ...], device: str):
        super().__init__(cfg, data_dim, device)
        self._pipe = RenderBufferPipe(
            {cfg.input_name: ImageBufferSpec(tuple(data_dim), cfg.dtype, device, cfg.encoding, cfg.layout)},
            cfg.pipeline,
        )
        if cfg.output_name not in self._pipe.output_specs:
            self._pipe.close()
            raise ValueError(f"Image processing does not provide output {cfg.output_name!r}.")
        if self._pipe.output_specs[cfg.output_name].shape != tuple(data_dim):
            self._pipe.close()
            raise ValueError("Observation modifiers must preserve shape; use mdp.processed_image for shape changes.")

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self._pipe.apply_to_tensor(data, input_name=self._cfg.input_name, output_name=self._cfg.output_name)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._pipe.reset(env_ids)

    def close(self) -> None:
        self._pipe.close()
