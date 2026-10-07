# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPISP as a modifier for camera outputs and observation terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import warp as wp

from isaaclab.sim import SimulationContext
from isaaclab.utils.modifiers import ModifierBase

from .cfg import PpispCfg, auto_any_ppisp_cfg, normalize_ppisp_cfg
from .pipeline import PpispPipeline

if TYPE_CHECKING:
    from .cfg import PpispModifierCfg


class PpispModifier(ModifierBase):
    """Convert scene-linear ``rgb_radiance`` of shape ``(N, H, W, 3)`` to 8-bit color.

    The pipeline is shared through the simulation context by every modifier with equal resolved
    settings. PPISP has no temporal state, so :meth:`reset` does nothing.
    """

    def __init__(self, cfg: PpispModifierCfg, data_dim: tuple[int, ...], device: str):
        """Resolve the PPISP settings and allocate the output buffer.

        Args:
            cfg: Modifier configuration.
            data_dim: Input shape ``(N, H, W, 3)``.
            device: Device of the input and output.

        Raises:
            ValueError: If the input shape is not ``(N, H, W, 3)``.
        """
        super().__init__(cfg, data_dim, device)
        if len(data_dim) != 4 or data_dim[-1] != 3:
            raise ValueError(f"PPISP expects rgb_radiance with shape (N, H, W, 3), got {data_dim}.")
        sim = SimulationContext.instance()
        isp_cfg = _resolve_isp_cfg(cfg.isp_cfg, sim.stage if sim is not None else None)
        self._pipeline: PpispPipeline = (
            sim.get_or_create_backend(isp_cfg) if sim is not None else PpispPipeline(isp_cfg)
        )
        self._rgba = wp.empty((*data_dim[:-1], 4), dtype=wp.uint8, device=device)
        rgba = wp.to_torch(self._rgba)
        self._output = rgba if cfg.output == "rgba" else rgba[..., :3]

    def reset(self, env_ids: Sequence[int] | None = None):
        pass

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply PPISP to ``data``.

        Args:
            data: Scene-linear radiance, shape ``(N, H, W, 3)``, dtype float32.

        Returns:
            8-bit color of shape ``(N, H, W, 3)`` or ``(N, H, W, 4)``. The buffer is reused by the next call.
        """
        if data.dtype != torch.float32:
            raise ValueError(f"PPISP expects float32 rgb_radiance, got {data.dtype}.")
        # Order the Warp kernel with the Torch stream that produced the input and consumes the output.
        stream = wp.stream_from_torch(torch.cuda.current_stream(data.device)) if data.is_cuda else None
        with wp.ScopedStream(stream, sync_enter=True, sync_exit=True):
            self._pipeline.apply(wp.from_torch(data.contiguous()), self._rgba)
        return self._output


def _resolve_isp_cfg(isp_cfg: PpispCfg | None, stage: Any | None) -> PpispCfg:
    """Return normalized PPISP settings without changing the configured object."""
    if isp_cfg is None:
        isp_cfg = auto_any_ppisp_cfg(stage) if stage is not None else None
        if isp_cfg is None:
            isp_cfg = PpispCfg()
    else:
        isp_cfg = isp_cfg.copy()
    return normalize_ppisp_cfg(isp_cfg, stage=stage)
