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

from isaaclab import sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.utils.modifiers import ModifierBase

from .cfg import PpispCfg, auto_any_ppisp_cfg, auto_camera_ppisp_cfg, normalize_ppisp_cfg
from .pipeline import PpispPipeline

if TYPE_CHECKING:
    from isaaclab.sensors import SensorBase

    from .cfg import PpispModifierCfg


class PpispModifier(ModifierBase):
    """Convert scene-linear ``rgb_radiance`` of shape ``(N, H, W, 3)`` to 8-bit color.

    Settings are resolved once, from :attr:`PpispModifierCfg.isp_cfg` or, when it is None, from USD:
    the owning camera's prim when the modifier runs on a camera, otherwise the first camera on the stage
    with ``ppisp:*`` attributes. The pipeline is shared through the simulation context by every modifier
    with equal resolved settings. PPISP has no temporal state, so :meth:`reset` does nothing.
    """

    def __init__(self, cfg: PpispModifierCfg, data_dim: tuple[int, ...], device: str):
        """Allocate the output buffer.

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
        self._camera_prim_path: str | None = None
        self._pipeline: PpispPipeline | None = None
        self._rgba = wp.empty((*data_dim[:-1], 4), dtype=wp.uint8, device=device)
        rgba = wp.to_torch(self._rgba)
        self._output = rgba if cfg.output == "rgba" else rgba[..., :3]

    @property
    def output_dim(self) -> tuple[int, ...]:
        """Shape of the 8-bit output: ``(N, H, W, 3)`` for ``rgb``, ``(N, H, W, 4)`` for ``rgba``."""
        return tuple(self._output.shape)

    def bind_sensor(self, sensor: SensorBase) -> None:
        """Read USD settings from the owning camera's first prim when no settings are configured."""
        sim = SimulationContext.instance()
        prim = sim_utils.find_first_matching_prim(sensor.cfg.prim_path, sim.stage if sim is not None else None)
        self._camera_prim_path = str(prim.GetPath()) if prim is not None else None

    def reset(self, env_ids: Sequence[int] | None = None):
        pass

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply PPISP to ``data``.

        Args:
            data: Scene-linear radiance, shape ``(N, H, W, 3)``, dtype float32.

        Returns:
            8-bit color with shape :attr:`output_dim`. The buffer is reused by the next call.
        """
        if data.dtype != torch.float32:
            raise ValueError(f"PPISP expects float32 rgb_radiance, got {data.dtype}.")
        pipeline = self._pipeline if self._pipeline is not None else self._create_pipeline()
        # Order the Warp kernel with the Torch stream that produced the input and consumes the output.
        stream = wp.stream_from_torch(torch.cuda.current_stream(data.device)) if data.is_cuda else None
        with wp.ScopedStream(stream, sync_enter=True, sync_exit=True):
            pipeline.apply(wp.from_torch(data.contiguous()), self._rgba)
        return self._output

    def _create_pipeline(self) -> PpispPipeline:
        sim = SimulationContext.instance()
        isp_cfg = _resolve_isp_cfg(self._cfg.isp_cfg, sim.stage if sim is not None else None, self._camera_prim_path)
        self._pipeline = sim.get_or_create_backend(isp_cfg) if sim is not None else PpispPipeline(isp_cfg)
        return self._pipeline


def _resolve_isp_cfg(isp_cfg: PpispCfg | None, stage: Any | None, camera_prim_path: str | None) -> PpispCfg:
    """Return normalized PPISP settings without changing the configured object."""
    if isp_cfg is not None:
        return normalize_ppisp_cfg(isp_cfg.copy(), stage=stage)
    discovered = None
    if stage is not None:
        if camera_prim_path is not None:
            discovered = auto_camera_ppisp_cfg(stage, camera_prim_path)
        if discovered is None:
            discovered = auto_any_ppisp_cfg(stage)
    return normalize_ppisp_cfg(discovered if discovered is not None else PpispCfg(), stage=stage)
