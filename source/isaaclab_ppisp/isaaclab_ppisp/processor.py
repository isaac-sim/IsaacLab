# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPISP dictionary processor shared by cameras and observations."""

from __future__ import annotations

from copy import deepcopy

import warp as wp

from isaaclab.utils.tensor_processors import DictTensorProcessor
from isaaclab.utils.warp import ProxyArray

from .cfg import PpispCfg
from .pipeline import PpispPipeline
from .processor_cfg import PpispProcessorCfg


class PpispProcessor(DictTensorProcessor):
    """Transform scene-linear HDR into uint8 color using persistent PPISP storage.

    Input is float32 NHWC with three channels. It must represent color before exposure and
    camera response; array storage cannot establish that semantic contract. Output has three
    RGB or four RGBA channels. Controllers may share immutable weights via configuration,
    while each processor owns its controller workspace and output buffer.
    """

    def __init__(self, cfg: PpispProcessorCfg, data: dict[str, ProxyArray]) -> None:
        super().__init__(cfg, data)
        if len(cfg.inputs) != 1 or len(cfg.outputs) != 1:
            raise ValueError("PPISP requires exactly one input and one output name.")
        self._input_name, self._output_name = cfg.inputs[0], cfg.outputs[0]
        source = data[self._input_name]
        if not isinstance(source, ProxyArray):
            raise TypeError("PPISP input must be a ProxyArray.")
        if len(source.shape) != 4 or source.shape[-1] != 3 or any(size <= 0 for size in source.shape):
            raise ValueError("PPISP input must have shape (N, H, W, 3) with positive dimensions.")
        if source.dtype is not wp.float32:
            raise ValueError("PPISP input must have dtype wp.float32.")
        channels = cfg.params.get("channels", 3)
        if channels not in (3, 4):
            raise ValueError("PPISP color output requires three RGB or four RGBA channels.")
        ppisp_cfg = cfg.params["cfg"]
        if not isinstance(ppisp_cfg, PpispCfg) or ppisp_cfg.camera_prim_path is not None:
            raise ValueError("PPISP requires a concrete PpispCfg; resolve USD camera attributes before construction.")
        self._shape, self._device = source.shape, source.device
        self._closed = False
        self._pipeline = PpispPipeline(deepcopy(ppisp_cfg), controller_weights=cfg.params.get("controller_weights"))
        try:
            self._pipeline.initialize(source.shape, source.device)
            self.outputs[self._output_name] = ProxyArray(
                wp.empty((*source.shape[:3], channels), dtype=wp.uint8, device=source.device)
            )
        except Exception:
            self.close()
            raise

    def __call__(self, data: dict[str, ProxyArray]) -> dict[str, ProxyArray]:
        """Read the current input and return the persistent named color tensor."""
        if self._closed:
            raise RuntimeError("Cannot execute a closed PPISP processor.")
        source = data[self._input_name]
        if source.shape != self._shape or source.dtype is not wp.float32 or source.device != self._device:
            raise ValueError("PPISP input must match the shape, dtype and device used during construction.")
        self._pipeline.apply(source.warp, self.outputs[self._output_name].warp)
        return dict(self.outputs)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._pipeline.close()
