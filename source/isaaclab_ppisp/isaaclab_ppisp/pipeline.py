# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-render PPISP pipeline and the camera tensor processor that applies it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab.utils import replace
from isaaclab.utils.tensor_processors import TensorProcessor
from isaaclab.utils.warp import ProxyArray

from .cfg import PpispCfg, normalize_ppisp_cfg, resolve_and_normalize
from .kernels import (
    PPISP_CONTROLLER_FEATURE_LEN,
    PPISP_CONTROLLER_PARAM_COUNT,
    apply_ppisp_to_rgba,
    apply_ppisp_to_rgba_with_controller_params,
    compute_ppisp_controller_params,
)
from .rtx_camera_overrides import apply_rtx_exposure_overrides

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.renderers import RendererCfg

    from .cfg import PpispProcessorCfg


class PpispPipeline:
    """Post-render PPISP kernel applier.

    Owns the normalised :class:`PpispCfg` and dispatches the PPISP Warp kernel via :meth:`apply`.

    One pipeline instance applies to the whole Camera sensor batch. The PPISP
    Warp kernel takes scalar coefficients, so every cloned view in a tiled
    batch shares the same ISP configuration — there is no per-view ISP today.
    Per-view support would require packing the cfg into GPU arrays and indexing
    by ``camera_id`` inside the kernel.
    """

    def __init__(self, cfg: PpispCfg):
        """Initialize the PPISP pipeline.

        Normalises ``cfg`` on construction (validates input keys, fills
        defaults).

        Args:
            cfg: The PPISP configuration.
        """
        normalized_cfg = normalize_ppisp_cfg(cfg)
        if normalized_cfg is None:
            raise ValueError("PpispPipeline requires a concrete PpispCfg.")
        self.cfg = normalized_cfg
        self._controller_weights_by_device: dict[str, wp.array] = {}
        self._controller_buffers_by_shape: dict[tuple[str, int], tuple[wp.array, wp.array]] = {}

    def apply(self, hdr: wp.array, rgba: wp.array) -> None:
        """Run the PPISP kernel: HDR scene-linear → LDR RGBA, in place on ``rgba``."""
        if self.cfg.controller_weights is None:
            apply_ppisp_to_rgba(hdr, rgba, self.cfg)
            return
        device = str(hdr.device)
        if device not in self._controller_weights_by_device:
            self._controller_weights_by_device[device] = wp.array(
                self.cfg.controller_weights, dtype=wp.float32, device=device
            )
        key = (device, hdr.shape[0])
        if key not in self._controller_buffers_by_shape:
            self._controller_buffers_by_shape[key] = (
                wp.empty((hdr.shape[0], PPISP_CONTROLLER_FEATURE_LEN), dtype=wp.float32, device=device),
                wp.empty((hdr.shape[0], PPISP_CONTROLLER_PARAM_COUNT), dtype=wp.float32, device=device),
            )
        features, controller_params = self._controller_buffers_by_shape[key]
        compute_ppisp_controller_params(
            hdr, self._controller_weights_by_device[device], features, controller_params,
            self.cfg.controller_prior_exposure, self.cfg.controller_responsivity,
        )  # fmt: skip
        apply_ppisp_to_rgba_with_controller_params(hdr, rgba, self.cfg, controller_params)


class PpispProcessor(TensorProcessor):
    """Camera processor that converts scene-linear HDR to ``rgba`` and its ``rgb`` view with PPISP."""

    def __init__(self, cfg: PpispProcessorCfg, data: dict[str, ProxyArray]) -> None:
        super().__init__(cfg)
        self._pipeline = PpispPipeline(cfg.isp_cfg)
        hdr = data[cfg.inputs[0]].warp
        rgba = wp.zeros((*hdr.shape[:3], 4), dtype=wp.uint8, device=hdr.device)
        rgb = wp.array(ptr=rgba.ptr, shape=(*hdr.shape[:3], 3), strides=rgba.strides, dtype=wp.uint8, device=hdr.device)
        self._outputs = {"rgb": ProxyArray(rgb), "rgba": ProxyArray(rgba)}

    def __call__(self, data: dict[str, ProxyArray]) -> dict[str, ProxyArray]:
        self._pipeline.apply(data[self._cfg.inputs[0]].warp, self._outputs["rgba"].warp)
        return dict(self._outputs)

    @classmethod
    def prepare_camera(
        cls, cfg: PpispProcessorCfg, stage: Usd.Stage, camera_prim_paths: list[str], renderer_cfg: RendererCfg
    ) -> PpispProcessorCfg | None:
        """Resolve the camera's PPISP coefficients and neutralize RTX exposure on its prims.

        Isaac RTX and OVRTX apply camera exposure to their HDR output, so this disables automatic exposure
        and authors neutral exposure, which also affects other users of the prims. Returns None when
        discovery finds no PPISP camera attributes.
        """
        isp_cfg = resolve_and_normalize(cfg.isp_cfg, stage, camera_prim_paths[0])
        if isp_cfg is None:
            return None
        if renderer_cfg.renderer_type in ("isaac_rtx", "ovrtx"):
            apply_rtx_exposure_overrides(stage, camera_prim_paths)
        return replace(cfg, isp_cfg=isp_cfg)
