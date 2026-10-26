# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-render PPISP kernel dispatch.

:class:`PpispPipeline` converts scene-linear HDR to LDR RGBA with a single Warp kernel.
:class:`~isaaclab_ppisp.PpispProcessor` composes it for cameras and owns the output buffer;
the caller owns the HDR input.
"""

from __future__ import annotations

import warp as wp

from .cfg import PpispCfg, normalize_ppisp_cfg
from .kernels import (
    PPISP_CONTROLLER_FEATURE_LEN,
    PPISP_CONTROLLER_PARAM_COUNT,
    apply_ppisp_to_rgba,
    apply_ppisp_to_rgba_with_controller_params,
    compute_ppisp_controller_params,
)


class PpispPipeline:
    """Post-render PPISP kernel applier.

    Owns the normalised :class:`PpispCfg` and dispatches the PPISP Warp kernel
    once per capture via :meth:`apply`.

    One pipeline instance applies to the whole Camera sensor batch. The PPISP
    Warp kernel takes scalar coefficients, so every cloned view in a tiled
    batch shares the same ISP configuration — there is no per-view ISP today.
    Per-view support would require packing the cfg into GPU arrays and indexing
    by ``camera_id`` inside the kernel.
    """

    def __init__(self, cfg: PpispCfg):
        """Initialize the PPISP pipeline.

        Normalises ``cfg`` on construction (validates input keys, fills
        defaults). Resolve USD camera attributes before construction, for example
        with :meth:`~isaaclab_ppisp.PpispProcessor.prepare_camera`.

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
