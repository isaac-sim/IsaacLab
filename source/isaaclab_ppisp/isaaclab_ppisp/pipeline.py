# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-render PPISP pipeline composed into renderer backends.

:class:`PpispPipeline` runs *after* a renderer fills its HDR scene-linear AOV,
converting HDR → LDR via a single Warp kernel. Renderer backends instantiate
this class when their cfg/spec carries a :class:`PpispCfg` and dispatch
:meth:`apply` once per render tick. The HDR scratch buffer is owned by the
renderer backend, not by this class.
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

    Constructed by renderer backends when ``spec.cfg.isp_cfg`` is set. Owns the
    normalised :class:`PpispCfg` and dispatches the PPISP Warp kernel once per
    render tick via :meth:`apply`.

    One pipeline instance applies to the whole Camera sensor batch. The PPISP
    Warp kernel takes scalar coefficients, so every cloned view in a tiled
    batch shares the same ISP configuration — there is no per-view ISP today.
    Per-view support would require packing the cfg into GPU arrays and indexing
    by ``camera_id`` inside the kernel.

    Today only :class:`PpispCfg` is accepted; future ISP implementations can
    either subclass or be selected by cfg type without changes to the backend
    renderers.
    """

    def __init__(self, cfg: PpispCfg):
        """Initialize the PPISP pipeline.

        Normalises ``cfg`` on construction (validates input keys, fills
        defaults).
        :class:`~isaaclab.sensors.camera.Camera` already normalises ``isp_cfg``
        before passing the :class:`~isaaclab.renderers.CameraRenderSpec` to
        the backend, so renderer backends pass a concrete, resolved config here.

        Args:
            cfg: The PPISP configuration.
        """
        normalized_cfg = normalize_ppisp_cfg(cfg)
        if normalized_cfg is None:
            raise ValueError("PpispPipeline requires a concrete PpispCfg.")
        self.cfg = normalized_cfg
        self._controller_weights_by_device: dict[str, wp.array] = {}
        self._controller_buffers_by_shape: dict[tuple[str, int, int, int], tuple[wp.array, ...]] = {}

    def apply(self, hdr: wp.array, rgba: wp.array) -> None:
        """Run the PPISP kernel: HDR scene-linear → LDR RGBA, in place on ``rgba``."""
        if self.cfg.controller_weights is None:
            apply_ppisp_to_rgba(hdr, rgba, self.cfg)
            return
        controller_params = self._compute_controller_params(hdr)
        apply_ppisp_to_rgba_with_controller_params(hdr, rgba, self.cfg, controller_params)

    def _compute_controller_params(self, hdr: wp.array) -> wp.array:
        """Run the exported PPISP controller and return a Warp view of ``(N, 9)`` params."""
        controller_weights = self.cfg.controller_weights
        assert controller_weights is not None

        device = str(hdr.device)
        weights = self._controller_weights_by_device.get(device)
        if weights is None:
            weights = wp.array(controller_weights, dtype=wp.float32, device=device)
            self._controller_weights_by_device[device] = weights

        features, controller_params = self._controller_buffers(hdr)
        compute_ppisp_controller_params(
            hdr,
            weights,
            features,
            controller_params,
            self.cfg.controller_prior_exposure,
            float(self.cfg.controller_responsivity),
        )
        return controller_params

    def _controller_buffers(self, hdr: wp.array) -> tuple[wp.array, ...]:
        """Return cached controller scratch buffers matching ``hdr`` shape/device."""
        num_cameras = int(hdr.shape[0])
        image_height = int(hdr.shape[1])
        image_width = int(hdr.shape[2])
        device = str(hdr.device)
        key = (device, num_cameras, image_height, image_width)
        buffers = self._controller_buffers_by_shape.get(key)
        if buffers is not None:
            return buffers

        # Camera output shape is fixed for a pipeline lifetime; cache buffers so
        # controller execution does not allocate every frame.
        buffers = (
            wp.empty((num_cameras, PPISP_CONTROLLER_FEATURE_LEN), dtype=wp.float32, device=device),
            wp.empty((num_cameras, PPISP_CONTROLLER_PARAM_COUNT), dtype=wp.float32, device=device),
        )
        self._controller_buffers_by_shape[key] = buffers
        return buffers
