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

from typing import Any

import warp as wp

from .cfg import PpispCfg, normalize_ppisp_cfg
from .kernels import apply_ppisp_to_rgba


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

    def __init__(self, cfg: PpispCfg, stage: Any = None):
        """Initialize the PPISP pipeline.

        Normalises ``cfg`` on construction (validates input keys, fills
        defaults, and — when ``cfg.shader_prim_path`` is set and ``stage`` is
        non-``None`` — merges shader-authored values with user overrides).
        :class:`~isaaclab.sensors.camera.Camera` already normalises ``isp_cfg``
        before passing the :class:`~isaaclab.renderers.CameraRenderSpec` to
        the backend, so renderer backends typically pass ``stage=None`` here.

        Args:
            cfg: The PPISP configuration.
            stage: Optional USD stage used to resolve ``cfg.shader_prim_path``.
        """
        self.cfg = normalize_ppisp_cfg(cfg, stage=stage)

    def apply(self, hdr: wp.array, rgba: wp.array) -> None:
        """Run the PPISP kernel: HDR scene-linear → LDR RGBA, in place on ``rgba``."""
        apply_ppisp_to_rgba(hdr, rgba, self.cfg)
