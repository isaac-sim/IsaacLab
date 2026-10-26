# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPISP tensor processor for camera HDR output."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab.utils import replace
from isaaclab.utils.tensor_processors import TensorProcessor
from isaaclab.utils.warp import ProxyArray

from .cfg import resolve_and_normalize
from .pipeline import PpispPipeline
from .rtx_camera_overrides import apply_rtx_exposure_overrides

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.renderers import RendererCfg

    from .cfg import PpispProcessorCfg


class PpispProcessor(TensorProcessor):
    """Convert scene-linear HDR color to 8-bit color with PPISP.

    The input is a float32 ``(N, H, W, 3)`` array of scene-linear radiance, before exposure and camera
    response. The processor returns persistent uint8 arrays with the same batch and image dimensions:
    ``"rgba"`` has opaque alpha, and ``"rgb"`` is a zero-copy view of the same storage. Static coefficients
    apply to every view; controller weights predict exposure and color latents for each view.

    On a camera, :meth:`prepare_camera` resolves USD-authored coefficients and makes RTX HDR output
    scene-linear before the scene is exported to the renderer.
    """

    def __init__(self, cfg: PpispProcessorCfg, data: dict[str, ProxyArray]) -> None:
        """Validate the routing and allocate the color output.

        Args:
            cfg: Configuration with concrete PPISP coefficients.
            data: Input arrays keyed by name. Only the configured HDR input is inspected.

        Raises:
            ValueError: If the configuration routes unsupported names or the HDR input does not
                have shape ``(N, H, W, 3)``.
        """
        super().__init__(cfg)
        if len(cfg.inputs) != 1:
            raise ValueError(f"PPISP reads exactly one HDR input; received {cfg.inputs}.")
        if not cfg.outputs or not set(cfg.outputs) <= {"rgb", "rgba"}:
            raise ValueError(f"PPISP outputs must be 'rgb', 'rgba', or both; received {cfg.outputs}.")
        hdr = data[cfg.inputs[0]].warp
        if hdr.ndim != 4 or hdr.shape[3] != 3:
            raise ValueError(f"PPISP HDR input must have shape (N, H, W, 3); received {hdr.shape}.")
        self._pipeline = PpispPipeline(cfg.isp_cfg)
        self._rgba = wp.zeros((*hdr.shape[:3], 4), dtype=wp.uint8, device=hdr.device)
        rgb = wp.array(
            ptr=self._rgba.ptr,
            shape=(*hdr.shape[:3], 3),
            strides=self._rgba.strides,
            dtype=wp.uint8,
            device=self._rgba.device,
            copy=False,
        )
        outputs = {"rgb": ProxyArray(rgb), "rgba": ProxyArray(self._rgba)}
        self._outputs = {name: outputs[name] for name in cfg.outputs}

    def __call__(self, data: dict[str, ProxyArray]) -> dict[str, ProxyArray]:
        """Apply PPISP to the current HDR input.

        Args:
            data: Input arrays keyed by name. The HDR input keeps the dimensions used during construction.

        Returns:
            The persistent color outputs keyed by the configured output names.
        """
        hdr = data[self._cfg.inputs[0]].warp
        if hdr.shape[:3] != self._rgba.shape[:3]:
            raise ValueError(f"PPISP HDR input changed from {self._rgba.shape[:3]} to {hdr.shape[:3]}.")
        self._pipeline.apply(hdr, self._rgba)
        return dict(self._outputs)

    @classmethod
    def prepare_camera(
        cls, cfg: PpispProcessorCfg, stage: Usd.Stage, camera_prim_paths: list[str], renderer_cfg: RendererCfg
    ) -> PpispProcessorCfg | None:
        """Resolve the camera's PPISP coefficients and make RTX HDR output scene-linear.

        :class:`~isaaclab.sensors.camera.Camera` calls this during construction, before the scene is cloned
        or exported to a renderer. Discovery and :attr:`~isaaclab_ppisp.PpispCfg.camera_prim_path` are
        resolved once for the camera batch; camera-bound discovery reads the first camera prim. Isaac RTX and
        OVRTX apply the camera's exposure to their HDR output, so this disables automatic exposure and authors
        neutral exposure on every camera prim. The settings also affect the renderer's own color outputs and
        other users of these prims.

        Args:
            cfg: Processor configuration attached to the camera.
            stage: USD stage containing the camera prims.
            camera_prim_paths: Camera prims rendered by the camera.
            renderer_cfg: Renderer configuration selected by the camera.

        Returns:
            The configuration with concrete coefficients, or None when discovery finds no PPISP camera
            attributes. The camera then publishes the renderer's color output.
        """
        isp_cfg = resolve_and_normalize(cfg.isp_cfg, stage, camera_prim_paths[0])
        if isp_cfg is None:
            return None
        if renderer_cfg.renderer_type in ("isaac_rtx", "ovrtx"):
            apply_rtx_exposure_overrides(stage, camera_prim_paths)
        return replace(cfg, isp_cfg=isp_cfg)
