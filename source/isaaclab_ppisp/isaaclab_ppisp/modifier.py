# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPISP as an observation modifier over a camera's published render buffers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import warp as wp

from isaaclab import sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierBase, ModifierCfg

from .cfg import PpispCfg, PpispDiscoveryMode, resolve_and_normalize
from .pipeline import PpispPipeline

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import Camera


class PpispModifier(ModifierBase):
    """Apply PPISP once per published camera image and return the selected color output."""

    borrows_input = True

    def __init__(self, cfg: PpispModifierCfg, data_dim: tuple[int, ...], device: str, *, env: ManagerBasedEnv):
        super().__init__(cfg, data_dim, device, env=env)
        self._camera: Camera = env.scene[cfg.sensor_cfg.name]
        self._pipeline: PpispPipeline | None = None
        self._rgba: wp.array | None = None
        self._last_capture: object | None = None
        self._cached: torch.Tensor | None = None
        self._radiance: torch.Tensor | None = None
        self._normalized: torch.Tensor | None = None
        self._closed = False

    @property
    def output_dim(self) -> tuple[int, ...]:
        """Shape of the selected output, including the environment dimension."""
        shape = (*self._data_dim[:-1], 4 if self._cfg.output == "rgba" else 3)
        return (shape[0], shape[3], shape[1], shape[2]) if self._cfg.permute else shape

    def __call__(self, env: ManagerBasedEnv, data: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("Cannot read a closed PPISP modifier.")
        camera = self._camera
        if self._cfg.isp_cfg is None:
            image = camera.render_outputs[self._cfg.output].torch
            return self._format_output(image)
        if self._pipeline is None:
            if not isinstance(self._cfg.isp_cfg, PpispCfg):
                raise RuntimeError("Call PpispModifierCfg.prepare_scene before simulation reset.")
            self._pipeline = PpispPipeline(self._cfg.isp_cfg)
        if self._cfg.input_source == "previous":
            radiance = data
        else:
            radiance = camera.render_outputs["rgb_radiance"].torch
        if radiance.ndim != 4 or radiance.shape[-1] != 3 or radiance.dtype != torch.float32:
            raise ValueError("PPISP requires NHWC float32 rgb_radiance with three channels.")
        if radiance.device != torch.device(self._device):
            raise ValueError(f"PPISP radiance is on {radiance.device}, expected {self._device}.")
        same_radiance = self._radiance is not None and radiance.data_ptr() == self._radiance.data_ptr()
        if self._radiance is not None and self._cfg.input_source == "camera" and not same_radiance:
            raise RuntimeError("Camera render buffers were recreated. Recreate the PPISP modifier to rebind them.")
        if self._radiance is not None and radiance.shape != self._radiance.shape:
            raise ValueError("PPISP radiance shape changed after output buffers were allocated.")
        self._radiance = radiance
        capture = None
        if self._cfg.input_source == "camera":
            capture = next(
                (
                    info["capture"]["frame"]
                    for info in camera.data.info.values()
                    if isinstance(info, dict) and "frame" in info.get("capture", {})
                ),
                None,
            )
        if (
            self._cfg.input_source == "camera"
            and self._cached is not None
            and capture is not None
            and capture is self._last_capture
            and same_radiance
        ):
            return self._cached
        if self._rgba is None:
            self._rgba = wp.empty((*radiance.shape[:-1], 4), dtype=wp.uint8, device=self._device)
        stream = wp.stream_from_torch(torch.cuda.current_stream(self._device)) if radiance.is_cuda else None
        with wp.ScopedStream(stream, sync_enter=True, sync_exit=True):
            hdr = wp.from_torch(radiance, dtype=wp.float32)
            self._pipeline.apply(hdr, self._rgba)
        self._last_capture = capture
        rgba = wp.to_torch(self._rgba)
        self._cached = self._format_output(rgba if self._cfg.output == "rgba" else rgba[..., :3])
        return self._cached

    def _format_output(self, image: torch.Tensor) -> torch.Tensor:
        if self._cfg.normalize:
            if self._normalized is None or self._normalized.shape != image.shape:
                self._normalized = torch.empty(image.shape, dtype=torch.float32, device=image.device)
            self._normalized.copy_(image).div_(255.0)
            self._normalized.sub_(self._normalized.mean(dim=(1, 2), keepdim=True))
            image = self._normalized
        if self._cfg.permute:
            image = image.permute(0, 3, 1, 2)
        return image

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pipeline is not None:
            self._pipeline.close()
        self._pipeline = None
        self._rgba = None
        self._last_capture = None
        self._cached = None
        self._radiance = self._normalized = None


@configclass
class PpispModifierCfg(ModifierCfg):
    """Configure PPISP in :attr:`ObservationTermCfg.modifiers`."""

    func: type[PpispModifier] = PpispModifier
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera")
    isp_cfg: PpispCfg | PpispDiscoveryMode | None = PpispDiscoveryMode.AUTO_CAMERA
    output: Literal["rgb", "rgba"] = "rgb"
    input_source: Literal["camera", "previous"] = "camera"
    normalize: bool = False
    permute: bool = False

    def prepare_scene(self, env: ManagerBasedEnv) -> None:
        """Resolve USD settings and request private radiance before camera initialization."""
        camera = env.scene[self.sensor_cfg.name]
        camera_path = None
        if isinstance(self.isp_cfg, PpispDiscoveryMode):
            camera_path = next(
                (str(prim.GetPath()) for prim in sim_utils.find_matching_prims(camera.cfg.prim_path, env.sim.stage)),
                None,
            )
        self.isp_cfg = resolve_and_normalize(self.isp_cfg, env.sim.stage, camera_path)
        if self.isp_cfg is None:
            camera.request_render_inputs((self.output,))
        elif self.input_source == "camera":
            camera.request_render_inputs(("rgb_radiance",))
