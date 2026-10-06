# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPISP as an observation modifier over a camera's published render buffers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierBase, ModifierCfg, ModifierOutput

from .cfg import PpispCfg, PpispDiscoveryMode, resolve_and_normalize
from .pipeline import PpispPipeline

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import Camera


class PpispModifier(ModifierBase):
    """Apply PPISP once per published camera image and return the selected color output."""

    borrows_input = True

    def __init__(self, cfg: PpispModifierCfg, data_dim: tuple[int, ...], device: str):
        super().__init__(cfg, data_dim, device)
        self._camera: Camera | None = None
        self._pipeline: PpispPipeline | None = None
        self._rgba: wp.array | None = None
        self._last_capture: object | None = None
        self._outputs: dict[str, torch.Tensor] = {}
        self._cached: ModifierOutput | None = None
        self._radiance: torch.Tensor | None = None
        self._normalized: torch.Tensor | None = None

    @property
    def output_dim(self) -> tuple[int, ...]:
        """Shape of the selected output, including the environment dimension."""
        shape = (*self._data_dim[:-1], 4 if self._cfg.output == "rgba" else 3)
        return (shape[0], shape[3], shape[1], shape[2]) if self._cfg.permute else shape

    @property
    def outputs(self) -> dict[str, torch.Tensor]:
        """The cached RGB and RGBA views, available after the first call."""
        return self._outputs

    def __call__(self, env: ManagerBasedEnv, data: torch.Tensor | ModifierOutput) -> torch.Tensor | ModifierOutput:
        if self._camera is None:
            self._camera = env.scene[self._cfg.sensor_cfg.name]
        camera = self._camera
        if self._cfg.isp_cfg is None:
            image = camera.data.output[self._cfg.output].torch
            return self._format_output(image, data)
        if self._pipeline is None:
            resolved = resolve_and_normalize(
                self._cfg.isp_cfg, env.sim.stage, camera.camera_prim_paths[0] if camera.camera_prim_paths else None
            )
            if resolved is None:
                return self._format_output(camera.data.output[self._cfg.output].torch, data)
            self._pipeline = PpispPipeline(resolved)
        if self._cfg.input_source == "previous":
            if not isinstance(data, ModifierOutput) or "rgb_radiance" not in data.named:
                raise ValueError("PPISP requires a preceding modifier to produce 'rgb_radiance'.")
            radiance = data.named["rgb_radiance"]
        else:
            radiance = camera.render_outputs["rgb_radiance"].torch
        if radiance.ndim != 4 or radiance.shape[-1] != 3 or radiance.dtype != torch.float32:
            raise ValueError("PPISP requires NHWC float32 rgb_radiance with three channels.")
        if radiance.device != torch.device(self._device):
            raise ValueError(f"PPISP radiance is on {radiance.device}, expected {self._device}.")
        if self._radiance is not None and radiance.data_ptr() != self._radiance.data_ptr():
            raise RuntimeError("Camera render buffers were recreated. Recreate the PPISP modifier to rebind them.")
        self._radiance = radiance
        capture = next(
            (
                info["capture"]["frame"]
                for info in camera.data.info.values()
                if isinstance(info, dict) and "frame" in info.get("capture", {})
            ),
            None,
        )
        if self._cached is not None and capture is not None and capture is self._last_capture:
            return self._cached
        if self._rgba is None:
            self._rgba = wp.empty((*radiance.shape[:-1], 4), dtype=wp.uint8, device=self._device)
            rgba = wp.to_torch(self._rgba)
            self._outputs = {"rgba": rgba, "rgb": rgba[..., :3]}
        stream = wp.stream_from_torch(torch.cuda.current_stream(self._device)) if radiance.is_cuda else None
        with wp.ScopedStream(stream, sync_enter=True, sync_exit=True):
            hdr = wp.from_torch(radiance, dtype=wp.float32)
            self._pipeline.initialize(hdr)
            self._pipeline.apply(hdr, self._rgba)
        self._last_capture = capture
        self._cached = self._format_output(self._outputs[self._cfg.output], data, radiance)
        return self._cached

    def _format_output(
        self, image: torch.Tensor, data: torch.Tensor | ModifierOutput, radiance: torch.Tensor | None = None
    ) -> ModifierOutput:
        if self._cfg.normalize:
            if self._normalized is None or self._normalized.shape != image.shape:
                self._normalized = torch.empty(image.shape, dtype=torch.float32, device=image.device)
            self._normalized.copy_(image).div_(255.0)
            self._normalized.sub_(self._normalized.mean(dim=(1, 2), keepdim=True))
            image = self._normalized
        if self._cfg.permute:
            image = image.permute(0, 3, 1, 2)
        named = {**(data.named if isinstance(data, ModifierOutput) else {}), **self._outputs}
        if radiance is not None:
            named["rgb_radiance"] = radiance
        named[self._cfg.output] = image
        return ModifierOutput(image, named, self._cfg.output)

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.close()
        self._pipeline = None
        self._camera = None
        self._rgba = None
        self._outputs = {}
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
        self.isp_cfg = resolve_and_normalize(
            self.isp_cfg, env.sim.stage, camera.camera_prim_paths[0] if camera.camera_prim_paths else None
        )
        if self.isp_cfg is None:
            camera.request_render_inputs((self.output,))
        elif self.input_source == "camera":
            camera.request_render_inputs(("rgb_radiance",))
