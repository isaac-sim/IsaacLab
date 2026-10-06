# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPISP as an observation modifier over a camera's published render buffers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab import sim as sim_utils
from isaaclab.utils.modifiers import ModifierBase

from .cfg import PpispCfg, PpispDiscoveryMode, resolve_and_normalize
from .pipeline import PpispPipeline

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import Camera

    from .modifier_cfg import PpispModifierCfg


class PpispModifier(ModifierBase):
    """Apply PPISP once per published camera image and return the selected color output."""

    @staticmethod
    def prepare_scene(cfg: PpispModifierCfg, env: ManagerBasedEnv) -> None:
        """Resolve USD settings and request render inputs before camera initialization."""
        camera = env.scene[cfg.sensor_cfg.name] if cfg.input_source == "camera" else None
        camera_path = None
        if camera is not None and isinstance(cfg.isp_cfg, PpispDiscoveryMode):
            camera_path = next(
                (str(prim.GetPath()) for prim in sim_utils.find_matching_prims(camera.cfg.prim_path, env.sim.stage)),
                None,
            )
        cfg.isp_cfg = resolve_and_normalize(cfg.isp_cfg, env.sim.stage, camera_path)
        if cfg.isp_cfg is None and camera is not None:
            camera.request_render_inputs((cfg.output,))
        elif camera is not None:
            camera.request_render_inputs(("rgb_radiance",))

    def __init__(self, cfg: PpispModifierCfg, data_dim: tuple[int, ...], *, env: ManagerBasedEnv):
        super().__init__(cfg, data_dim, env=env)
        self._camera: Camera | None = env.scene[cfg.sensor_cfg.name] if cfg.input_source == "camera" else None
        self._pipeline: PpispPipeline | None = None
        self._rgba: wp.array | None = None
        self._last_capture: object | None = None
        self._cached: torch.Tensor | None = None
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
        if self._cfg.isp_cfg is None:
            return self._format_output(data)
        if self._pipeline is None:
            if not isinstance(self._cfg.isp_cfg, PpispCfg):
                raise RuntimeError("Call PpispModifier.prepare_scene before simulation reset.")
            self._pipeline = PpispPipeline(self._cfg.isp_cfg)
        radiance = data
        if radiance.ndim != 4 or radiance.shape[-1] != 3 or radiance.dtype != torch.float32:
            raise ValueError("PPISP requires NHWC float32 rgb_radiance with three channels.")
        if radiance.device != torch.device(env.device):
            raise ValueError(f"PPISP radiance is on {radiance.device}, expected {env.device}.")
        if self._rgba is not None and radiance.shape[:-1] != self._rgba.shape[:-1]:
            raise ValueError("PPISP radiance shape changed after output buffers were allocated.")
        capture = None
        if self._camera is not None:
            capture = next(
                (
                    info["capture"]["frame"]
                    for info in self._camera.data.info.values()
                    if isinstance(info, dict) and "frame" in info.get("capture", {})
                ),
                None,
            )
        if self._cached is not None and capture is not None and capture is self._last_capture:
            return self._cached
        if self._rgba is None:
            self._rgba = wp.empty((*radiance.shape[:-1], 4), dtype=wp.uint8, device=env.device)
        stream = wp.stream_from_torch(torch.cuda.current_stream(env.device)) if radiance.is_cuda else None
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
        self._normalized = None


def ppisp_camera_input(env: ManagerBasedEnv, modifier_cfg: PpispModifierCfg) -> torch.Tensor:
    """Read the render buffer selected by a PPISP modifier's prepared configuration."""
    if isinstance(modifier_cfg.isp_cfg, PpispDiscoveryMode):
        raise RuntimeError("Call PpispModifier.prepare_scene before reading its camera input.")
    camera = env.scene[modifier_cfg.sensor_cfg.name]
    data_type = "rgb_radiance" if modifier_cfg.isp_cfg is not None else modifier_cfg.output
    return camera.render_outputs[data_type].torch
