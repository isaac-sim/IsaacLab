# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer that publishes the frames a world model predicted instead of rendering a scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
import warp as wp

from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec

if TYPE_CHECKING:
    from isaaclab.renderers import CameraRenderSpec
    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray

    from ..physics import WorldModelManager
    from .world_model_renderer_cfg import WorldModelRendererCfg


@dataclass
class _RenderData:
    """Per-camera state: the frame serials the camera has already received."""

    seen: np.ndarray


class WorldModelRenderer(BaseRenderer):
    """Copy the frames of the :class:`~isaaclab_neural.physics.WorldModelManager` into a camera.

    Every other renderer hook is a no-op. The model predicts a whole chunk per update, so this renderer fills the
    camera's frame history itself.
    """

    provides_frame_history: ClassVar[bool] = True

    cfg: WorldModelRendererCfg

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        return self.cfg.supported_output_types()

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        pass

    def create_render_data(self, spec: CameraRenderSpec) -> _RenderData:
        return _RenderData(seen=np.full(spec.num_instances, -1, np.int64))

    def set_outputs(self, render_data: _RenderData, output_data: dict[str, ProxyArray]) -> None:
        pass

    def update_transforms(self) -> None:
        pass

    def update_geometries(self) -> None:
        pass

    def update_camera(self, render_data: Any, positions: Any, orientations: Any, intrinsics: Any) -> None:
        pass

    def render(self, render_data: _RenderData) -> None:
        pass

    def read_output(self, render_data: _RenderData, camera_data: CameraData) -> None:
        manager: WorldModelManager = self._sim.backend_manager
        mask = torch.from_numpy(manager.frame_serial != render_data.seen)
        render_data.seen = manager.frame_serial.copy()
        history = wp.to_torch(camera_data.frame_history.warp)
        mask = mask.to(history.device)
        frames = torch.from_numpy(manager.frame_history).to(history.device)
        history[mask] = frames[mask]
        rgba = wp.to_torch(camera_data.output["rgba"].warp)
        rgba[mask, ..., :3] = frames[mask, -1]
        rgba[mask, ..., 3] = 255
        wp.to_torch(camera_data.frame_history_mask.warp).copy_(mask)

    def cleanup(self, render_data: Any) -> None:
        pass
