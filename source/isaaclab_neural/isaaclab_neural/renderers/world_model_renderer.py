# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer that publishes the frames a world model predicted instead of rendering a scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

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
    """Per-camera state: the frames the camera has already received."""

    seen: torch.Tensor
    """The manager's frame serial of each environment at the last copy, shape ``(N,)``."""

    version: int = -1
    """The manager's frame version at the last copy."""


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
        return _RenderData(seen=torch.full((spec.num_instances,), -1, dtype=torch.int64, device=self._sim.device))

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
        """Copy the environments whose frames changed since the last call, and only on steps where any did."""
        manager: WorldModelManager = self._sim.backend_manager
        if render_data.version < 0:
            _check_camera(camera_data, manager)
        mask = wp.to_torch(camera_data.frame_history_mask.warp)
        if manager.frame_version == render_data.version:
            mask.zero_()
            return
        render_data.version = manager.frame_version
        torch.ne(manager.frame_serial, render_data.seen, out=mask)
        render_data.seen.copy_(manager.frame_serial)
        history = wp.to_torch(camera_data.frame_history.warp)
        torch.where(mask[:, None, None, None, None], manager.frame_history, history, out=history)
        rgba = wp.to_torch(camera_data.output["rgba"].warp)
        last = torch.cat((history[:, -1], torch.full_like(history[:, -1, ..., :1], 255)), dim=-1)
        torch.where(mask[:, None, None, None], last, rgba, out=rgba)

    def cleanup(self, render_data: Any) -> None:
        pass


def _check_camera(camera_data: CameraData, manager: WorldModelManager) -> None:
    """Check that the camera keeps one chunk of frames at the model's image size.

    Raises:
        ValueError: If the camera has no frame history, or its length or image size differs from the model's.
    """
    expected = (manager.spec.chunk_length, *manager.spec.image_size)
    history = camera_data.frame_history
    actual = None if history is None else tuple(history.torch.shape[1:4])
    if actual != expected:
        raise ValueError(
            f"A world-model camera needs frame_history_length={expected[0]} and a {expected[1]} x {expected[2]} image,"
            f" the model's chunk length and image size; it has "
            + ("no frame history." if actual is None else f"{actual[0]} frames of {actual[1]} x {actual[2]}.")
        )
