# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer that publishes the frames a world model predicted instead of rendering a scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import warp as wp

from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec

from ..physics.world_model_manager import WorldModelManager

if TYPE_CHECKING:
    from isaaclab.renderers import CameraRenderSpec
    from isaaclab.sensors import CameraCfg
    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray

    from ..physics.frame_buffer import FrameReader
    from .world_model_renderer_cfg import WorldModelRendererCfg


@dataclass
class _RenderData:
    """Per-camera state."""

    reader: FrameReader
    """The camera's binding to the world model's frames, made when its render data is created."""


class WorldModelRenderer(BaseRenderer):
    """Copy the frames of the :class:`~isaaclab_neural.physics.WorldModelManager` into a camera.

    Every other renderer hook is a no-op. The model predicts a whole chunk per update, so this renderer fills the
    camera's frame history itself (:attr:`~.WorldModelRendererCfg.provides_frame_history`).
    """

    cfg: WorldModelRendererCfg

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        return self.cfg.supported_output_types()

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        pass

    def create_render_data(self, spec: CameraRenderSpec) -> _RenderData:
        """Bind the camera to the world model's frames and check that it keeps one chunk at the model's image size.

        Raises:
            ValueError: If the simulation does not run the world-model backend, the camera does not match the model,
                or :attr:`~.WorldModelRendererCfg.view` names no view of the model.
        """
        manager = self._sim.backend_manager
        if not isinstance(manager, WorldModelManager):
            raise ValueError(
                "WorldModelRenderer publishes the frames of the world-model physics backend (WorldModelCfg), but the"
                f" simulation runs {type(manager).__name__}."
            )
        _check_camera(spec.cfg, manager)
        return _RenderData(reader=manager.frames(self.cfg.view).reader())

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
        mask = wp.to_torch(camera_data.frame_history_mask.warp)
        history = wp.to_torch(camera_data.frame_history.warp)
        if not render_data.reader.read_into(history, mask):
            return
        # The newest frame becomes the camera's image, written in place: RGB from the frame, alpha opaque.
        rgba = wp.to_torch(camera_data.output["rgba"].warp)
        torch.where(mask[:, None, None, None], history[:, -1], rgba[..., :3], out=rgba[..., :3])
        rgba[..., 3].masked_fill_(mask[:, None, None], 255)

    def cleanup(self, render_data: Any) -> None:
        pass


def _check_camera(cfg: CameraCfg, manager: WorldModelManager) -> None:
    """Check that the camera keeps one chunk of frames at the model's image size.

    Raises:
        RuntimeError: If the model is not built yet; the articulation builds it and initializes before cameras.
        ValueError: If the camera has no frame history, or its length or image size differs from the model's.
    """
    if manager.spec is None:
        raise RuntimeError("The world model is not built yet: the articulation must initialize before the cameras.")
    expected = (manager.spec.chunk_length, *manager.spec.image_size)
    actual = (cfg.frame_history_length, cfg.height, cfg.width)
    if actual != expected:
        raise ValueError(
            f"A world-model camera needs frame_history_length={expected[0]} and a {expected[1]} x {expected[2]} image,"
            f" the model's chunk length and image size; it has "
            + ("no frame history." if not actual[0] else f"{actual[0]} frames of {actual[1]} x {actual[2]}.")
        )
