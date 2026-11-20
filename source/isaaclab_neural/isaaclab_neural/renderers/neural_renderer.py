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

from ..physics.neural_manager import NeuralManager

if TYPE_CHECKING:
    from isaaclab.renderers import CameraRenderSpec
    from isaaclab.sensors import CameraCfg
    from isaaclab.sensors.camera.camera_data import CameraData
    from isaaclab.utils.warp import ProxyArray

    from ..frame_buffer import FrameReader
    from .neural_renderer_cfg import NeuralRendererCfg


@dataclass
class _RenderData:
    """Per-camera state."""

    readers: dict[str, FrameReader]
    """The camera's binding to the world model's frames of each of its outputs, made when its render data is
    created."""


class NeuralRenderer(BaseRenderer):
    """Copy the frames of the :class:`~isaaclab_neural.physics.NeuralManager` into a camera.

    Every other renderer hook is a no-op. The model predicts a whole chunk per update, so this renderer fills the
    camera's frame history itself (:attr:`~.NeuralRendererCfg.provides_frame_history`).
    """

    cfg: NeuralRendererCfg

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        return self.cfg.supported_output_types()

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        pass

    def create_render_data(self, spec: CameraRenderSpec) -> _RenderData:
        """Bind each output of the camera to the world model's frames of it, and check that the camera keeps one chunk
        of them at the model's image size and the declared layout.

        Raises:
            ValueError: If the simulation does not run the world-model backend, the camera does not match the model,
                :attr:`~.NeuralRendererCfg.view` names no view of the model, or the model does not predict an output
                of the camera at its declared layout.
        """
        manager = self._sim.backend_manager
        if not isinstance(manager, NeuralManager):
            raise ValueError(
                "NeuralRenderer publishes the frames of the neural physics backend (NeuralCfg), but the"
                f" simulation runs {type(manager).__name__}."
            )
        _check_camera(spec.cfg, manager)
        readers = {}
        for name in dict.fromkeys(spec.cfg.data_types):
            buffer = manager.frames(self.cfg.view, name)
            _check_layout(name, buffer.frames, self.cfg.outputs[RenderBufferKind(name)])
            readers[name] = buffer.reader()
        return _RenderData(readers=readers)

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
        for name, reader in render_data.readers.items():
            history = camera_data.frame_history[name].torch
            # Every output of a view is published together, so either all of them changed or none did.
            if not reader.read_into(history, mask):
                return
            # The newest frame becomes the camera's output, written in place.
            output = wp.to_torch(camera_data.output[name].warp)
            torch.where(mask.view(-1, 1, 1, 1), history[:, -1], output, out=output)
        if "rgba" in camera_data.output and "rgb" in render_data.readers:
            # RGB is a view of the RGBA buffer; its alpha is opaque.
            wp.to_torch(camera_data.output["rgba"].warp)[..., 3].masked_fill_(mask[:, None, None], 255)

    def cleanup(self, render_data: Any) -> None:
        pass


def _check_camera(cfg: CameraCfg, manager: NeuralManager) -> None:
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


def _check_layout(name: str, frames: torch.Tensor, spec: RenderBufferSpec) -> None:
    """Check that the model's frames of an output have the layout the renderer declares for it.

    Raises:
        ValueError: If the channels or dtype differ.
    """
    channels, dtype = frames.shape[-1], wp.dtype_to_torch(spec.dtype)
    if (channels, frames.dtype) != (spec.channels, dtype):
        raise ValueError(
            f"The world model predicts '{name}' with {channels} channels of {frames.dtype}, but NeuralRendererCfg"
            f" declares {spec.channels} channels of {dtype}; set NeuralRendererCfg.outputs to the model's layout."
        )
