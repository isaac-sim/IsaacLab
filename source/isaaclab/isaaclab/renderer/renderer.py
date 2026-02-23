# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer base class matching the Newton-style interface.

RenderData is renderer-specific: each renderer implements create_render_data() and returns
its own RenderData type. Lifecycle: initialize() -> create_render_data(sensor) -> per frame:
update_transforms(), update_camera(), render(), write_output().
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from .renderer_cfg import RendererCfg

if TYPE_CHECKING:
    import torch

    from isaaclab.sensors import SensorBase


class RendererBase(ABC):
    """Base class for renderers.

    Lifecycle: __init__() -> initialize() -> create_render_data(sensor) -> per frame:
    update_transforms(), update_camera(), render(), write_output().
    """

    def __init__(self, cfg: RendererCfg):
        self.cfg = cfg
        self._height = cfg.height
        self._width = cfg.width
        self._num_envs = cfg.num_envs
        self._num_cameras = 1  # TODO: currently only supports 1 camera per environment
        self._data_types: list[str] = []

    def initialize(self, stage=None, camera_prim_path=None):
        """Initialize the renderer. Subclasses use stage and camera_prim_path as needed."""
        raise NotImplementedError("initialize() is not implemented.")

    def create_render_data(self, sensor: SensorBase) -> Any:
        """Create renderer-specific RenderData. Each renderer returns its own type."""
        raise NotImplementedError("create_render_data() is not implemented.")

    def set_outputs(self, render_data: Any, output_data: dict[str, "torch.Tensor"]) -> None:
        """Set output targets (e.g. wrap camera tensors). Override; default no-op."""
        pass

    def update_transforms(self) -> None:
        """Update scene transforms (e.g. sync physics to renderer)."""
        raise NotImplementedError("update_transforms() is not implemented.")

    def update_camera(
        self,
        render_data: Any,
        positions: "torch.Tensor",
        orientations: "torch.Tensor",
        intrinsics: "torch.Tensor",
    ) -> None:
        """Update camera state in render_data (poses, rays, etc.)."""
        raise NotImplementedError("update_camera() is not implemented.")

    def render(self, render_data: Any) -> None:
        """Render the scene into the provided RenderData."""
        raise NotImplementedError("render() is not implemented.")

    def write_output(
        self,
        render_data: Any,
        output_name: str,
        output_data: "torch.Tensor",
    ) -> None:
        """Copy from render_data to the given output tensor."""
        raise NotImplementedError("write_output() is not implemented.")

    def step(self) -> None:
        """Step the renderer."""
        raise NotImplementedError("step() is not implemented.")

    def reset(self) -> None:
        """Reset the renderer."""
        raise NotImplementedError("reset() is not implemented.")

    def clone(self, cameras):
        """TODO: Clone the camera in renderer."""
        raise NotImplementedError("clone() is not implemented.")
