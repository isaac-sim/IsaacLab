# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer that copies camera images another component wrote into the simulation's OVStage."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import ovstage
import torch
import warp as wp

from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec

from ..stage import OvstageBackend, OvstageBackendCfg

if TYPE_CHECKING:
    from isaaclab.renderers import CameraRenderSpec
    from isaaclab.sensors.camera.camera_data import CameraData

    from .ovstage_image_renderer_cfg import OvstageImageRendererCfg


@dataclass
class _RenderData:
    """Per-camera binding to its prims' image attribute."""

    scene: OvstageBackend
    query: ovstage.Query
    attribute: ovstage.Token
    image_shape: tuple[int, int, int]
    """Height, width and channels of one image."""
    placement: ovstage.Placement | None
    """Device the read places the images on, or ``None`` for the host."""
    resources: contextlib.ExitStack
    """Releases the query and its path list."""


class OvstageImageRenderer(BaseRenderer):
    """Copy RGB images that another component wrote into the simulation's OVStage into cameras; render nothing.

    The producer, for example a learned world model, creates the simulation stage with
    ``sim.get_or_create_backend(OvstageBackendCfg())`` before the cameras initialize. Each step it writes every
    environment's image to that environment's camera prim as :attr:`~.OvstageImageRendererCfg.attribute`, a ``uint8``
    array of ``height * width * 3`` row-major RGB values, at the stage's ordinal, and seals it with
    :meth:`~isaaclab_ov.stage.OvstageBackend.commit` before the cameras update. A camera copies the latest sealed image
    of each environment; an environment without one keeps its previous image.
    """

    cfg: OvstageImageRendererCfg

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        return self.cfg.supported_output_types()

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        pass

    def create_render_data(self, spec: CameraRenderSpec) -> _RenderData:
        """Bind the camera's prims, one per environment, to their image attribute in the simulation stage.

        Raises:
            RuntimeError: If no component has created the simulation stage.
            ValueError: If the camera does not have one prim per environment.
        """
        scene = next((s for s in self._sim.get_backends(OvstageBackendCfg) if s.cfg.scene_key is None), None)
        if scene is None:
            raise RuntimeError(
                "OvstageImageRenderer reads images from the simulation's OVStage, but nothing has created it; the"
                " producer of the images creates it with sim.get_or_create_backend(OvstageBackendCfg()) before the"
                " cameras initialize."
            )
        if len(spec.camera_prim_paths) != spec.num_instances:
            raise ValueError(
                f"OvstageImageRenderer reads one image per environment, but the camera has {len(spec.camera_prim_paths)}"
                f" prims for {spec.num_instances} environments."
            )
        device = torch.device(spec.device)
        placement = None
        if device.type == "cuda":
            placement = ovstage.Placement(device_type=ovstage.DLDeviceType.kDLCUDA, device_id=device.index or 0)
        resources = contextlib.ExitStack()
        query = resources.enter_context(scene.query(spec.camera_prim_paths))
        # The stage releases its borrowers before itself; cleanup may come later and finds nothing left to release.
        scene.resources.callback(resources.close)
        return _RenderData(
            scene=scene,
            query=query,
            attribute=scene.paths.intern_token(self.cfg.attribute),
            image_shape=(spec.cfg.height, spec.cfg.width, 3),
            placement=placement,
            resources=resources,
        )

    def set_outputs(self, render_data: _RenderData, output_data: dict[str, Any]) -> None:
        pass

    def update_transforms(self) -> None:
        pass

    def update_geometries(self) -> None:
        pass

    def update_camera(self, render_data: _RenderData, positions: Any, orientations: Any, intrinsics: Any) -> None:
        pass

    def render(self, render_data: _RenderData) -> None:
        pass

    def read_output(self, render_data: _RenderData, camera_data: CameraData) -> None:
        """Copy the latest sealed image of each environment that has one into the camera's RGB output.

        Raises:
            ValueError: If an image does not have the camera's size.
        """
        stage = render_data.scene.stage
        rgba = wp.to_torch(camera_data.output["rgba"].warp) if "rgba" in camera_data.output else None
        rgb = wp.to_torch(camera_data.output["rgb"].warp) if "rgb" in camera_data.output else rgba[..., :3]
        read = stage.read_attributes(
            render_data.query,
            [render_data.attribute],
            ovstage.OrdinalRange.latest(ovstage.ORDINAL_MAX),
            placement=render_data.placement,
        )
        try:
            for group in read.groups():
                try:
                    column = group.columns[0]
                    for row in range(group.prim_count):
                        image = torch.from_dlpack(column.dlpack(column.data_row_index(row)))
                        if image.dtype != torch.uint8 or image.numel() != rgb[0].numel():
                            raise ValueError(
                                f"OvstageImageRenderer expects images of {render_data.image_shape} uint8 values in"
                                f" '{self.cfg.attribute}', but one holds {image.numel()} values of {image.dtype}."
                            )
                        env = group.prim_index(row)
                        rgb[env].copy_(image.view(render_data.image_shape))
                        if rgba is not None:
                            # RGB is a view of the RGBA buffer; its alpha is opaque.
                            rgba[env, ..., 3] = 255
                finally:
                    stage.release_group(group)
        finally:
            read.release().wait()

    def cleanup(self, render_data: _RenderData | None) -> None:
        if render_data is not None:
            render_data.resources.close()
