# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton Warp renderer for tiled camera rendering."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import newton
import torch
import warp as wp

from isaaclab.renderers import BaseRenderer, RenderBufferKind, RenderBufferSpec
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import convert_camera_frame_orientation_convention

from .newton_warp_renderer_cfg import NewtonWarpRendererCfg

if TYPE_CHECKING:
    from isaaclab.physics import BaseSceneDataProvider
    from isaaclab.sensors.camera.camera_data import CameraData

logger = logging.getLogger(__name__)


class RenderData:
    # Back-compat alias for callers of ``RenderData.OutputNames``.
    OutputNames = RenderBufferKind

    @dataclass
    class CameraOutputs:
        color_image: wp.array(dtype=wp.uint32, ndim=4) = None
        albedo_image: wp.array(dtype=wp.uint32, ndim=4) = None
        depth_image: wp.array(dtype=wp.float32, ndim=4) = None
        normals_image: wp.array(dtype=wp.vec3f, ndim=4) = None
        instance_segmentation_image: wp.array(dtype=wp.uint32, ndim=4) = None

    def __init__(self, newton_sensor: newton.sensors.SensorTiledCamera, spec: CameraRenderSpec):
        self.newton_sensor = newton_sensor

        self.num_cameras = 1

        self.camera_rays: wp.array(dtype=wp.vec3f, ndim=4) = None
        self.camera_transforms: wp.array(dtype=wp.transformf, ndim=2) = None
        self.outputs = RenderData.CameraOutputs()
        self.width = getattr(spec.cfg, "width", 100)
        self.height = getattr(spec.cfg, "height", 100)

    def set_outputs(self, output_data: dict[str, torch.Tensor]):
        for output_name, tensor_data in output_data.items():
            if output_name == RenderBufferKind.RGBA:
                self.outputs.color_image = self._from_torch(tensor_data, dtype=wp.uint32)
            elif output_name == RenderBufferKind.ALBEDO:
                self.outputs.albedo_image = self._from_torch(tensor_data, dtype=wp.uint32)
            elif output_name == RenderBufferKind.DEPTH:
                self.outputs.depth_image = self._from_torch(tensor_data, dtype=wp.float32)
            elif output_name == RenderBufferKind.NORMALS:
                self.outputs.normals_image = self._from_torch(tensor_data, dtype=wp.vec3f)
            elif output_name == RenderBufferKind.INSTANCE_SEGMENTATION_FAST:
                self.outputs.instance_segmentation_image = self._from_torch(tensor_data, dtype=wp.uint32)
            elif output_name == RenderBufferKind.RGB:
                pass
            else:
                logger.warning(f"NewtonWarpRenderer - output type {output_name} is not yet supported")

    def get_output(self, output_name: str) -> wp.array:
        if output_name == RenderBufferKind.RGBA:
            return self.outputs.color_image
        elif output_name == RenderBufferKind.ALBEDO:
            return self.outputs.albedo_image
        elif output_name == RenderBufferKind.DEPTH:
            return self.outputs.depth_image
        elif output_name == RenderBufferKind.NORMALS:
            return self.outputs.normals_image
        elif output_name == RenderBufferKind.INSTANCE_SEGMENTATION_FAST:
            return self.outputs.instance_segmentation_image
        return None

    def update(self, positions: torch.Tensor, orientations: torch.Tensor, intrinsics: torch.Tensor):
        converted_orientations = convert_camera_frame_orientation_convention(
            orientations, origin="world", target="opengl"
        )

        self.camera_transforms = wp.empty(
            (1, self.newton_sensor.model.world_count), dtype=wp.transformf, device=self.newton_sensor.model.device
        )
        wp.launch(
            RenderData._update_transforms,
            self.newton_sensor.model.world_count,
            [positions, converted_orientations, self.camera_transforms],
            device=self.newton_sensor.model.device,
        )

        if self.camera_rays is None:
            first_focal_length = intrinsics[:, 1, 1][0:1]
            fov_radians_all = 2.0 * torch.atan(self.height / (2.0 * first_focal_length))

            self.camera_rays = self.newton_sensor.utils.compute_pinhole_camera_rays(
                self.width, self.height, wp.from_torch(fov_radians_all, dtype=wp.float32)
            )

    def _from_torch(self, tensor: torch.Tensor, dtype) -> wp.array:
        proxy_array = wp.from_torch(tensor)
        if tensor.is_contiguous():
            return wp.array(
                ptr=proxy_array.ptr,
                dtype=dtype,
                shape=(self.newton_sensor.model.world_count, self.num_cameras, self.height, self.width),
                device=proxy_array.device,
                copy=False,
            )

        logger.warning("NewtonWarpRenderer - torch output array is non-contiguous")
        return wp.zeros(
            (self.newton_sensor.model.world_count, self.num_cameras, self.height, self.width),
            dtype=dtype,
            device=proxy_array.device,
        )

    @wp.kernel
    def _update_transforms(
        positions: wp.array(dtype=wp.vec3f),
        orientations: wp.array(dtype=wp.quatf),
        output: wp.array(dtype=wp.transformf, ndim=2),
    ):
        tid = wp.tid()
        output[0, tid] = wp.transformf(positions[tid], orientations[tid])


class NewtonWarpRenderer(BaseRenderer):
    """Newton Warp backend for tiled camera rendering."""

    RenderData = RenderData

    def __init__(self, cfg: NewtonWarpRendererCfg):
        from isaaclab.physics.scene_data_requirements import (
            aggregate_requirements,
            requirement_for_renderer_type,
        )

        self.cfg = cfg
        sim = SimulationContext.instance()
        current_req = sim.get_scene_data_requirements()
        renderer_req = requirement_for_renderer_type("newton_warp")
        merged = aggregate_requirements([current_req, renderer_req])
        if merged != current_req:
            sim.update_scene_data_requirements(merged)

        newton_model = self.get_scene_data_provider().get_newton_model()
        if newton_model is None:
            raise RuntimeError(
                "NewtonWarpRenderer requires a Newton model but the scene data provider returned None. "
                "This usually means the Newton model failed to build from the USD stage "
                "(e.g., unsupported PhysX schemas such as tendons). "
                "Check the log for earlier Newton model build errors."
            )

        self.newton_sensor = newton.sensors.SensorTiledCamera(
            newton_model,
            config=newton.sensors.SensorTiledCamera.RenderConfig(
                enable_textures=cfg.enable_textures,
                enable_shadows=cfg.enable_shadows,
                enable_ambient_lighting=cfg.enable_ambient_lighting,
                enable_backface_culling=cfg.enable_backface_culling,
                max_distance=cfg.max_distance,
            ),
        )

        if cfg.create_default_light:
            self.newton_sensor.utils.create_default_light(enable_shadows=cfg.enable_shadows)

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Publish the per-output layout this Newton Warp backend writes.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.supported_output_types`."""
        seg_spec = (
            RenderBufferSpec(4, torch.uint8)
            if self.cfg.colorize_instance_segmentation
            else RenderBufferSpec(1, torch.int32)
        )
        return {
            RenderBufferKind.RGBA: RenderBufferSpec(4, torch.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, torch.uint8),
            RenderBufferKind.ALBEDO: RenderBufferSpec(4, torch.uint8),
            RenderBufferKind.DEPTH: RenderBufferSpec(1, torch.float32),
            RenderBufferKind.NORMALS: RenderBufferSpec(3, torch.float32),
            RenderBufferKind.INSTANCE_SEGMENTATION_FAST: seg_spec,
        }

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        """No-op for Newton Warp - uses Newton scene directly without stage export.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.prepare_stage`."""
        pass

    def create_render_data(self, spec: CameraRenderSpec) -> RenderData:
        """Create render data for the Newton tiled camera.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data`."""
        return RenderData(self.newton_sensor, spec)

    def set_outputs(self, render_data: RenderData, output_data: dict[str, torch.Tensor]):
        """Store output buffers. See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.set_outputs`."""
        render_data.set_outputs(output_data)

    def update_transforms(self):
        """Sync Newton scene state before rendering.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_transforms`."""
        SimulationContext.instance().update_scene_data_provider(True)

    def update_camera(
        self, render_data: RenderData, positions: torch.Tensor, orientations: torch.Tensor, intrinsics: torch.Tensor
    ):
        """Update camera poses and intrinsics.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.update_camera`."""
        render_data.update(positions, orientations, intrinsics)

    def render(self, render_data: RenderData):
        """Render and write to output buffers. See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.render`."""
        self.newton_sensor.update(
            self.get_scene_data_provider().get_newton_state(),
            render_data.camera_transforms,
            render_data.camera_rays,
            color_image=render_data.outputs.color_image,
            albedo_image=render_data.outputs.albedo_image,
            depth_image=render_data.outputs.depth_image,
            normal_image=render_data.outputs.normals_image,
            shape_index_image=render_data.outputs.instance_segmentation_image,
            # ARGB 93% gray to improve visibility of dark objects and align with RTX renderer background
            clear_data=newton.sensors.SensorTiledCamera.ClearData(clear_color=0xFFEEEEEE),
        )

    def read_output(self, render_data: RenderData, camera_data: CameraData) -> None:
        """Copy rendered outputs to the camera data buffers.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.read_output`."""
        for output_name in camera_data.output:
            if output_name == "rgb":
                continue
            image_data = render_data.get_output(output_name)
            if image_data is not None:
                output_data = camera_data.output[output_name]
                if image_data.ptr != output_data.data_ptr():
                    wp.copy(wp.from_torch(output_data), image_data)

    def cleanup(self, render_data: RenderData | None):
        """Release resources. No-op for Newton Warp.
        See :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.cleanup`."""
        pass

    def get_scene_data_provider(self) -> BaseSceneDataProvider:
        return SimulationContext.instance().initialize_scene_data_provider()
