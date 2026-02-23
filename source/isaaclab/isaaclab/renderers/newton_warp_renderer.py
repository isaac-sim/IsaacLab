# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import newton
import torch
import warp as wp

from isaaclab.sim import SimulationContext
from isaaclab.utils.math import convert_camera_frame_orientation_convention

from ..visualizers import VisualizerCfg

if TYPE_CHECKING:
    from isaaclab.sensors import SensorBase

    from ..sim.scene_data_providers import SceneDataProvider


logger = logging.getLogger(__name__)


class RenderData:
    class OutputNames:
        RGB = "rgb"
        RGBA = "rgba"
        ALBEDO = "albedo"
        DEPTH = "depth"
        NORMALS = "normals"
        INSTANCE_SEGMENTATION = "instance_segmentation_fast"

    @dataclass
    class CameraOutputs:
        color_image: wp.array(dtype=wp.uint32, ndim=4) = None
        albedo_image: wp.array(dtype=wp.uint32, ndim=4) = None
        depth_image: wp.array(dtype=wp.float32, ndim=4) = None
        normals_image: wp.array(dtype=wp.vec3f, ndim=4) = None
        instance_segmentation_image: wp.array(dtype=wp.uint32, ndim=4) = None

    def __init__(self, render_context: newton.sensors.SensorTiledCamera.RenderContext, sensor: SensorBase):
        self.render_context = render_context
        self.sensor = sensor
        self.num_cameras = 1

        self.camera_rays: wp.array(dtype=wp.vec3f, ndim=4) = None
        self.camera_transforms: wp.array(dtype=wp.transformf, ndim=2) = None
        self.outputs = RenderData.CameraOutputs()
        self.width = getattr(sensor.cfg, "width", 100)
        self.height = getattr(sensor.cfg, "height", 100)

    def set_outputs(self, output_data: dict[str, torch.Tensor]):
        for output_name, tensor_data in output_data.items():
            if output_name == RenderData.OutputNames.RGBA:
                self.outputs.color_image = self._from_torch(tensor_data, dtype=wp.uint32)
            elif output_name == RenderData.OutputNames.ALBEDO:
                self.outputs.albedo_image = self._from_torch(tensor_data, dtype=wp.uint32)
            elif output_name == RenderData.OutputNames.DEPTH:
                self.outputs.depth_image = self._from_torch(tensor_data, dtype=wp.float32)
            elif output_name == RenderData.OutputNames.NORMALS:
                self.outputs.normals_image = self._from_torch(tensor_data, dtype=wp.vec3f)
            elif output_name == RenderData.OutputNames.INSTANCE_SEGMENTATION:
                self.outputs.instance_segmentation_image = self._from_torch(tensor_data, dtype=wp.uint32)
            elif output_name == RenderData.OutputNames.RGB:
                pass
            else:
                logger.warning(f"NewtonWarpRenderer - output type {output_name} is not yet supported")

    def get_output(self, output_name: str) -> wp.array:
        if output_name == RenderData.OutputNames.RGBA:
            return self.outputs.color_image
        elif output_name == RenderData.OutputNames.ALBEDO:
            return self.outputs.albedo_image
        elif output_name == RenderData.OutputNames.DEPTH:
            return self.outputs.depth_image
        elif output_name == RenderData.OutputNames.NORMALS:
            return self.outputs.normals_image
        elif output_name == RenderData.OutputNames.INSTANCE_SEGMENTATION:
            return self.outputs.instance_segmentation_image
        return None

    def update(self, positions: torch.Tensor, orientations: torch.Tensor, intrinsics: torch.Tensor):
        converted_orientations = convert_camera_frame_orientation_convention(
            orientations, origin="world", target="opengl"
        )

        self.camera_transforms = wp.empty((1, self.render_context.world_count), dtype=wp.transformf)
        wp.launch(
            RenderData._update_transforms,
            self.render_context.world_count,
            [positions, converted_orientations, self.camera_transforms],
        )

        if self.render_context is not None:
            first_focal_length = intrinsics[:, 1, 1][0:1]
            fov_radians_all = 2.0 * torch.atan(self.height / (2.0 * first_focal_length))

            self.camera_rays = self.render_context.utils.compute_pinhole_camera_rays(
                self.width, self.height, wp.from_torch(fov_radians_all, dtype=wp.float32)
            )

    def _from_torch(self, tensor: torch.Tensor, dtype) -> wp.array:
        torch_array = wp.from_torch(tensor)
        if tensor.is_contiguous():
            return wp.array(
                ptr=torch_array.ptr,
                dtype=dtype,
                shape=(self.render_context.world_count, self.num_cameras, self.height, self.width),
                device=torch_array.device,
                copy=False,
            )

        logger.warning("NewtonWarpRenderer - torch output array is non-contiguous")
        return wp.zeros(
            (self.render_context.world_count, self.num_cameras, self.height, self.width),
            dtype=dtype,
            device=torch_array.device,
        )

    @wp.kernel
    def _update_transforms(
        positions: wp.array(dtype=wp.vec3f),
        orientations: wp.array(dtype=wp.quatf),
        output: wp.array(dtype=wp.transformf, ndim=2),
    ):
        tid = wp.tid()
        output[0, tid] = wp.transformf(positions[tid], orientations[tid])


class NewtonWarpRenderer:
    RenderData = RenderData

    def __init__(self):
        self.newton_sensor = newton.sensors.SensorTiledCamera(self.get_scene_data_provider().get_newton_model())

    def create_render_data(self, sensor: SensorBase) -> RenderData:
        return RenderData(self.newton_sensor.render_context, sensor)

    def set_outputs(self, render_data: RenderData, output_data: dict[str, torch.Tensor]):
        render_data.set_outputs(output_data)

    def update_transforms(self):
        SimulationContext.instance().update_scene_data_provider(True)

    def update_camera(
        self, render_data: RenderData, positions: torch.Tensor, orientations: torch.Tensor, intrinsics: torch.Tensor
    ):
        render_data.update(positions, orientations, intrinsics)

    def render(self, render_data: RenderData):
        self.newton_sensor.render(
            self.get_scene_data_provider().get_newton_state(),
            render_data.camera_transforms,
            render_data.camera_rays,
            color_image=render_data.outputs.color_image,
            albedo_image=render_data.outputs.albedo_image,
            depth_image=render_data.outputs.depth_image,
            normal_image=render_data.outputs.normals_image,
            shape_index_image=render_data.outputs.instance_segmentation_image,
        )

    def write_output(self, render_data: RenderData, output_name: str, output_data: torch.Tensor):
        image_data = render_data.get_output(output_name)
        if image_data is not None:
            if image_data.ptr != output_data.data_ptr():
                wp.copy(wp.from_torch(output_data), image_data)

    def get_scene_data_provider(self) -> SceneDataProvider:
        return SimulationContext.instance().initialize_scene_data_provider([VisualizerCfg(visualizer_type="newton")])
