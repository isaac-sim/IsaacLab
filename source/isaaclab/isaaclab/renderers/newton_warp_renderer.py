# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import newton
import torch
import warp as wp

import isaaclab.sim as isaaclab_sim
import isaaclab.utils.math

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sensors import SensorBase


class CameraManager:
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

    @dataclass
    class CameraData:
        camera_rays: wp.array(dtype=wp.vec3f, ndim=4) = None
        camera_transforms: wp.array(dtype=wp.transformf, ndim=2) = None
        outputs: CameraManager.CameraOutputs = field(default_factory=lambda: CameraManager.CameraOutputs())
        name: str | None = None
        width: int = 100
        height: int = 100

    def __init__(self, scene: InteractiveScene):
        self.render_context: newton.sensors.SensorTiledCamera.RenderContext | None = None
        self.scene = scene
        self.num_cameras = 1
        self.camera_data: dict[SensorBase, CameraManager.CameraData] = {}

        for name, sensor in self.scene.sensors.items():
            camera_data = CameraManager.CameraData()
            camera_data.name = name
            camera_data.width = getattr(sensor.cfg, "width", camera_data.width)
            camera_data.height = getattr(sensor.cfg, "height", camera_data.height)
            self.camera_data[sensor] = camera_data

    def create_outputs(self, render_context: newton.sensors.SensorTiledCamera.RenderContext):
        self.render_context = render_context
        for name, sensor in self.scene.sensors.items():
            if camera_data := self.camera_data.get(sensor):
                if data_types := getattr(sensor.cfg, "data_types"):
                    if CameraManager.OutputNames.RGBA in data_types or CameraManager.OutputNames.RGB in data_types:
                        camera_data.outputs.color_image = render_context.create_color_image_output(
                            camera_data.width, camera_data.height, self.num_cameras
                        )
                    if CameraManager.OutputNames.ALBEDO in data_types:
                        camera_data.outputs.albedo_image = render_context.create_albedo_image_output(
                            camera_data.width, camera_data.height, self.num_cameras
                        )
                    if CameraManager.OutputNames.DEPTH in data_types:
                        camera_data.outputs.depth_image = render_context.create_depth_image_output(
                            camera_data.width, camera_data.height, self.num_cameras
                        )
                    if CameraManager.OutputNames.NORMALS in data_types:
                        camera_data.outputs.normals_image = render_context.create_normal_image_output(
                            camera_data.width, camera_data.height, self.num_cameras
                        )
                    if CameraManager.OutputNames.INSTANCE_SEGMENTATION in data_types:
                        camera_data.outputs.instance_segmentation_image = (
                            render_context.create_shape_index_image_output(
                                camera_data.width, camera_data.height, self.num_cameras
                            )
                        )

    def update(
        self, camera_data: CameraData, positions: torch.Tensor, orientations: torch.Tensor, intrinsics: torch.Tensor
    ):
        converted_orientations = isaaclab.utils.math.convert_camera_frame_orientation_convention(
            orientations, origin="world", target="opengl"
        )

        camera_data.camera_transforms = wp.empty((1, self.scene.num_envs), dtype=wp.transformf)
        wp.launch(
            CameraManager.__update_transforms,
            self.scene.num_envs,
            [positions, converted_orientations, camera_data.camera_transforms],
        )

        if self.render_context is not None:
            first_focal_length = intrinsics[:, 1, 1][0:1]
            fov_radians_all = 2.0 * torch.atan(camera_data.height / (2.0 * first_focal_length))

            camera_data.camera_rays = self.render_context.utils.compute_pinhole_camera_rays(
                camera_data.width, camera_data.height, wp.from_torch(fov_radians_all, dtype=wp.float32)
            )

    @wp.kernel
    def __update_transforms(
        positions: wp.array(dtype=wp.vec3f),
        orientations: wp.array(dtype=wp.quatf),
        output: wp.array(dtype=wp.transformf, ndim=2),
    ):
        tid = wp.tid()
        output[0, tid] = wp.transformf(positions[tid], orientations[tid])


class NewtonWarpRenderer:
    def __init__(self, scene: InteractiveScene):
        assert scene is not None, "NewtonWarpRenderer needs an InteractiveScene to initialize!"

        self.scene = scene

        builder = newton.ModelBuilder()
        builder.add_usd(isaaclab_sim.get_current_stage(), ignore_paths=[r"/World/envs/.*"])
        for world_id in range(self.scene.num_envs):
            builder.begin_world()
            for name, articulation in self.scene.articulations.items():
                path = articulation.cfg.prim_path.replace(".*", str(world_id))
                builder.add_usd(isaaclab_sim.get_current_stage(), root_path=path)
            builder.end_world()

        self.newton_model = builder.finalize()
        self.newton_state = self.newton_model.state()

        self.physx_to_newton_body_mapping: dict[str, wp.array(dtype=wp.int32, ndim=2)] = {}

        self.camera_manager = CameraManager(self.scene)
        self.newton_sensor = newton.sensors.SensorTiledCamera(self.newton_model)
        self.camera_manager.create_outputs(self.newton_sensor.render_context)

    def update_transforms(self):
        self.__update_mapping()
        for name, articulation in self.scene.articulations.items():
            if mapping := self.physx_to_newton_body_mapping.get(name):
                physx_pos = wp.from_torch(articulation.data.body_pos_w)
                physx_quat = wp.from_torch(articulation.data.body_quat_w)
                wp.launch(
                    NewtonWarpRenderer.__update_transforms,
                    mapping.shape,
                    [mapping, self.newton_model.body_world, physx_pos, physx_quat, self.newton_state.body_q],
                )

    def update_camera(
        self, sensor: SensorBase, positions: torch.Tensor, orientations: torch.Tensor, intrinsics: torch.Tensor
    ):
        if camera_data := self.camera_manager.camera_data.get(sensor):
            self.camera_manager.update(camera_data, positions, orientations, intrinsics)

    def render(self, sensor: SensorBase):
        if camera_data := self.camera_manager.camera_data.get(sensor):
            self.__render(camera_data)

    def render_all(self):
        for name, camera_data in self.camera_manager.camera_data.items():
            self.__render(camera_data)

    def convert_output(self, sensor: SensorBase, output_name: str, output_data: torch.Tensor):
        if camera_data := self.camera_manager.camera_data.get(sensor):
            if output_name == CameraManager.OutputNames.RGBA:
                wp.copy(wp.from_torch(output_data), camera_data.outputs.color_image)
            elif output_name == CameraManager.OutputNames.ALBEDO:
                wp.copy(wp.from_torch(output_data), camera_data.outputs.albedo_image)
            elif output_name == CameraManager.OutputNames.DEPTH:
                wp.copy(wp.from_torch(output_data), camera_data.outputs.depth_image)
            elif output_name == CameraManager.OutputNames.NORMALS:
                wp.copy(wp.from_torch(output_data), camera_data.outputs.normals_image)
            elif output_name == CameraManager.OutputNames.INSTANCE_SEGMENTATION:
                wp.copy(wp.from_torch(output_data), camera_data.outputs.instance_segmentation_image)
            else:
                print(f"NewtonWarpRenderer - output type {output_name} is not yet supported")

    def __render(self, camera_data: CameraManager.CameraData):
        self.newton_sensor.render(
            self.newton_state,
            camera_data.camera_transforms,
            camera_data.camera_rays,
            color_image=camera_data.outputs.color_image,
            albedo_image=camera_data.outputs.albedo_image,
            depth_image=camera_data.outputs.depth_image,
            normal_image=camera_data.outputs.normals_image,
            shape_index_image=camera_data.outputs.instance_segmentation_image,
        )

    def __update_mapping(self):
        if self.physx_to_newton_body_mapping:
            return

        self.physx_to_newton_body_mapping.clear()
        for name, articulation in self.scene.articulations.items():
            articulation_mapping = []
            for prim in isaaclab_sim.find_matching_prims(articulation.cfg.prim_path):
                body_indices = []
                for body_name in articulation.body_names:
                    prim_path = prim.GetPath().AppendChild(body_name).pathString
                    body_indices.append(self.newton_model.body_key.index(prim_path))
                articulation_mapping.append(body_indices)
            self.physx_to_newton_body_mapping[name] = wp.array(articulation_mapping, dtype=wp.int32)

    @wp.kernel(enable_backward=False)
    def __update_transforms(
        mapping: wp.array(dtype=wp.int32, ndim=2),
        newton_body_world: wp.array(dtype=wp.int32),
        physx_pos: wp.array(dtype=wp.float32, ndim=3),
        physx_quat: wp.array(dtype=wp.float32, ndim=3),
        out_transform: wp.array(dtype=wp.transformf),
    ):
        physx_world_id, physx_body_id = wp.tid()

        newton_body_index = mapping[physx_world_id, physx_body_id]
        newton_world_id = newton_body_world[newton_body_index]

        pos_raw = physx_pos[newton_world_id, physx_body_id]
        pos = wp.vec3f(pos_raw[0], pos_raw[1], pos_raw[2])

        quat_raw = physx_quat[newton_world_id, physx_body_id]
        quat = wp.quatf(quat_raw[0], quat_raw[1], quat_raw[2], quat_raw[3])

        out_transform[newton_body_index] = wp.transformf(pos, quat)
