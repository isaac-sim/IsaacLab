# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import dataclass, field

import newton
import warp as wp

from pxr import Usd

import isaaclab.sim as isaaclab_sim
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import TiledCamera


class CameraManager:
    @dataclass
    class CameraData:
        camera_rays: wp.array(dtype=wp.vec3f, ndim=4) = None
        color_image: wp.array(dtype=wp.uint32, ndim=4) = None
        prims: list[Usd.Prim] = field(default_factory=lambda: [])
        width: int = 100
        height: int = 100

    def __init__(self, scene: InteractiveScene):
        self.render_context: newton.sensors.SensorTiledCamera.RenderContext | None = None
        self.scene = scene
        self.num_cameras = 1
        self.camera_data: dict[str, CameraManager.CameraData] = {}

        for name, sensor in self.scene.sensors.items():
            camera_data = CameraManager.CameraData()
            camera_data.prims = isaaclab_sim.find_matching_prims(sensor.cfg.prim_path)
            if isinstance(sensor, TiledCamera):
                camera_data.width = sensor.cfg.width
                camera_data.height = sensor.cfg.height
            self.camera_data[name] = camera_data

    def create_outputs(self, render_context: newton.sensors.SensorTiledCamera.RenderContext):
        self.render_context = render_context
        for name in self.scene.sensors:
            camera_fovs = wp.array([20.0] * self.num_cameras, dtype=wp.float32)
            width = self.camera_data[name].width
            height = self.camera_data[name].height
            self.camera_data[name].camera_rays = render_context.utils.compute_pinhole_camera_rays(
                width, height, camera_fovs
            )
            self.camera_data[name].color_image = render_context.create_color_image_output(
                width, height, self.num_cameras
            )

    def get_camera_transforms(self, camera_data: CameraData) -> wp.array(dtype=wp.transformf):
        camera_transforms = []
        for prim in camera_data.prims:
            camera_transforms.append(self.__resolve_camera_transform(prim))
        return wp.array([camera_transforms], dtype=wp.transformf)

    def save_images(self, filename: str):
        if self.render_context is None:
            return

        for name, camera_data in self.camera_data.items():
            color_data = self.render_context.utils.flatten_color_image_to_rgba(camera_data.color_image)

            from PIL import Image

            os.makedirs(os.path.dirname(filename), exist_ok=True)
            Image.fromarray(color_data.numpy()).save(filename % name)

    def __resolve_camera_transform(self, prim: Usd.Prim) -> wp.transformf:
        position, orientation = isaaclab_sim.resolve_prim_pose(prim)
        return wp.transformf(position, wp.quatf(orientation[1], -orientation[2], -orientation[3], orientation[0]))
        # position, orientation = isaaclab_sim.resolve_prim_pose(prim)
        # t = torch.tensor(orientation, dtype=torch.float32, device="cpu").unsqueeze(0)
        # t = convert_camera_frame_orientation_convention(t)
        # orientation = t.squeeze(0).cpu().numpy()
        # return wp.transformf(position, wp.quatf(orientation))


class NewtonWarpRenderer:
    def __init__(self, scene: InteractiveScene):
        self.scene = scene

        builder = newton.ModelBuilder()
        builder.add_usd(isaaclab_sim.get_current_stage(), ignore_paths=[r"/World/envs/.*"])
        for world_id in range(scene.num_envs):
            builder.begin_world()
            for name, articulation in self.scene.articulations.items():
                path = articulation.cfg.prim_path.replace(".*", str(world_id))
                builder.add_usd(isaaclab_sim.get_current_stage(), root_path=path)
            builder.end_world()

        self.newton_model = builder.finalize()
        self.newton_state = self.newton_model.state()

        self.physx_to_newton_body_mapping: dict[str, wp.array(dtype=wp.int32, ndim=2)] = {}

        self.camera_manager = CameraManager(scene)
        self.newton_sensor = newton.sensors.SensorTiledCamera(self.newton_model)
        self.camera_manager.create_outputs(self.newton_sensor.render_context)

    def update(self):
        self.__update_mapping()
        for name, articulation in self.scene.articulations.items():
            if mapping := self.physx_to_newton_body_mapping.get(name):
                physx_pos = wp.from_torch(articulation.data.body_pos_w)
                physx_quat = wp.from_torch(articulation.data.body_quat_w)
                inputs = [mapping, self.newton_model.body_world, physx_pos, physx_quat, self.newton_state.body_q]
                wp.launch(
                    NewtonWarpRenderer.__update_transforms,
                    mapping.shape,
                    inputs,
                )

    def render(self, sensor_name: str):
        if camera_data := self.camera_manager.camera_data.get(sensor_name):
            self.__render(camera_data)

    def render_all(self):
        for name, camera_data in self.camera_manager.camera_data.items():
            self.__render(camera_data)

    def __render(self, camera_data: CameraManager.CameraData):
        camera_transforms = self.camera_manager.get_camera_transforms(camera_data)
        self.newton_sensor.render(
            self.newton_state,
            camera_transforms,
            camera_data.camera_rays,
            camera_data.color_image,
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
            self.physx_to_newton_body_mapping[name] = wp.array(
                articulation_mapping,
                dtype=wp.int32,
            )

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
        quat = wp.quatf(quat_raw[1], quat_raw[2], quat_raw[3], quat_raw[0])

        out_transform[newton_body_index] = wp.transformf(pos, quat)
