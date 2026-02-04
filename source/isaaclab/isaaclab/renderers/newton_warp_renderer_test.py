from isaaclab.scene import InteractiveScene
import isaaclab.sim as isaaclab_sim

from typing import Any
from pxr import Usd

from newton.sensors import TiledCameraSensor
import numpy as np
import warp as wp
import os
import math


@wp.kernel(enable_backward=False)
def compute_triangle_count(
    num_face_counts: wp.int32, face_counts: wp.array(dtype=wp.int32), out_face_offsets: wp.array(dtype=wp.int32)
):
    offset = wp.int32(0)
    for i in range(num_face_counts):
        out_face_offsets[i] = offset
        offset += face_counts[i]
    out_face_offsets[num_face_counts] = offset - num_face_counts * 2


@wp.kernel(enable_backward=False)
def triangulate_faces(
    num_face_counts: wp.int32,
    face_counts: wp.array(dtype=wp.int32),
    face_indices: wp.array(dtype=wp.int32),
    face_offsets: wp.array(dtype=wp.int32),
    out_triangles: wp.array(dtype=wp.int32),
):
    offset = face_offsets[wp.tid()]
    num_triangles = face_counts[wp.tid()] - 2
    tri = wp.atomic_add(face_offsets, num_face_counts + 1, num_triangles * 3)

    for i in range(num_triangles):
        out_triangles[tri + i * 3 + 0] = face_indices[offset]
        out_triangles[tri + i * 3 + 1] = face_indices[offset + i + 1]
        out_triangles[tri + i * 3 + 2] = face_indices[offset + i + 2]


@wp.kernel(enable_backward=False)
def compute_pinhole_camera_rays(
    width: int,
    height: int,
    camera_fovs: wp.array(dtype=wp.float32),
    out_rays: wp.array(dtype=wp.vec3f, ndim=4),
):
    camera_index, py, px = wp.tid()
    aspect_ratio = float(width) / float(height)
    u = (float(px) + 0.5) / float(width) - 0.5
    v = (float(py) + 0.5) / float(height) - 0.5
    h = wp.tan(camera_fovs[camera_index] / 2.0)
    ray_direction_camera_space = wp.vec3f(u * 2.0 * h * aspect_ratio, -v * 2.0 * h, -1.0)
    out_rays[camera_index, py, px, 0] = wp.vec3f(0.0)
    out_rays[camera_index, py, px, 1] = wp.normalize(ray_direction_camera_space)


@wp.kernel(enable_backward=False)
def flatten_color_image(
    color_image: wp.array(dtype=wp.uint32, ndim=3),
    buffer: wp.array(dtype=wp.uint8, ndim=3),
    width: wp.int32,
    height: wp.int32,
    num_cameras: wp.int32,
    num_worlds_per_row: wp.int32,
):
    world_id, camera_id, y, x = wp.tid()

    view_id = world_id * num_cameras + camera_id

    row = view_id // num_worlds_per_row
    col = view_id % num_worlds_per_row

    px = col * width + x
    py = row * height + y
    color = color_image[world_id, camera_id, y * width + x]

    buffer[py, px, 0] = wp.uint8((color >> wp.uint32(0)) & wp.uint32(0xFF))
    buffer[py, px, 1] = wp.uint8((color >> wp.uint32(8)) & wp.uint32(0xFF))
    buffer[py, px, 2] = wp.uint8((color >> wp.uint32(16)) & wp.uint32(0xFF))
    buffer[py, px, 3] = wp.uint8((color >> wp.uint32(24)) & wp.uint32(0xFF))


@wp.kernel(enable_backward=False)
def compute_mesh_bounds(in_meshes: wp.array(dtype=wp.uint64), out_bounds: wp.array2d(dtype=wp.vec3f)):
    tid = wp.tid()

    min_point = wp.vec3(wp.inf)
    max_point = wp.vec3(-wp.inf)

    if in_meshes[tid] != 0:
        mesh = wp.mesh_get(in_meshes[tid])
        for i in range(mesh.points.shape[0]):
            min_point = wp.min(min_point, mesh.points[i])
            max_point = wp.max(max_point, mesh.points[i])

    out_bounds[tid, 0] = min_point
    out_bounds[tid, 1] = max_point


class NewtonWarpRenderer:
    def __init__(self, scene: InteractiveScene, width: int, height: int):
        self._width = width
        self._height = height
        self._scene_data_provider = None
        self._model = None
        self._state = None
        self._metadata: dict[str, Any] = {}
        self._sensor: TiledCameraSensor | None = None
        self._camera_order: list[str] = []
        self._camera_transforms = None
        self._mesh_data = None

        self.num_worlds = 0
        self.num_cameras = 0

        self.render_context = None
        self.camera_fovs = None
        self.camera_rays = None
        self.color_image = None
        self._warp_meshes = []
        self._mesh_index_map = {}

    def initialize(self, scene_data_provider) -> None:
        if self._sensor is not None:
            return
        if scene_data_provider is None:
            raise RuntimeError("NewtonWarpRenderer requires a scene_data_provider.")

        self._scene_data_provider = scene_data_provider
        self._model = self._scene_data_provider.get_newton_model()
        if self._model is None:
            raise RuntimeError("Scene data provider did not return a Newton model.")

        self._metadata = self._scene_data_provider.get_metadata()
        self.num_worlds = getattr(self._model, "num_worlds", 0) or self._metadata.get("num_envs", 0)

        camera_data = self._scene_data_provider.get_camera_data()
        if not camera_data:
            raise RuntimeError("Scene data provider did not return camera data.")

        self._camera_order = list(camera_data.get("shared_paths") or [])
        self.num_cameras = len(self._camera_order)
        if self.num_cameras <= 0:
            raise RuntimeError("No shared cameras found for NewtonWarpRenderer.")

        self._mesh_data = self._scene_data_provider.get_mesh_data()
        if not self._mesh_data:
            raise RuntimeError("Scene data provider did not return mesh data.")

        self.render_context = TiledCameraSensor.RenderContext(
            self._width,
            self._height,
            num_worlds=self.num_worlds,
            num_cameras=self.num_cameras,
        )
        self._build_render_context_from_mesh_data(self._mesh_data)
        self.__create_default_light(False)
        self.__assign_random_colors_per_shape()

        self.camera_fovs = wp.array([20.0] * self.render_context.num_cameras, dtype=wp.float32)
        self.camera_rays = self.__compute_pinhole_camera_rays(self.camera_fovs)
        self.color_image = self.render_context.create_color_image_output()

    def update(self):
        if self._scene_data_provider is None:
            raise RuntimeError("NewtonWarpRenderer.update() called before initialize().")
        self._scene_data_provider.update()
        self._state = self._scene_data_provider.get_newton_state()
        self._camera_transforms = self._scene_data_provider.get_camera_transforms()

    def render(self):
        if self.render_context is None or self._state is None:
            raise RuntimeError("NewtonWarpRenderer.render() called before initialize()/update().")
        self.render_context.render(self.__get_camera_transforms(), self.camera_rays, self.color_image)

    def save_image(self, filename: str):
        color_data = self.__flatten_color_image_to_rgba(self.color_image)

        from PIL import Image

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        Image.fromarray(color_data.numpy()).save(filename)

    def __resolve_camera_transform(self, position: list[float], orientation: list[float]) -> wp.transformf:
        return wp.transformf(
            wp.vec3f(position[0], position[1], position[2]),
            wp.quatf(orientation[1], -orientation[2], -orientation[3], orientation[0]),
        )

    def __resolve_shape_transform(self, position: list[float], orientation: list[float]) -> wp.transformf:
        return wp.transformf(
            wp.vec3f(position[0], position[1], position[2]),
            wp.quatf(orientation[1], orientation[2], orientation[3], orientation[0]),
        )

    def __get_camera_transforms(self) -> wp.array(dtype=wp.transformf):
        if not self._camera_transforms:
            raise RuntimeError("Camera transforms not available from scene data provider.")

        order = self._camera_transforms.get("order") or []
        positions = self._camera_transforms.get("positions") or []
        orientations = self._camera_transforms.get("orientations") or []

        if order != self._camera_order:
            raise RuntimeError("Camera order mismatch between renderer and scene data provider.")

        camera_transforms: list[list[wp.transformf]] = [
            [wp.transformf() for _ in range(self.num_worlds)] for _ in range(self.num_cameras)
        ]
        for cam_index, prim_path in enumerate(order):
            for world_id in range(self.num_worlds):
                pos = positions[cam_index][world_id]
                ori = orientations[cam_index][world_id]
                if pos is None or ori is None:
                    raise RuntimeError(f"Missing camera transforms for '{prim_path}'.")
                camera_transforms[cam_index][world_id] = self.__resolve_camera_transform(pos, ori)

        return wp.array(camera_transforms, dtype=wp.transformf)

    def __create_default_light(self, enable_shadows: bool = True):
        self.render_context.enable_shadows = enable_shadows
        self.render_context.lights_active = wp.array([True], dtype=wp.bool)
        self.render_context.lights_type = wp.array([TiledCameraSensor.RenderLightType.DIRECTIONAL], dtype=wp.int32)
        self.render_context.lights_cast_shadow = wp.array([True], dtype=wp.bool)
        self.render_context.lights_position = wp.array([wp.vec3f(0.0)], dtype=wp.vec3f)
        self.render_context.lights_orientation = wp.array(
            [wp.vec3f(-0.57735026, 0.57735026, -0.57735026)], dtype=wp.vec3f
        )

    def __assign_random_colors_per_shape(self, seed: int = 100):
        colors = np.random.default_rng(seed).random((self.render_context.num_shapes_total, 4)) * 0.5 + 0.5
        colors[:, -1] = 1.0
        self.render_context.shape_colors = wp.array(colors, dtype=wp.vec4f)

    def __compute_pinhole_camera_rays(self, camera_fovs: wp.array(dtype=wp.float32)) -> wp.array(dtype=wp.vec3f, ndim=4):
        camera_rays = wp.empty(
            (self.render_context.num_cameras, self.render_context.height, self.render_context.width, 2), dtype=wp.vec3f
        )
        wp.launch(
            kernel=compute_pinhole_camera_rays,
            dim=(self.render_context.num_cameras, self.render_context.height, self.render_context.width),
            inputs=[self.render_context.width, self.render_context.height, camera_fovs, camera_rays],
        )
        return camera_rays

    def _build_render_context_from_mesh_data(self, mesh_data: dict[str, Any]) -> None:
        mesh_entries = mesh_data.get("meshes") or []
        shape_entries = mesh_data.get("shape_entries") or []

        self.render_context.num_shapes = len(shape_entries)
        self.render_context.mesh_ids = wp.array([mesh.id for mesh in self._build_meshes(mesh_entries)], dtype=wp.uint64)
        self.render_context.shape_mesh_indices = wp.array(
            [entry["mesh_index"] for entry in shape_entries], dtype=wp.int32
        )
        self.render_context.shape_types = wp.array([entry["shape_type"] for entry in shape_entries], dtype=wp.int32)
        self.render_context.mesh_bounds = wp.empty((len(self._warp_meshes), 2), dtype=wp.vec3f, ndim=2)
        self.render_context.shape_enabled = wp.array(np.arange(self.render_context.num_shapes_total), dtype=wp.uint32)
        self.render_context.shape_sizes = wp.array([entry["size"] for entry in shape_entries], dtype=wp.vec3f)
        self.render_context.shape_transforms = wp.array(
            [self.__resolve_shape_transform(entry["position"], entry["orientation"]) for entry in shape_entries],
            dtype=wp.transformf,
        )
        self.render_context.shape_materials = wp.array(
            np.full(self.render_context.num_shapes_total, fill_value=-1, dtype=np.int32), dtype=wp.int32
        )
        self.render_context.shape_world_index = wp.array([entry["world_id"] for entry in shape_entries], dtype=wp.int32)

        if len(self._warp_meshes) > 0:
            wp.launch(
                kernel=compute_mesh_bounds,
                dim=len(self._warp_meshes),
                inputs=[self.render_context.mesh_ids, self.render_context.mesh_bounds],
            )

    def _build_meshes(self, mesh_entries: list[dict[str, Any]]):
        if self._warp_meshes:
            return self._warp_meshes
        for mesh_entry in mesh_entries:
            points = wp.array(mesh_entry["points"], dtype=wp.vec3f)
            face_vertex_counts = wp.array(mesh_entry["face_counts"], dtype=wp.int32)
            face_vertex_indices = wp.array(mesh_entry["face_indices"], dtype=wp.int32)

            num_face_counts = face_vertex_counts.size
            wp_face_offsets = wp.zeros(num_face_counts + 2, dtype=wp.int32)
            wp.launch(kernel=compute_triangle_count, dim=1, inputs=[num_face_counts, face_vertex_counts, wp_face_offsets])

            num_triangles = wp_face_offsets.numpy()[num_face_counts].item()
            wp_triangle_indices = wp.empty(num_triangles * 3, dtype=wp.int32)
            wp.launch(
                kernel=triangulate_faces,
                dim=num_face_counts,
                inputs=[
                    num_face_counts,
                    face_vertex_counts,
                    face_vertex_indices,
                    wp_face_offsets,
                    wp_triangle_indices,
                ],
            )

            self._warp_meshes.append(wp.Mesh(points=points, velocities=None, indices=wp_triangle_indices))
        return self._warp_meshes

    def __flatten_color_image_to_rgba(
        self, image: wp.array(dtype=wp.uint32, ndim=3)
    ) -> wp.array(dtype=wp.uint8, ndim=3):
        num_worlds_and_cameras = self.render_context.num_worlds * self.render_context.num_cameras
        num_worlds_per_row = math.ceil(math.sqrt(num_worlds_and_cameras))
        num_worlds_per_col = math.ceil(num_worlds_and_cameras / num_worlds_per_row)
        out_buffer = wp.empty(
            (num_worlds_per_col * self.render_context.height, num_worlds_per_row * self.render_context.width, 4),
            dtype=wp.uint8,
        )
        wp.launch(
            flatten_color_image,
            (
                self.render_context.num_worlds,
                self.render_context.num_cameras,
                self.render_context.height,
                self.render_context.width,
            ),
            [
                image,
                out_buffer,
                self.render_context.width,
                self.render_context.height,
                self.render_context.num_cameras,
                num_worlds_per_row,
            ],
        )
        return out_buffer

