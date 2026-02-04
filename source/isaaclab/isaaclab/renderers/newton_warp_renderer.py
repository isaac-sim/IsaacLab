from isaaclab.utils.math import convert_camera_frame_orientation_convention
from isaaclab.scene import InteractiveScene
import isaaclab.sim as isaaclab_sim

from dataclasses import dataclass, field
from pxr import Usd, UsdGeom

from newton.sensors import TiledCameraSensor
import numpy as np
import warp as wp
import usdrt
import re
import os
import math


@wp.kernel(enable_backward=False)
def compute_triangle_count(num_face_counts: wp.int32, face_counts: wp.array(dtype=wp.int32), out_face_offsets: wp.array(dtype=wp.int32)):
    offset = wp.int32(0)
    for i in range(num_face_counts):
        out_face_offsets[i] = offset
        offset += face_counts[i]
    out_face_offsets[num_face_counts] = offset - num_face_counts * 2


@wp.kernel(enable_backward=False)
def triangulate_faces(num_face_counts: wp.int32, face_counts: wp.array(dtype=wp.int32), face_indices: wp.array(dtype=wp.int32), face_offsets: wp.array(dtype=wp.int32), out_triangles: wp.array(dtype=wp.int32)):
    offset = face_offsets[wp.tid()]
    num_triangles = face_counts[wp.tid()] - 2
    tri = wp.atomic_add(face_offsets, num_face_counts + 1, num_triangles * 3)

    for i in range(num_triangles):
        out_triangles[tri + i * 3 + 0] = face_indices[offset]
        out_triangles[tri + i * 3 + 1] = face_indices[offset + i + 1]
        out_triangles[tri + i * 3 + 2] = face_indices[offset + i + 2]


@wp.kernel(enable_backward=False)
def update_transforms(fabric_matrices: wp.fabricarray(dtype=wp.mat44d), mapping: wp.array(dtype=wp.int32), transforms: wp.array(dtype=wp.transformf)):
    tid = wp.tid()
    if mapping[tid] > -1:
        m = fabric_matrices[mapping[tid]]

        orientation = wp.mat33f(wp.float32(m[0, 0]), wp.float32(m[1, 0]), wp.float32(m[2, 0]),
                                wp.float32(m[0, 1]), wp.float32(m[1, 1]), wp.float32(m[2, 1]),
                                wp.float32(m[0, 2]), wp.float32(m[1, 2]), wp.float32(m[2, 2]))

        position = wp.vec3f(wp.float32(m[3, 0]), wp.float32(m[3, 1]), wp.float32(m[3, 2]))

        transforms[tid] = wp.transformf(position, wp.normalize(wp.quat_from_matrix(orientation)))


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


class NewtonWarpRenderer:
    @dataclass
    class PrimData:
        is_shared: bool = False
        shape_type: TiledCameraSensor.RenderShapeType = TiledCameraSensor.RenderShapeType.NONE
        master_prim: Usd.Prim | None = None
        prims: list[tuple[int, Usd.Prim]] = field(default_factory=lambda: [])

    @dataclass
    class CameraData:
        is_shared: bool = False
        master_prim: Usd.Prim | None = None
        prims: list[tuple[int, Usd.Prim]] = field(default_factory=lambda: [])

    def __init__(self, scene: InteractiveScene, width: int, height: int):
        self.num_worlds = 0
        self.num_cameras = 1
        self.camera_data: dict[str, NewtonWarpRenderer.CameraData] = {}
        self.prim_data: dict[str, NewtonWarpRenderer.PrimData] = {}
        self.__collect_prims(isaaclab_sim.get_current_stage())

        shape_transforms = []
        shape_sizes = []
        shape_types = []
        shape_mesh_indices = []
        shape_world_indices = []

        self.__warp_meshes = []
        self.__fabric_stage_id = None
        self.__fabric_selection: usdrt.RtPrimSelection = None
        self.__fabric_selection_mapping: list[int] = []

        for prim_path, prim_data in self.prim_data.items():
            if prim_data.is_shared:
                mesh_index = -1
                if prim_data.shape_type == TiledCameraSensor.RenderShapeType.MESH:
                    mesh_index = len(self.__warp_meshes)
                    self.__warp_meshes.append(self.__build_mesh(prim_data.master_prim))

                for world_id, prim in prim_data.prims:
                    shape_types.append(prim_data.shape_type)
                    shape_transforms.append(self.__resolve_transform(prim))
                    shape_sizes.append(self.__resolve_shape_size(prim_data.shape_type, prim))
                    shape_mesh_indices.append(mesh_index)
                    shape_world_indices.append(world_id)

        self.render_context = TiledCameraSensor.RenderContext(width, height, self.num_worlds, self.num_cameras)
        self.render_context.num_shapes = len(shape_transforms)
        self.render_context.num_shapes_enabled = self.render_context.num_shapes_total
        self.render_context.mesh_ids = wp.array([mesh.id for mesh in self.__warp_meshes], dtype=wp.uint64)
        self.render_context.shape_mesh_indices = wp.array(shape_mesh_indices, dtype=wp.int32)
        self.render_context.shape_types = wp.array(shape_types, dtype=wp.int32)
        self.render_context.mesh_bounds = wp.empty((len(self.__warp_meshes), 2), dtype=wp.vec3f, ndim=2)
        self.render_context.shape_enabled = wp.array(np.arange(self.render_context.num_shapes_total), dtype=wp.uint32)
        self.render_context.shape_sizes = wp.array(shape_sizes, dtype=wp.vec3f)
        self.render_context.shape_transforms = wp.array(shape_transforms, dtype=wp.transformf)
        self.render_context.shape_materials = wp.array(np.full(self.render_context.num_shapes_total, fill_value=-1, dtype=np.int32), dtype=wp.int32)
        self.render_context.shape_world_index = wp.array(shape_world_indices, dtype=wp.int32)
        if len(self.__warp_meshes) > 0:
            wp.launch(
                kernel=compute_mesh_bounds,
                dim=len(self.__warp_meshes),
                inputs=[self.render_context.mesh_ids, self.render_context.mesh_bounds],
            )

        self.__create_default_light(False)
        self.__assign_random_colors_per_shape()

        self.camera_fovs = wp.array([20.0] * self.render_context.num_cameras, dtype=wp.float32)
        self.camera_rays = self.__compute_pinhole_camera_rays(self.camera_fovs)
        self.color_image = self.render_context.create_color_image_output()

    def __update_fabric_selection(self):
        stage_id = isaaclab_sim.get_current_stage_id()
        if self.__fabric_stage_id != stage_id:
            self.__fabric_stage_id = stage_id
            self.__fabric_selection = None

        stage = usdrt.Usd.Stage.Attach(stage_id)

        update_mapping = False
        if self.__fabric_selection is None:
            self.__fabric_selection = stage.SelectPrims(require_attrs=[(usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.Read)], want_paths=True, device=wp.get_device().alias)
            update_mapping = True
        else:
            update_mapping = self.__fabric_selection.PrepareForReuse()

        if update_mapping:
            self.__fabric_selection_mapping = []
            selection_paths = self.__fabric_selection.GetPaths()
            for prim_path, prim_data in self.prim_data.items():
                if prim_data.is_shared:
                    for world_id, prim in prim_data.prims:
                        try:
                            self.__fabric_selection_mapping.append(selection_paths.index(prim_path % world_id))
                        except ValueError:
                            self.__fabric_selection_mapping.append(-1)

    def update(self):
        self.__update_fabric_selection()
        fabric_matrices = wp.fabricarray(self.__fabric_selection.__fabric_arrays_interface__, "omni:fabric:worldMatrix", dtype=wp.mat44d)
        wp.launch(update_transforms, len(self.__fabric_selection_mapping), [fabric_matrices, wp.array(self.__fabric_selection_mapping, dtype=wp.int32), self.render_context.shape_transforms])

    def render(self):
        self.render_context.render(self.__get_camera_transforms(), self.camera_rays, self.color_image)

    def save_image(self, filename: str):
        color_data = self.__flatten_color_image_to_rgba(self.color_image)
        
        from PIL import Image
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        Image.fromarray(color_data.numpy()).save(filename)

    def __collect_prims(self, stage: Usd.Stage, env_regex: str = r"(?P<root>/World/envs/env_)(?P<id>\d+)(?P<path>/.*)"):
        env_pattern = re.compile(env_regex)

        num_worlds = -1
        stage_prims: list[Usd.Prim] = [stage.GetPseudoRoot()]
        while stage_prims:
            prim = stage_prims.pop(0)
            prim_path = prim.GetPath().pathString

            world_id = -1
            if match := env_pattern.match(prim_path):
                world_id = int(match.group("id"))
                prim_path = match.group("root") + "%d" + match.group("path")

            if world_id > -1:
                if world_id > num_worlds:
                    num_worlds = world_id

            imageable = UsdGeom.Imageable(prim)
            if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue

            shape_type = TiledCameraSensor.RenderShapeType.NONE
            
            if prim.IsA(UsdGeom.Mesh):
                shape_type = TiledCameraSensor.RenderShapeType.MESH
            elif prim.IsA(UsdGeom.Sphere):
                shape_type = TiledCameraSensor.RenderShapeType.SPHERE
            elif prim.IsA(UsdGeom.Capsule):
                shape_type = TiledCameraSensor.RenderShapeType.CAPSULE
            elif prim.IsA(UsdGeom.Cube):
                shape_type = TiledCameraSensor.RenderShapeType.BOX
            elif prim.IsA(UsdGeom.Cylinder):
                shape_type = TiledCameraSensor.RenderShapeType.CYLINDER
            elif prim.IsA(UsdGeom.Cone):
                shape_type = TiledCameraSensor.RenderShapeType.CONE
            elif prim.IsA(UsdGeom.Plane):
                shape_type = TiledCameraSensor.RenderShapeType.PLANE
            elif prim.IsA(UsdGeom.Camera):
                if not prim_path in self.camera_data:
                    self.camera_data[prim_path] = NewtonWarpRenderer.CameraData(world_id > -1, prim)
                self.camera_data[prim_path].prims.append((world_id, prim))

            if shape_type != TiledCameraSensor.RenderShapeType.NONE:
                if not prim_path in self.prim_data:
                    self.prim_data[prim_path] = NewtonWarpRenderer.PrimData(world_id > -1, shape_type, prim)
                self.prim_data[prim_path].prims.append((world_id, prim))

            if child_prims := prim.GetFilteredChildren(Usd.TraverseInstanceProxies()):
                stage_prims.extend(child_prims)

        self.num_cameras = 0
        for prim_path, camera_data in self.camera_data.items():
            if camera_data.is_shared:
                self.num_cameras += 1
        self.num_worlds = num_worlds + 1

    def __build_mesh(self, prim: Usd.Prim):
        mesh = UsdGeom.Mesh(prim)
        points = wp.array(mesh.GetPointsAttr().Get(), dtype=wp.vec3f)
        face_vertex_counts = wp.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=wp.int32)
        face_vertex_indices = wp.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=wp.int32)

        num_face_counts = face_vertex_counts.size
        wp_face_offsets = wp.zeros(num_face_counts + 2, dtype=wp.int32)

        wp.launch(kernel=compute_triangle_count, dim=1, inputs=[num_face_counts, face_vertex_counts, wp_face_offsets])

        num_triangles = wp_face_offsets.numpy()[num_face_counts].item()
        wp_triangle_indices = wp.empty(num_triangles * 3, dtype=wp.int32)

        wp.launch(kernel=triangulate_faces, dim=num_face_counts, inputs=[num_face_counts, face_vertex_counts, face_vertex_indices, wp_face_offsets, wp_triangle_indices])

        return wp.Mesh(points=points, velocities=None, indices=wp_triangle_indices)

    def __resolve_transform(self, prim: Usd.Prim) -> wp.transformf:
        position, orientation = isaaclab_sim.resolve_prim_pose(prim)
        return wp.transformf(position, wp.quatf(orientation[1], orientation[2], orientation[3], orientation[0]))

    def __resolve_scale(self, prim: Usd.Prim) -> wp.vec3f:
        scale = isaaclab_sim.resolve_prim_scale(prim)
        return wp.vec3f(scale)

    def __resolve_camera_transform(self, prim: Usd.Prim) -> wp.transformf:
        position, orientation = isaaclab_sim.resolve_prim_pose(prim)
        return wp.transformf(position, wp.quatf(orientation[1], -orientation[2], -orientation[3], orientation[0]))
        # position, orientation = isaaclab_sim.resolve_prim_pose(prim)
        # t = torch.tensor(orientation, dtype=torch.float32, device="cpu").unsqueeze(0)
        # t = convert_camera_frame_orientation_convention(t)
        # orientation = t.squeeze(0).cpu().numpy()
        # return wp.transformf(position, wp.quatf(orientation))

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

    def __resolve_shape_size(self, shape_type: TiledCameraSensor.RenderShapeType, prim: Usd.Prim) -> wp.vec3f:
        if shape_type == TiledCameraSensor.RenderShapeType.SPHERE:
            return self.__resolve_shape_size_sphere(prim)
        if shape_type == TiledCameraSensor.RenderShapeType.CAPSULE:
            return self.__resolve_shape_size_capsule(prim)
        if shape_type == TiledCameraSensor.RenderShapeType.BOX:
            return self.__resolve_shape_size_box(prim)
        if shape_type == TiledCameraSensor.RenderShapeType.CYLINDER:
            return self.__resolve_shape_size_cylinder(prim)
        if shape_type == TiledCameraSensor.RenderShapeType.CONE:
            return self.__resolve_shape_size_cone(prim)
        if shape_type == TiledCameraSensor.RenderShapeType.PLANE:
            return self.__resolve_shape_size_plane(prim)
        return self.__resolve_scale(prim)

    def __resolve_shape_size_sphere(self, prim: Usd.Prim) -> wp.vec3f:
        sphere = UsdGeom.Sphere(prim)
        radius = sphere.GetRadiusAttr().Get()
        return wp.vec3f(radius, 0.0, 0.0)

    def __resolve_shape_size_capsule(self, prim: Usd.Prim) -> wp.vec3f:
        capsule = UsdGeom.Capsule(prim)
        radius = capsule.GetRadiusAttr().Get()
        height = capsule.GetHeightAttr().Get()
        return wp.vec3f(radius, height, 0.0)

    def __resolve_shape_size_box(self, prim: Usd.Prim) -> wp.vec3f:
        cube = UsdGeom.Cube(prim)
        size = cube.GetSizeAttr().Get()
        scale = self.__resolve_scale(prim)
        return wp.vec3f(size * scale[0], size * scale[1], size * scale[2])

    def __resolve_shape_size_cylinder(self, prim: Usd.Prim) -> wp.vec3f:
        cylinder = UsdGeom.Cylinder(prim)
        radius = cylinder.GetRadiusAttr().Get()
        height = cylinder.GetHeightAttr().Get()
        return wp.vec3f(radius, height, 0.0)

    def __resolve_shape_size_cone(self, prim: Usd.Prim) -> wp.vec3f:
        cone = UsdGeom.Cone(prim)
        radius = cone.GetRadiusAttr().Get()
        height = cone.GetHeightAttr().Get()
        return wp.vec3f(radius, height, 0.0)

    def __resolve_shape_size_plane(self, prim: Usd.Prim) -> wp.vec3f:
        return wp.vec3f(0.0)

    def __get_camera_transforms(self) -> wp.array(dtype=wp.transformf):
        camera_transforms = [[] for i in range(self.num_worlds)]
        for prim_path, camera_data in self.camera_data.items():
            if camera_data.is_shared:
                for world_id, camera_prim in camera_data.prims:
                    camera_transforms[world_id].append(self.__resolve_camera_transform(camera_prim))

        return wp.array(camera_transforms, dtype=wp.transformf)