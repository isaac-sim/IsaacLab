from isaaclab.scene import InteractiveScene
import isaaclab.sim as isaaclab_sim

from dataclasses import dataclass, field
from pxr import Usd, UsdGeom

from newton.sensors import SensorTiledCamera
import numpy as np
import warp as wp
import usdrt
import re

from isaaclab.renderers.newton_warp_renderer import CameraManager

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


class NewtonWarpRendererDirect:
    @dataclass
    class PrimData:
        is_shared: bool = False
        shape_type: SensorTiledCamera.RenderShapeType = SensorTiledCamera.RenderShapeType.NONE
        master_prim: Usd.Prim | None = None
        prims: list[tuple[int, Usd.Prim]] = field(default_factory=lambda: [])

    def __init__(self, scene: InteractiveScene, width: int, height: int):
        self.prim_data: dict[str, NewtonWarpRendererDirect.PrimData] = {}
        self.num_worlds = self.__collect_prims(isaaclab_sim.get_current_stage())

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
                if prim_data.shape_type == SensorTiledCamera.RenderShapeType.MESH:
                    mesh_index = len(self.__warp_meshes)
                    self.__warp_meshes.append(self.__build_mesh(prim_data.master_prim))

                for world_id, prim in prim_data.prims:
                    shape_types.append(prim_data.shape_type)
                    shape_transforms.append(self.__resolve_transform(prim))
                    shape_sizes.append(self.__resolve_shape_size(prim_data.shape_type, prim))
                    shape_mesh_indices.append(mesh_index)
                    shape_world_indices.append(world_id)

        self.camera_manager = CameraManager(scene)

        self.render_context = SensorTiledCamera.RenderContext(width, height, self.num_worlds, self.camera_manager.num_cameras)
        self.render_context.num_shapes_total = len(shape_transforms)
        self.render_context.num_shapes_enabled = self.render_context.num_shapes_total
        self.render_context.mesh_ids = wp.array([mesh.id for mesh in self.__warp_meshes], dtype=wp.uint64)
        self.render_context.shape_mesh_indices = wp.array(shape_mesh_indices, dtype=wp.int32)
        self.render_context.shape_types = wp.array(shape_types, dtype=wp.int32)
        self.render_context.mesh_bounds = wp.empty((self.render_context.num_shapes_total, 2), dtype=wp.vec3f, ndim=2)
        self.render_context.shape_enabled = wp.array(np.arange(self.render_context.num_shapes_total), dtype=wp.uint32)
        self.render_context.shape_sizes = wp.array(shape_sizes, dtype=wp.vec3f)
        self.render_context.shape_transforms = wp.array(shape_transforms, dtype=wp.transformf)
        self.render_context.shape_materials = wp.array(np.full(self.render_context.num_shapes_total, fill_value=-1, dtype=np.int32), dtype=wp.int32)
        self.render_context.shape_world_index = wp.array(shape_world_indices, dtype=wp.int32)
        self.render_context.utils.compute_mesh_bounds()
        self.render_context.utils.create_default_light(False)
        self.render_context.utils.assign_random_colors_per_shape()

        self.camera_manager.create_outputs(self.render_context)

    def update(self):
        self.__update_fabric_selection()
        fabric_matrices = wp.fabricarray(self.__fabric_selection.__fabric_arrays_interface__, "omni:fabric:worldMatrix", dtype=wp.mat44d)
        wp.launch(update_transforms, len(self.__fabric_selection_mapping), [fabric_matrices, wp.array(self.__fabric_selection_mapping, dtype=wp.int32), self.render_context.shape_transforms])

    def render(self, sensor_name: str):
        if camera_data := self.camera_manager.camera_data.get(sensor_name):
            self.__render(camera_data)

    def render_all(self):
        for name, camera_data in self.camera_manager.camera_data.items():
            self.__render(camera_data)

    def __render(self, camera_data: CameraManager.CameraData):
        self.render_context.render(self.camera_manager.get_camera_transforms(camera_data), camera_data.camera_rays, camera_data.color_image)

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

    def __collect_prims(self, stage: Usd.Stage, env_regex: str = r"(?P<root>/World/envs/env_)(?P<id>\d+)(?P<path>/.*)") -> int:
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

            shape_type = SensorTiledCamera.RenderShapeType.NONE
            
            if prim.IsA(UsdGeom.Mesh):
                shape_type = SensorTiledCamera.RenderShapeType.MESH
            elif prim.IsA(UsdGeom.Sphere):
                shape_type = SensorTiledCamera.RenderShapeType.SPHERE
            elif prim.IsA(UsdGeom.Capsule):
                shape_type = SensorTiledCamera.RenderShapeType.CAPSULE
            elif prim.IsA(UsdGeom.Cube):
                shape_type = SensorTiledCamera.RenderShapeType.BOX
            elif prim.IsA(UsdGeom.Cylinder):
                shape_type = SensorTiledCamera.RenderShapeType.CYLINDER
            elif prim.IsA(UsdGeom.Cone):
                shape_type = SensorTiledCamera.RenderShapeType.CONE
            elif prim.IsA(UsdGeom.Plane):
                shape_type = SensorTiledCamera.RenderShapeType.PLANE

            if shape_type != SensorTiledCamera.RenderShapeType.NONE:
                if not prim_path in self.prim_data:
                    self.prim_data[prim_path] = NewtonWarpRendererDirect.PrimData(world_id > -1, shape_type, prim)
                self.prim_data[prim_path].prims.append((world_id, prim))

            if child_prims := prim.GetFilteredChildren(Usd.TraverseInstanceProxies()):
                stage_prims.extend(child_prims)
        return num_worlds + 1

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

    def __resolve_shape_size(self, shape_type: SensorTiledCamera.RenderShapeType, prim: Usd.Prim) -> wp.vec3f:
        if shape_type == SensorTiledCamera.RenderShapeType.SPHERE:
            return self.__resolve_shape_size_sphere(prim)
        if shape_type == SensorTiledCamera.RenderShapeType.CAPSULE:
            return self.__resolve_shape_size_capsule(prim)
        if shape_type == SensorTiledCamera.RenderShapeType.BOX:
            return self.__resolve_shape_size_box(prim)
        if shape_type == SensorTiledCamera.RenderShapeType.CYLINDER:
            return self.__resolve_shape_size_cylinder(prim)
        if shape_type == SensorTiledCamera.RenderShapeType.CONE:
            return self.__resolve_shape_size_cone(prim)
        if shape_type == SensorTiledCamera.RenderShapeType.PLANE:
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
