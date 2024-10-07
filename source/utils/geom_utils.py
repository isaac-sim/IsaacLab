import copy
import os
import pdb
from typing import Optional, Tuple
import pathlib
import numpy as np
import torch
import trimesh
from einops import rearrange
from kaolin.ops.mesh import check_sign
# import igl
from scipy.spatial.transform import Rotation
import open3d as o3d

import pytorch3d
import pytorch_volumetric as pv
from configs.user_config import BASE_PATH
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points
from pytorch3d.ops.point_mesh_closest_point import point_mesh_closest_point
from utils.data_utils import readOBJ
from visualization.plot import plot_mesh, plot_pointclouds, plot_seg_fig
from pytorch3d.structures import Meshes, join_meshes_as_batch


def o3d_mesh_to_pytorch3d(mesh: o3d.geometry.TriangleMesh, device="cpu") -> Meshes:
    """
    Input:
    mesh: o3d.geometry.TriangleMesh
    """
    vertices = torch.tensor(np.array(mesh.vertices), dtype=torch.float32)
    faces = torch.tensor(np.array(mesh.triangles), dtype=torch.long)
    mesh = Meshes(verts=[vertices], faces=[faces])
    mesh = mesh.to(device)
    return mesh


def is_collinear(pts):
    """3x3"""
    pts = pts[:, :2]
    area = pts[0, 0] * (pts[1, 1] - pts[2, 1]) + pts[1, 0] * (pts[2, 1] - pts[0, 1]) \
           + pts[2, 0] * (pts[0, 1] - pts[1, 1])
    return np.abs(area) < 1e-8


def rotate_around_axis(pts: np.ndarray, axis: np.ndarray, angle: float):
    """
    pts: np.array[N, 3] float
    axis: np.array[2, 3] float
    angle: float
    """
    u = axis[0] - axis[1]
    u = u / np.linalg.norm(u)
    r = Rotation.from_rotvec(u * angle)
    return r.apply(pts - axis[1]) + axis[1]


def barycentric_interpolation(query_coords: np.array, verts: np.array, faces: np.array) -> np.array:
    """
    Input:
    query_coords: np.array[M, 3] float barycentric coorindates
    verts: np.array[N, 3] float vertecies
    faces: np.array[M, 3] int face index into verts, 1:1 coorespondace to query_coords

    Output
    result: np.array[M, 3] float interpolated points
    """
    assert (len(verts.shape) == 2)
    result = np.zeros((len(query_coords), verts.shape[1]), dtype=verts.dtype)
    for c in range(verts.shape[1]):
        for i in range(query_coords.shape[1]):
            result[:, c] += query_coords[:, i] * verts[:, c][faces[:, i]]
    return result


def barycentric_interpolation_torch1(query_coords: torch.Tensor, verts: torch.Tensor,
                                     faces: torch.Tensor) -> torch.Tensor:
    """
    Input:
    query_coords: torch.Tensor[M, 3] float barycentric coorindates
    verts: torch.Tensor[N, K] float vertecies
    faces: torch.Tensor[M, 3] int face index into verts, 1:1 coorespondace to query_coords

    Output
    result: torch.Tensor[M, 3] float interpolated points
    """
    is_batch = len(verts.shape) == 3
    if not is_batch:
        result = torch.zeros((len(query_coords), verts.shape[1]), dtype=verts.dtype, device=verts.device)
        # rewrite the following for loop using einsum
        for c in range(verts.shape[1]):
            for i in range(query_coords.shape[1]):
                result[:, c] += query_coords[:, i] * verts[:, c][faces[:, i]]
        return result
    else:
        # verts [B, N, 3]
        result = torch.zeros((verts.shape[0], len(query_coords), verts.shape[-1]), device=verts.device)
        for c in range(verts.shape[-1]):
            for i in range(query_coords.shape[1]):
                result[:, :, c] += query_coords[:, i] * verts[:, :, c][:, faces[:, i]]
        return result


def barycentric_interpolation_torch2(query_coords: torch.Tensor, verts: torch.Tensor,
                                     faces: torch.Tensor) -> torch.Tensor:
    """
    Input:
    query_coords: torch.Tensor[M, 3] float barycentric coorindates
    verts: torch.Tensor[N, 3] float vertecies
    faces: torch.Tensor[M, 3] int face index into verts, 1:1 coorespondace to query_coords

    Output
    result: torch.Tensor[M, 3] float interpolated points
    """
    assert (len(verts.shape) == 2)
    result = torch.zeros((len(query_coords), verts.shape[1]), dtype=verts.dtype, device=verts.device)

    # Expand dimensions for broadcasting
    expanded_verts = verts.unsqueeze(1).expand(-1, query_coords.shape[1], -1)
    expanded_faces = faces.unsqueeze(0).expand(verts.shape[0], -1, -1)

    # Use advanced indexing to gather the vertices
    gathered_verts = expanded_verts.gather(2, expanded_faces)

    # Multiply and sum along the appropriate dimension
    result = (query_coords.unsqueeze(2) * gathered_verts).sum(dim=1)
    return result


def barycentric_interpolation_torch3(query_coords: torch.Tensor, verts: torch.Tensor,
                                     faces: torch.Tensor) -> torch.Tensor:
    """
    Input:
    query_coords: torch.Tensor[M, 3] float barycentric coorindates
    verts: torch.Tensor[N, 3] float vertecies
    faces: torch.Tensor[M, 3] int face index into verts, 1:1 coorespondace to query_coords

    Output
    result: torch.Tensor[M, 3] float interpolated points
    """
    assert (len(verts.shape) == 2)

    # Use einsum to perform the multiplication and summation
    result = torch.einsum('bi,ci,ci->bc', query_coords, verts[:, faces], verts)
    return result


def mesh_sample_barycentric_torch(
        verts: torch.Tensor, faces: torch.Tensor, face_areas: torch.Tensor,
        num_samples: int, seed: Optional[int] = 1234,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniformly sample points (as their barycentric coordinate) on suface
    """
    face_areas = face_areas / torch.sum(face_areas)
    assert (len(face_areas) == len(faces))

    rs = torch.manual_seed(seed)
    # select faces
    selected_face_idx = torch.multinomial(
        face_areas, num_samples, replacement=True).long()
    selected_faces = faces[selected_face_idx]
    barycentric_uv = torch.rand(num_samples, 2, device=verts.device)
    not_triangle = (torch.sum(barycentric_uv, dim=1) >= 1)
    barycentric_uv[not_triangle] = 1 - barycentric_uv[not_triangle]

    barycentric_all = torch.zeros((num_samples, 3), dtype=barycentric_uv.dtype, device=verts.device)
    barycentric_all[:, :2] = barycentric_uv
    barycentric_all[:, 2] = 1 - torch.sum(barycentric_uv, dim=1)

    return barycentric_all, selected_face_idx


base_path = pathlib.Path(BASE_PATH)
scene_path = str(base_path.joinpath('configs', 'scene_mesh2.obj'))
v, f, _, _ = readOBJ(scene_path)
FILE_DIR = pathlib.Path(__file__).parent.parent.joinpath('configs/box_template.obj')
mesh_template = o3d.io.read_triangle_mesh(str(FILE_DIR))

ROPE_SCENE_RANGE = torch.tensor([-1, -1., -0.2, 1., 1, 1]).view(2, 3)
HINGE_SCENE_RANGE = torch.tensor([-0.5, -1., -0.5, 1.5, 1, 1.5]).view(2, 3)


def create_box(size, trans):
    vs = np.array([
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, -1, -1],
        [1, 1, -1, ],
        [1, 1, 1]
    ])
    vs = vs * size.reshape(1, 3) + trans.reshape(1, 3)
    vs[:, 2] = np.maximum(vs[:, 2], 0)
    return vs


def create_box_mesh(scale, trans):
    mesh = copy.deepcopy(mesh_template)
    # mesh2 = o3d.geometry.TriangleMesh.create_box(2,2,2).translate(np.array([-1,-1,-1]))
    v = np.array(mesh.vertices)
    v = v * np.array(scale).reshape(1, 3) + np.array(trans).reshape(1, 3)
    mesh.vertices = o3d.utility.Vector3dVector(v)
    return mesh


def create_table_scene_mesh(table_cfg):
    t_mesh = create_box_mesh(
        scale=(0.4, 0.4, table_cfg[1] / 2),
        trans=(table_cfg[0], 0, table_cfg[1] / 2))
    t_mesh = trimesh.Trimesh(vertices=t_mesh.vertices, faces=t_mesh.triangles)
    f_mesh = create_box_mesh(
        scale=(1, 1, 1),
        trans=(0.5, 0, -1)
    )
    f_mesh = trimesh.Trimesh(vertices=f_mesh.vertices, faces=f_mesh.triangles)

    obs_cfgs = table_cfg[2:].reshape(-1, 6)
    num_obs = len(obs_cfgs)
    obs_list = []
    for i in range(num_obs):
        obs_mesh = create_box_mesh(
            trans=obs_cfgs[i, :3],
            scale=obs_cfgs[i, 3:6],
        )
        obs_mesh = trimesh.Trimesh(vertices=obs_mesh.vertices, faces=obs_mesh.triangles)
        obs_list.append(obs_mesh)
    mesh_list = [f_mesh, t_mesh] + obs_list
    scene_mesh = trimesh.boolean.union(mesh_list).as_open3d
    return scene_mesh

def create_hook_scene_mesh(hook_cfg, origin=(0, 0, 0)):
    if hook_cfg is None:
        hook_cfg = [{
            "pos": [0, 0, 0],
            "scale": 1,
            "rotation": 0
        }]
    elif isinstance(hook_cfg, np.ndarray):
        hook_cfg_array = hook_cfg.reshape(-1, 5)
        hook_cfg = []
        for i in range(hook_cfg_array.shape[0]):
            hook_cfg.append({
                "pos": hook_cfg_array[i, :3],
                "scale": hook_cfg_array[i, 3],
                "rotation": hook_cfg_array[i, 4]
            })
    num_hooks = len(hook_cfg)
    mesh_list = []
    mesh_file_list = []
    for i in range(num_hooks):
        mesh_file_list.append(f"./model_lib/asset/hooks/nist4_wire_corner_{i}/nist4_wire_corner_{i}.obj")
    f_mesh = create_box_mesh(
        scale=(1, 1, 1),
        trans=(0, 0, -1)
    )
    f_mesh = f_mesh.translate(np.array(origin))
    f_mesh = trimesh.Trimesh(vertices=f_mesh.vertices, faces=f_mesh.triangles)
    mesh_list.append(f_mesh)
    for i in range(len(mesh_file_list)):
        pos_config, scale_config, rotation_angle_config = hook_cfg[i]['pos'], hook_cfg[i]['scale'], hook_cfg[i][
            'rotation']
        mesh_template = o3d.io.read_triangle_mesh(mesh_file_list[i])
        mesh = copy.deepcopy(mesh_template)
        v = np.array(mesh.vertices)
        scale = np.array([scale_config, scale_config, scale_config])
        trans = np.array(pos_config)
        rotate_angle = rotation_angle_config / 180 * np.pi  # Counter-clockwise rotation angle
        cos_angle = np.cos(rotate_angle)
        sin_angle = np.sin(rotate_angle)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
        v = v @ rotation_matrix.T * np.array(scale).reshape(1, 3) + np.array(trans).reshape(1, 3)
        mesh.vertices = o3d.utility.Vector3dVector(v)
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.triangles)
        mesh_list.append(mesh)
    scene_mesh = trimesh.boolean.union(mesh_list).as_open3d
    # plot_mesh(np.array(mesh_template.vertices), np.array(mesh_template.triangles)).show()
    return scene_mesh


# def create_hook_scene_mesh(hook_cfg):
#     if hook_cfg is None:
#         hook_cfg = [{
#             "pos": [0, 0, 0],
#             "scale": 1,
#             "rotation": 0
#         }]
#     elif isinstance(hook_cfg, np.ndarray):
#         hook_cfg_array = hook_cfg.reshape(-1, 5)
#         hook_cfg = []
#         for i in range(hook_cfg_array.shape[0]):
#             hook_cfg.append({
#                 "pos": hook_cfg_array[i, :3],
#                 "scale": hook_cfg_array[i, 3],
#                 "rotation": hook_cfg_array[i, 4]
#             })
#     num_hooks = len(hook_cfg)
#     mesh_list = []
#     mesh_template = o3d.io.read_triangle_mesh("model_lib/asset/hooks/nist4_wire_corner_0/nist4_wire_corner_0.obj")
#     f_mesh = create_box_mesh(
#         scale=(1, 1, 1),
#         trans=(0, 0, -1)
#     )
#     f_mesh = trimesh.Trimesh(vertices=f_mesh.vertices, faces=f_mesh.triangles)
#     mesh_list.append(f_mesh)
#     for i in range(num_hooks):
#         pos_config, scale_config, rotation_config = hook_cfg[i]['pos'], hook_cfg[i]['scale'], hook_cfg[i][
#             'rotation']
#         mesh = copy.deepcopy(mesh_template)
#         new_mesh = (mesh.rotate(Rotation.from_euler('z', rotation_config, degrees=True).as_matrix())
#                     .scale(scale_config, (0,0,0) )
#                     .translate(pos_config))
#         mesh = trimesh.Trimesh(vertices=new_mesh.vertices, faces=new_mesh.triangles)
#         mesh_list.append(mesh)
#     scene_mesh = trimesh.boolean.union(mesh_list).as_open3d
#     # plot_mesh(np.array(scene_mesh.vertices), np.array(scene_mesh.triangles)).show()
#     return scene_mesh

def create_rope_scene_mesh(table_cfg):
    cur_v = v.copy()
    box_v = create_box(size=np.array([.1, .1, table_cfg[1] / 2]),
                       trans=np.array([0.6, .15, table_cfg[1] / 2]))
    # order after union will change, so we need to match them
    cur_v[[0, 1, 2, 3, 4, 5, 6, 10]] = box_v[[0, 3, 7, 2, 6, 4, 1, 5]]
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(cur_v), o3d.utility.Vector3iVector(f))
    return mesh


def mesh_farthest_points_sampling(mesh, num_points, init_factor=5, return_normal=True):
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    face_areas = mesh.faces_areas_packed()
    faces_verts = verts[faces]
    invalid_face = torch.any(faces_verts[..., 2] < -0.005, dim=-1)
    face_areas[invalid_face] = 0
    barycentric_coords, init_face_idx = mesh_sample_barycentric_torch(verts, faces, face_areas,
                                                                      num_points * init_factor)
    init_points = barycentric_interpolation_torch1(barycentric_coords, verts, faces[init_face_idx])

    # plot_seg_fig(init_points.detach().cpu().numpy(), init_face_idx.detach().cpu().numpy()).show()
    pts_pos, point_id = sample_farthest_points(init_points[None], K=num_points)
    pts_pos = pts_pos[0]
    point_id = point_id[0]
    face_idx = init_face_idx[point_id]
    points_normal = mesh.faces_normals_packed()[face_idx]
    if return_normal:
        points = torch.concatenate([pts_pos, points_normal], dim=-1)
    else:
        points = pts_pos
    # plot_seg_fig(points.detach().cpu().numpy(), face_idx.detach().cpu().numpy()).show()
    return points, face_idx


def mesh_farthest_points_sampling_rope(mesh, num_points, init_factor=10, hook_ratio=0.1,
                                       scene_range=None,
                                       cached_barycentric=None,
                                       ):
    """
    Sample the point cloud on the hook separately since it's very small
    """
    if scene_range is None:
        scene_range = ROPE_SCENE_RANGE
    scene_range = scene_range.to(mesh.device)
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()

    if cached_barycentric is not None:
        barycentric_coords, init_face_idx = cached_barycentric
        points = barycentric_interpolation_torch1(barycentric_coords, verts, faces[init_face_idx])
        plot_pointclouds([points]).show()
        return points
    face_areas = mesh.faces_areas_packed()
    faces_verts = verts[faces]
    invalid_face = torch.any(faces_verts[..., 2] < -0.005, dim=-1)
    face_areas[invalid_face] = 0
    hook_face = torch.any(faces_verts[..., 2] > 0.005, dim=-1)
    face_areas[hook_face] *= 10
    barycentric_coords, init_face_idx = mesh_sample_barycentric_torch(verts, faces, face_areas,
                                                                      num_points * init_factor)
    init_points = barycentric_interpolation_torch1(barycentric_coords, verts, faces[init_face_idx])
    is_hook_pts = init_points[:, 2] > 0.005
    hook_points = init_points[is_hook_pts]
    other_points = init_points[~is_hook_pts]
    hook_pts_tgt_num = min(int(num_points * hook_ratio), len(hook_points))
    other_pts_tgt_num = num_points - hook_pts_tgt_num
    hook_pts_pos, hook_idx = sample_farthest_points(hook_points[None], K=hook_pts_tgt_num)
    other_pts_mask = torch.all(other_points > scene_range[0], dim=-1) & torch.all(other_points < scene_range[1], dim=-1)
    other_pts_pos, other_idx = sample_farthest_points(other_points[other_pts_mask][None], K=other_pts_tgt_num)

    points = torch.cat([hook_pts_pos[0], other_pts_pos[0]], dim=0)
    return points


def get_table_scene_sdf(table_cfg, query_range=None, res=0.02):
    if query_range is None:
        query_range = torch.tensor([
            [-0.5, 1.5],
            [-1, 1],
            [-0.5, 1.5],
        ], device="cuda")

    coords, pts = pv.get_coordinates_and_points_in_grid(res, query_range,
                                                        device=query_range.device,
                                                        dtype=torch.float16)
    pts = pts + torch.randn(*pts.shape, device=query_range.device).float() * 1e-4
    scene_mesh = create_table_scene_mesh(table_cfg)
    scene_sdf = pv.MeshSDF(pv.MeshObjectFactory(mesh=scene_mesh))
    sdf = scene_sdf(pts)[0]
    sdf_np = sdf.view(len(coords[0]), len(coords[1]), len(coords[2])).half().detach().cpu()
    del query_range
    del sdf
    del pts
    del coords
    return sdf_np


def get_hook_scene_sdf(hook_cfg, query_range=None, res=0.02):
    if query_range is None:
        query_range = torch.tensor([
            [-0.5, 1.5],
            [-1, 1],
            [-0.5, 1.5],
        ], device="cuda")
    scene_cfg = np.array([0, 0.5])
    coords, pts = pv.get_coordinates_and_points_in_grid(res, query_range, dtype=torch.float16,
                                                        device=query_range.device)
    pts = pts + torch.randn(*pts.shape, device=query_range.device).float() * 1e-4
    scene_mesh = create_hook_scene_mesh(hook_cfg)
    scene_sdf = pv.MeshSDF(pv.MeshObjectFactory(mesh=scene_mesh))
    sdf = scene_sdf(pts)[0]
    sdf_np = sdf.view(len(coords[0]), len(coords[1]), len(coords[2])).half().detach().cpu()
    del query_range
    del sdf
    del pts
    del coords
    return sdf_np


def get_rope_scene_sdf(scene_cfg, query_range=None, res=0.02):
    if query_range is None:
        query_range = torch.tensor([
            [-0.5, 1.5],
            [-1, 1],
            [-0.5, 1.5],
        ], device="cuda")
    scene_cfg = np.array([0, 0.5])
    coords, pts = pv.get_coordinates_and_points_in_grid(res, query_range, dtype=torch.float16)
    pts = pts + torch.randn(*pts.shape).float() * 1e-4
    scene_mesh = create_rope_scene_mesh(scene_cfg)
    scene_sdf = pv.MeshSDF(pv.MeshObjectFactory(mesh=scene_mesh))
    sdf = scene_sdf(pts)[0]
    sdf = sdf.view(len(coords[0]), len(coords[1]), len(coords[2])).half()
    return sdf


def sdf_query_pytorch(
        pcl: pytorch3d.structures.Pointclouds,
        mesh: pytorch3d.structures.Meshes,
        min_triangle_area=1e-4
):
    """
    scene_mesh: 1 meshes
    smpl_pc: B pointclouds

    - Distance is computed per point cloud to the corresponding mesh
    - check_sign: check signs for B point clouds to the same mesh in B configurations.
        *For loop over B* so we can also iterate over mesh!
        - verts: B x N x 3
        - faces: M x 3
        - points: B x K x 3

    Return:
    distance: B x K float
    closest: B x K x 3 float
    """
    # TODO: add B point clouds to B mesh support
    verts = mesh.verts_packed()[None]
    faces = mesh.faces_packed()
    points = pcl.points_packed()[None]
    distance, face_idx, closest = point_mesh_closest_point(mesh, pcl, min_triangle_area=min_triangle_area)
    distance = torch.sqrt(distance)

    is_inside = check_sign(verts,
                           faces,
                           points)[0]
    distance[is_inside] *= -1
    return distance, closest, face_idx


def batch_sdf_query_pytorch(
        pcls: pytorch3d.structures.Pointclouds,
        meshes: pytorch3d.structures.Meshes,
        min_triangle_area=1e-4
):
    """
    25% faster than for loop over single query function above since point_mesh_dis query is batched

    scene_mesh: B meshes
    smpl_pc: B pointclouds  B x N x 3

    - Distance is computed per point cloud to the corresponding mesh
    - check_sign: check signs for B point clouds to the same mesh in B configurations.
        *For loop over B* so we can also iterate over mesh!
        - verts: B x N x 3
        - faces: M x 3
        - points: B x K x 3

    Return:
    distance: B x K float
    closest: B x K x 3 float
    """

    distance, face_idx, closest = point_mesh_closest_point(meshes, pcls, min_triangle_area=min_triangle_area)
    distance = torch.sqrt(distance)
    distance = rearrange(distance, "(b n) -> b n", b=len(pcls))
    closest = rearrange(closest, "(b n) d -> b n d", b=len(pcls))

    for i in range(len(pcls)):
        m = meshes[i]
        pc = pcls[i]
        verts = m.verts_packed()[None]
        faces = m.faces_packed()
        points = pc.points_packed()[None]
        is_inside = check_sign(verts,
                               faces,
                               points)[0]
        distance[i, is_inside] *= -1

    return distance, closest, face_idx
