import pdb

from einops import rearrange
# from pytorch3d.loss.point_mesh_distance import point_face_distance, face_point_distance
from pytorch3d.ops import knn_points, knn_gather
from pytorch3d.loss.chamfer import _validate_chamfer_reduction_inputs, _handle_pointcloud_input
from typing import Union
import torch
from pytorch3d.structures import Meshes, Pointclouds
import numpy as np


def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def chamfer_distance(
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
        one_chamfer=False,
        batch_reduction: Union[str, None] = "mean",
        point_reduction: str = "mean",
        K=1
):
    # _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=K)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=K)

    cham_x = x_nn.dists.mean(-1)
    cham_y = y_nn.dists.mean(-1)
    # cham_x = x_nn.dists[..., 0]  # (N, P1)
    # cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if point_reduction == "none":
        return cham_x, cham_y

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    if one_chamfer:
        cham_dist = cham_x
        cham_normals = cham_norm_x if return_normals else None
    else:
        cham_dist = cham_x + cham_y
        cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist, cham_normals


def chamfer_oneway_distance(
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
        batch_reduction: Union[str, None] = "mean",
        point_reduction: str = "mean",
        K=1
):
    # _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=K)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=K)

    cham_x = x_nn.dists.mean(-1)
    cham_y = y_nn.dists.mean(-1)
    # cham_x = x_nn.dists[..., 0]  # (N, P1)
    # cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if point_reduction == "none":
        return cham_x, cham_y

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist, cham_normals


def kl_normal(dist1, dist2, stop_grad1=False, stop_grad2=False, eps=1e-6, dist_mode='std'):
    """
    dist1, dist2: (mu, std)
    dist1 will be the weight
    mode: std, var, logvar
    forward KL divergence between two normal distribution. Stop gradient for more flexibility.
    """
    if dist_mode == 'std':
        mu1, std1 = dist1
        mu2, std2 = dist2
        if stop_grad1:
            mu1, std1 = mu1.detach(), std1.detach()
        if stop_grad2:
            mu2, std2 = mu2.detach(), std2.detach()
        std1 = torch.maximum(std1, torch.ones_like(std1) * eps)
        std2 = torch.maximum(std2, torch.ones_like(std2) * eps)
        var1 = std1 ** 2
        var2 = std2 ** 2
        var_ratio = var1 / var2
        t1 = (mu1 - mu2).pow(2) / var2
        KLD = 0.5 * torch.mean(var_ratio + t1 - 1 - var_ratio.log())
    elif dist_mode == 'var':
        mu1, var1 = dist1
        mu2, var2 = dist2
        if stop_grad1:
            mu1, var1 = mu1.detach(), var1.detach()
        if stop_grad2:
            mu2, var2 = mu2.detach(), var2.detach()
        var1 = torch.maximum(var1, torch.ones_like(var1) * eps ** 2)
        var2 = torch.maximum(var2, torch.ones_like(var2) * eps ** 2)
        var_ratio = var1 / var2
        t1 = (mu1 - mu2).pow(2) / var2
        KLD = 0.5 * torch.mean(var_ratio + t1 - 1 - var_ratio.log())
    elif dist_mode == 'logvar':
        mu1, logvar1 = dist1
        mu2, logvar2 = dist2
        if stop_grad1:
            mu1, logvar1 = mu1.detach(), logvar1.detach()
        if stop_grad2:
            mu2, logvar2 = mu2.detach(), logvar2.detach()
        t1 = (mu1 - mu2).pow(2) / torch.exp(logvar2)
        KLD = 0.5 * torch.mean((logvar1 - logvar2).exp() + t1 - 1 - logvar1 + logvar2)
    return KLD


def compute_contact_feat_scala(pcs, table_cfgs, tolerance=0.005):
    table_cfgs = rearrange(table_cfgs, "b d -> b 1 d")
    penetrate_floor = pcs[..., 2] + tolerance < 0
    penetrate_table = (pcs[..., 0] - tolerance > table_cfgs[..., 0] - 0.4) & \
                      (pcs[..., 0] + tolerance < table_cfgs[..., 0] + 0.4) & \
                      (pcs[..., 2] + tolerance < table_cfgs[..., 1])
    penetrate = penetrate_floor | penetrate_table
    batch_penetrate = penetrate.float()
    batch_penetrate = batch_penetrate.mean(-1, keepdim=True)
    return batch_penetrate


def compute_contact_feat_diff(pcs, table_cfgs, tolerance=0.005):
    # penetration dis
    # b is number of real batch x num_subgoal
    # pcs is B x K x 3
    table_cfgs = rearrange(table_cfgs, "b d -> b 1 d")
    # penetrate_floor = pcs[..., 2] + tolerance < 0
    penetrate_table = (pcs[..., 0] - tolerance > table_cfgs[..., 0] - 0.4) & \
                      (pcs[..., 0] + tolerance < table_cfgs[..., 0] + 0.4) & \
                      (pcs[..., 2] + tolerance < table_cfgs[..., 1])
    penetrate_floor_score = torch.relu(-pcs[..., 2] - tolerance) ** 2

    # dis to left side of table, dis to right side of table, dis to top of table
    penetrate_table_score = torch.relu(pcs[..., 0] - tolerance - table_cfgs[..., 0] + 0.4) ** 2 + \
                            torch.relu(table_cfgs[..., 0] + 0.4 - pcs[..., 0] - tolerance) ** 2 + \
                            torch.relu(table_cfgs[..., 1] - pcs[..., 2] - tolerance) ** 2
    penetrate_score = penetrate_floor_score + penetrate_table_score * penetrate_table  # B x K
    penetrate_score = penetrate_score.mean(-1, keepdim=True)
    return penetrate_score


def compute_contact_feat_diff2(pcs,
                               table_cfgs,
                               tolerance=0.005,
                               metric="l1",
                               reduction="mean"):
    # penetration dis sdf
    # b is number of real batch x num_subgoal
    # pcs is B x K x 3
    table_cfgs = rearrange(table_cfgs, "b d -> b 1 d")

    penetrate_floor_score = torch.relu(-pcs[..., 2] - tolerance) ** 2
    # B x K x 3
    penetrate_table_dis = torch.stack([
        torch.relu(pcs[..., 0] - tolerance - table_cfgs[..., 0] + 0.4),
        torch.relu(table_cfgs[..., 0] + 0.4 - pcs[..., 0] - tolerance),
        torch.relu(table_cfgs[..., 1] - pcs[..., 2] - tolerance)
    ], dim=-1)

    # penetrate_dis = penetrate_table_dis[penetrate_table]
    # if not penetrate table, min will give zero
    approx_penetrate_sdf = penetrate_table_dis.min(-1)[0]
    if metric == "l2":
        penetrate_table_score = approx_penetrate_sdf ** 2
    elif metric == "l1":
        penetrate_table_score = approx_penetrate_sdf
    else:
        raise NotImplementedError
    # penetrate_table_score = torch.relu(pcs[..., 0] - tolerance - table_cfgs[..., 0] + 0.4) ** 2 + \
    #                         torch.relu(table_cfgs[..., 0] + 0.4 - pcs[..., 0] - tolerance) ** 2 + \
    #                         torch.relu(table_cfgs[..., 1] - pcs[..., 2] - tolerance) ** 2
    penetrate_score = penetrate_floor_score + penetrate_table_score  # B x K
    # reduction over points
    if reduction == "mean":
        penetrate_score = penetrate_score.mean(-1, keepdim=True)
    elif reduction == "max":
        penetrate_score = penetrate_score.max(-1, keepdim=True)[0]
    return penetrate_score


def traj_len_loss(trajs):
    # trajs is B x T x N
    # distance in the state space?
    dis = 0
    for k in trajs:
        new_dis = torch.norm(trajs[k][:, 1:] - trajs[k][:, :-1], dim=-1)
        while len(new_dis.shape) > 2:
            new_dis = new_dis.mean(-1)
        dis += new_dis
    # pdb.set_trace()
    # dis = torch.mean((trajs[:, 1:] - trajs[:, :-1])**2, dim=-1)
    return dis


def compute_pts_dis(plan_pts, goal_pts, mode="norm", reduction="mean", dis_func=None):
    """
    M x N x 3
    """
    is_registered = plan_pts.shape[-2] == goal_pts.shape[-2]
    if is_registered:
        if plan_pts.shape[0] != goal_pts.shape[0]:
            plan_pts = plan_pts.unsqueeze(1)
            goal_pts = goal_pts.unsqueeze(0)
        plan_pts, goal_pts = torch.broadcast_tensors(plan_pts, goal_pts)
        if dis_func is None:
            if mode == "l2":
                dis = ((plan_pts - goal_pts) ** 2).mean(-1)
            else:
                dis = torch.norm(plan_pts - goal_pts, dim=-1, p=2)
        else:
            dis = dis_func(plan_pts, goal_pts)
        if reduction == "mean":
            dis = dis.mean(-1)
        return dis
    else:
        print("Unregister ")
        exit()


from pytorch3d import _C
from pytorch3d.structures import Meshes, Pointclouds
from torch.autograd import Function
from torch.autograd.function import once_differentiable

_DEFAULT_MIN_TRIANGLE_AREA: float = 5e-3


class _PointFaceDistance(Function):
    """
    Torch autograd Function wrapper PointFaceDistance Cuda implementation
    """

    @staticmethod
    def forward(
            ctx,
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area=_DEFAULT_MIN_TRIANGLE_AREA,
    ):
        """
        Args:
            ctx: Context object used to calculate gradients.
            points: FloatTensor of shape `(P, 3)`
            points_first_idx: LongTensor of shape `(N,)` indicating the first point
                index in each example in the batch
            tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
                triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
            tris_first_idx: LongTensor of shape `(N,)` indicating the first face
                index in each example in the batch
            max_points: Scalar equal to maximum number of points in the batch
            min_triangle_area: (float, defaulted) Triangles of area less than this
                will be treated as points/lines.
        Returns:
            dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
                euclidean distance of `p`-th point to the closest triangular face
                in the corresponding example in the batch
            idxs: LongTensor of shape `(P,)` indicating the closest triangular face
                in the corresponding example in the batch.

            `dists[p]` is
            `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
            where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
            face `(v0, v1, v2)`

        """
        dists, idxs = _C.point_face_dist_forward(
            points,
            points_first_idx,
            tris,
            tris_first_idx,
            max_points,
            min_triangle_area,
        )
        ctx.save_for_backward(points, tris, idxs)
        ctx.min_triangle_area = min_triangle_area
        return dists, idxs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists):
        grad_dists = grad_dists.contiguous()
        points, tris, idxs = ctx.saved_tensors
        min_triangle_area = ctx.min_triangle_area
        grad_points, grad_tris = _C.point_face_dist_backward(
            points, tris, idxs, grad_dists, min_triangle_area
        )
        return grad_points, None, grad_tris, None, None, None


point_face_distance = _PointFaceDistance.apply


def point_mesh_face_distance(
        meshes: Meshes,
        pcls: Pointclouds,
        min_triangle_area: float = 5e-3,
):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds
        min_triangle_area: (float, defaulted) Triangles of area less than this
            will be treated as points/lines.

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    # max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face, idxs = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )

    return point_to_face, idxs
