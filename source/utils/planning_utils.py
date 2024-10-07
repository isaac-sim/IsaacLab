import time

import numpy as np
import torch
from einops import rearrange
from scipy.spatial import ckdtree
from scipy.spatial.distance import cdist

from pytorch3d.structures import Pointclouds
from utils.camera_utils import is_in_frame
from utils.geom_utils import get_table_scene_sdf, get_rope_scene_sdf, get_hook_scene_sdf, create_hook_scene_mesh, \
    o3d_mesh_to_pytorch3d, batch_sdf_query_pytorch
import torch.nn.functional as F


def find_2D_rigid_trans(pts1, pts2, use_chamfer=False):
    """Given two set aligned points, find the best-fit 2D rigid transformation by kabsch algorithm"""
    uv1 = pts1[:, [0, 1]]
    uv2 = pts2[:, [0, 1]]
    centroid1 = uv1.mean(0, keepdims=True)
    centroid2 = uv2.mean(0, keepdims=True)
    zero_pts1 = uv1 - centroid1
    zero_pts2 = uv2 - centroid2
    H = zero_pts1.T @ zero_pts2

    U, S, Vh = np.linalg.svd(H)
    d = np.linalg.det(Vh.T @ U.T)
    m = np.eye(2)
    m[1, 1] = d

    R = Vh.T @ m @ U.T
    t = centroid2.T - R @ centroid1.T

    rigid_mat = np.concatenate([R, np.zeros((2, 1))], axis=1)
    rigid_mat[:, 2:3] += t
    return rigid_mat


def get_closest_point(pred_points, gt_points):
    gt_tree = ckdtree.cKDTree(gt_points)
    forward_distance, forward_nn_idx = gt_tree.query(pred_points, k=1)

    return forward_nn_idx


def parallel_generate_actions(args):
    """Sample trajectories for pick and place primitive"""
    planning_cfg, pc, depth, matrix_world_to_camera, = args
    uv_sample_method, delta_y, bb_margin, move_distance_range, is_softgym, a_type = planning_cfg
    if is_softgym:
        height_dim = 1
    else:
        height_dim = 2

    if a_type is None:  # randomly sample a_type if not specified
        if np.random.rand() < 0.5:
            a_type = 'lift'
        else:
            a_type = 'slant'

    image_size = depth.shape
    us, vs = depth.nonzero()
    uv = np.stack([us, vs], axis=1)

    # first stage: move to a random chosen point
    if uv_sample_method == 'uniform':
        u, v = np.random.randint(0, image_size[1]), np.random.randint(0, image_size[0])
    elif uv_sample_method == 'bounding_box':
        lb_u, lb_v, ub_u, ub_v = int(np.min(us)), int(np.min(vs)), int(np.max(us)), int(np.max(vs))
        u = np.random.randint(max(lb_u - bb_margin, 0), min(ub_u + bb_margin, image_size[1]))
        v = np.random.randint(max(lb_v - bb_margin, 0), min(ub_v + bb_margin, image_size[0]))
    elif uv_sample_method == 'bias_towards_edge':
        lb_u, lb_v, ub_u, ub_v = int(np.min(us)), int(np.min(vs)), int(np.max(us)), int(np.max(vs))
        while True:
            u = np.random.randint(max(lb_u - bb_margin, 0), min(ub_u + bb_margin, image_size[1]))
            v = np.random.randint(max(lb_v - bb_margin, 0), min(ub_v + bb_margin, image_size[0]))
            if u not in us and v not in vs:  # this will be then projected to the edge
                break
    elif uv_sample_method == 'cloth_mask':
        pick_id = np.random.randint(0, len(uv))
        u, v = uv[pick_id]
    dist = cdist(np.array([u, v]).reshape(1, 2), uv)
    pick_id = np.argmin(dist.flatten())
    pick_xyz = pc[pick_id].copy()
    pick_xyz[height_dim] += 0.01
    # print(pick_xyz)
    # place_uv = np.uniform(0, image_size[0]-bb_margin, size=(2,)).astype(np.int32)

    # second stage: choose a random (x, z) direction, move towards that direction for 30 steps.
    while True:
        # move_theta = npn = np.zeros(3)
        move_theta = np.random.rand() * 2 * np.pi - np.pi
        move_direction = np.zeros(3)
        move_direction[0] = np.cos(move_theta)
        if is_softgym:
            move_direction[2] = np.sin(move_theta)
        else:
            move_direction[1] = np.sin(move_theta)
        move_distance = np.random.uniform(move_distance_range[0], move_distance_range[1])
        movement = move_distance * move_direction
        after_pos = pick_xyz + movement
        if is_in_frame(after_pos.reshape(1, 3), matrix_world_to_camera, image_size):
            break
    up_height = np.random.uniform(delta_y[0], delta_y[1])
    # todo: check it to a ratio that is proportional to the total length
    num_step1 = 40
    num_step2 = 60
    actions = np.zeros((180, 4))
    if a_type == "lift":
        actions[:num_step1, height_dim] = up_height / num_step1
        actions[num_step1:num_step1 + num_step2, :3] = movement / num_step2
        actions[:num_step1 + num_step2, 3] = 1  # pick
        actions[num_step1 + num_step2:, 3] = -1  # release
    elif a_type == "slant":
        actions[:num_step1 + num_step2, :3] = movement / (num_step1 + num_step2)
        actions[:num_step1 + num_step2, height_dim] = up_height / (num_step1 + num_step2)
        actions[:num_step1 + num_step2, 3] = 1
        actions[num_step1 + num_step2:, 3] = -1  # release
    # return target_pos, delta_move
    return pick_xyz, actions


class CollisionMesh:
    def __init__(self,
                 device=torch.device('cuda'),
                 ):
        self.d = device
        self.mesh = None

    def compute_collision_cost(self, trajs, is_sg=False, col_buf=0.0):
        if len(trajs.shape) == 2:
            trajs = trajs.unsqueeze(0)
        trajs = trajs.type(self.mesh.verts_packed().dtype).to(self.mesh.device)
        num_traj, traj_len, a_dim = trajs.shape
        trajs = trajs.reshape(-1, a_dim)
        pcls = Pointclouds(points=[trajs])
        sampled_sdf, _, _ = batch_sdf_query_pytorch(pcls, self.mesh, min_triangle_area=1e-4)
        trajs_cost = torch.relu(-sampled_sdf + col_buf)
        trajs_cost = trajs_cost.view(num_traj, traj_len)
        return trajs_cost


class CollisionSDFGrid:
    def __init__(self,
                 lower_corner=(-0.5, -1, -0.5),
                 upper_corner=(1.5, 1, 1.5),
                 grid_res=0.02,
                 device=torch.device('cuda'),
                 float_dtype=torch.float32,
                 ):
        """A virtual grid that capture the collision geometry of the environment."""
        self.lower_corner = lower_corner
        self.upper_corner = upper_corner
        self.query_range = torch.tensor([lower_corner, upper_corner], device=device)
        self.grid_res = grid_res
        self.sdf_grid = None
        self.d = device
        # self.grid_shape = (
        #         (torch.tensor(upper_corner) - torch.tensor(lower_corner)) / grid_res).ceil().long().to(
        #     device)
        # print("Collision grid shape: ", self.grid_shape)
        # self.cost_volume = torch.zeros(tuple(self.grid_shape.tolist()), device=device)

    def compute_collision_cost(self, trajs, is_sg=False, col_buf=0.0):
        """
        Compute the collision cost at the given position.
        trajs: a tensor of shape (N, T, 3)
        """

        if len(trajs.shape) == 2:
            trajs = trajs.unsqueeze(0)

        trajs = trajs.type(self.sdf_grid.dtype).to(self.sdf_grid.device)
        num_traj, traj_len, a_dim = trajs.shape
        trajs = trajs.reshape(-1, a_dim)
        scene_range = self.query_range
        query_pts = (trajs - scene_range[0]) / (scene_range[1] - scene_range[0])
        query_pts = torch.clamp(query_pts, 0.0, 1.0)
        query_points_normalized = 2.0 * query_pts - 1.0
        query_points_normalized = rearrange(query_points_normalized, "(b k) d -> b k 1 1 d", b=1)

        sampled_sdf = F.grid_sample(
            input=self.sdf_grid,
            grid=query_points_normalized,
            mode='bilinear', padding_mode='border',
            align_corners=True)
        # import plotly.express as px
        # # pdb.set_trace()
        # print(sampled_sdf)
        trajs_cost = torch.relu(-sampled_sdf + col_buf)
        # trajs_cost = trajs_cost**2
        trajs_cost = trajs_cost.view(num_traj, traj_len)
        # trajs_cost = trajs_cost.sum(dim=1)
        return trajs_cost

    def update_hinge_scene_from_cfg(self, cfg):
        self.sdf_grid = get_table_scene_sdf(cfg, query_range=self.query_range.T,
                                            res=self.grid_res).float().to(self.d)
        self.sdf_grid = rearrange(self.sdf_grid, 'h w d -> 1 1 d w h')

    def update_rope_scene_from_cfg(self, cfg):
        self.sdf_grid = get_rope_scene_sdf(cfg, query_range=self.query_range.T,
                                           res=self.grid_res).float().to(self.d)
        self.sdf_grid = rearrange(self.sdf_grid, 'h w d -> 1 1 d w h')

    def update_hook_scene_from_cfg(self, cfg):
        self.sdf_grid = get_hook_scene_sdf(cfg, query_range=self.query_range.T,
                                           res=self.grid_res).float().to(self.d)
        self.sdf_grid = rearrange(self.sdf_grid, 'h w d -> 1 1 d w h')


mujoco_env = None


def mujoco_worker_init(*args):
    env_class, env_kwargs = args
    global mujoco_env
    if mujoco_env is None:
        mujoco_env = env_class(**env_kwargs)
        print("create a new mujoco env")
    time.sleep(1.5)


def mujoco_rollout_worker(args):
    env_class, env_kwargs, initial_state, batch_actions, reward_func, render_all, return_state, reset_picker_constraint = args
    global mujoco_env
    if mujoco_env is None:
        print("Lazy create")
        mujoco_env = env_class(**env_kwargs)
    rollout_results = []
    new_table_cfg = initial_state[mujoco_env.rs_table_cfg_slice]
    if np.abs(mujoco_env.table_cfg.flatten() - new_table_cfg.flatten()).sum() > 1e-5:
        is_same_scene = False
        mujoco_env.table_cfg = new_table_cfg
    else:
        is_same_scene = True
    if not is_same_scene:
        mujoco_env.reset()
        print("Reset scene")
    if "x_limit" in env_kwargs:
        mujoco_env.set_workspace_limit(env_kwargs["x_limit"])
    for actions in batch_actions:
        result = mujoco_env.rollout(initial_state, actions, render_all, return_state, reset_picker_constraint)
        # result = mujoco_env.rollout(initial_state, actions, None, render_all, False)
        rollout_results.append(result)

    return rollout_results


def pos_reward_fn(plan_states, goal_state, pos_dim=3, rwd_state='state'):
    ps = np.stack([x[rwd_state][:pos_dim] for x in plan_states])
    ps = ps.reshape(len(plan_states), -1, 10, 3)
    gs = goal_state[rwd_state][None, None, :pos_dim]
    r = -((gs - ps) ** 2).mean(axis=(-1, -2))
    return r * 1000


class ModelSelectionTree:
    def __init__(self):
        pass


class PlanningTreeNode:
    def __init__(self, parent):
        self.parent = parent
        # self.state = state
        # self.action = action
        # self.depth = depth
        self.best_reward = -1e8
        self.best_child_id = -1
        self.children = []

    def add_child(self, child):
        self.children.append(child)
