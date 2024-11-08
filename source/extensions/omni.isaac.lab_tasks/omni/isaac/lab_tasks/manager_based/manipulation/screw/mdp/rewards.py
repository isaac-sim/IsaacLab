# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import torch
import torch.cuda
from torch.autograd import Function
from typing import TYPE_CHECKING

from numba import cuda, jit, prange

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

import omni.isaac.lab_tasks.manager_based.manipulation.screw.mdp as mdp

from .dtw_loss import *

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def l2_norm(diff: torch.Tensor) -> torch.Tensor:
    """Compute the L2-norm of a tensor."""
    return torch.norm(diff, dim=1)


def forge_kernel(diff: torch.Tensor, a: float = 100, b: float = 0, tol: float = 0) -> torch.Tensor:
    """Compute the kernel function using the Forge kernel.

    The kernel function is computed as:
    .. math::
        k(x) = \\frac{1}{e^{-a(x - \\text{tol})} + b + e^{a(x - \\text{tol})}}
    """
    l2_dis = l2_norm(diff)
    clamped_dis = torch.clamp(l2_dis - tol, min=0)
    dis = 1 / (torch.exp(-a * clamped_dis) + b + torch.exp(a * clamped_dis))
    return dis


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def nut_upright_reward_forge(env: ManagerBasedRLEnv, a: float = 300, b: float = 0, tol: float = 0):
    # penalize if nut is not upright
    # compute the cosine distance between the nut normal and the global up vector
    nut_quat = env.scene["nut_frame"].data.target_quat_w[:, 0]
    up_vec = torch.tensor([[0, 0, 1.0]], device=nut_quat.device)
    up_vecs = up_vec.expand(nut_quat.shape[0], 3)
    nut_up_vec = math_utils.quat_apply(nut_quat, up_vecs)
    cos_sim = torch.sum(nut_up_vec * up_vecs, dim=1, keepdim=True) / torch.norm(nut_up_vec, dim=1, keepdim=True)
    rewards = mdp.forge_kernel(1 - cos_sim, a, b, tol)
    return rewards


def get_imitation_reward_from_dtw(ref_traj, curr_ee_pos, prev_ee_traj, criterion, device):
    """Get imitation reward based on dynamic time warping."""

    soft_dtw = torch.zeros((curr_ee_pos.shape[0]), device=device)
    prev_ee_pos = prev_ee_traj[:, 0, :].squeeze()  # select the first ee pos in robot traj
    min_dist_traj_idx, min_dist_step_idx, min_dist_per_env = get_closest_state_idx(ref_traj, prev_ee_pos)
    cur_ee_traj = torch.cat([prev_ee_traj[:, 1:, :], curr_ee_pos[:, None]], dim=1)
    # cur_ee_traj = torch.roll(prev_ee_traj, shifts=-1, dims=1)
    # cur_ee_traj[:, -1, :] = curr_ee_pos

    for i in range(curr_ee_pos.shape[0]):
        traj_idx = min_dist_traj_idx[i]
        step_idx = min_dist_step_idx[i]
        curr_ee_pos_i = curr_ee_pos[i].reshape(1, 3)
        prev_ee_pos_i = prev_ee_pos[i].reshape(1, 3)

        # NOTE: in reference trajectories, larger index -> closer to goal
        traj = ref_traj[traj_idx, step_idx:, :].reshape((1, -1, 3))

        _, curr_step_idx, _ = get_closest_state_idx(traj, curr_ee_pos_i)

        if curr_step_idx == 0:
            selected_pos = ref_traj[traj_idx, step_idx, :].reshape((1, 1, 3))
            selected_traj = torch.cat([selected_pos, selected_pos], dim=1)
        else:
            selected_traj = ref_traj[traj_idx, step_idx : (curr_step_idx + step_idx), :].reshape((1, -1, 3))
        # eef_traj = torch.cat([prev_ee_pos_i, curr_ee_pos_i], dim=0).reshape((1, -1, 3))
        # eef_traj = torch.cat((prev_ee_traj[i, 1:, :], curr_ee_pos_i)).reshape((1, -1, 3))
        soft_dtw[i] = criterion(cur_ee_traj[i : i + 1], selected_traj)

    # w_task_progress = 1-(min_dist_step_idx / ref_traj.shape[1])
    w_task_progress = min_dist_step_idx / ref_traj.shape[1]

    # imitation_rwd = torch.exp(-soft_dtw)
    imitation_rwd = 1 - torch.tanh(soft_dtw)

    return imitation_rwd * w_task_progress, cur_ee_traj
