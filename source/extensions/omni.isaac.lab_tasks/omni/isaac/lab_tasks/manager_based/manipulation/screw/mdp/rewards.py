# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

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

# def position_error_l2(env: ManagerBasedRLEnv, src_body_name: str, tgt_body_name: str) -> torch.Tensor:
#     """Penalize tracking of the position error using L2-norm.

#     The function computes the position error between the desired position (from the command) and the
#     current position of the asset's body (in world frame). The position error is computed as the L2-norm
#     of the difference between the desired and current positions.
#     """
#     # extract the asset (to enable type hinting)
#     src_asset: RigidObject = env.scene[src_body_name]
#     tgt_asset: RigidObject = env.scene[tgt_body_name]
#     src_pos = src_asset.data.root_pos_w - env.scene.env_origins
#     tgt_pos = tgt_asset.data.root_pos_w - env.scene.env_origins
#     return torch.norm(src_pos - tgt_pos, dim=1)

# def position_error_forge(env: ManagerBasedRLEnv, src_body_name: str, tgt_body_name: str,
#                          a=100, b=0, tol=0.) -> torch.Tensor:
#     l2_dis = position_error_l2(env, src_body_name, tgt_body_name)
#     clamped_dis = torch.clamp(l2_dis - tol, min=0)
#     dis = 1 / (torch.exp(-a * clamped_dis) + b + torch.exp(a * clamped_dis))
#     return dis    

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
