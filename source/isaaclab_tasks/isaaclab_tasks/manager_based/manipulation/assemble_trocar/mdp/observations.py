# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 29DOF (body) + Dex3 joint state helpers for the assemble_trocar task.

Notes:
- DDS has been removed (simulation-only observations).
- These functions are designed to be used as Isaac Lab observation terms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab_tasks.manager_based.manipulation.assemble_trocar.config import (
    G1_29DOF_BODY_JOINT_INDICES,
    G1_DEX3_JOINT_INDICES,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Observation cache: index tensors + preallocated output buffers (body joints)
_body_obs_cache = {
    "device": None,
    "batch": None,
    "idx_t": None,
    "idx_batch": None,
    "pos_buf": None,
    "vel_buf": None,
    "torque_buf": None,
    "combined_buf": None,
}


def get_robot_body_joint_states(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return body joint states as a single tensor: [pos(29) | vel(29) | torque(29)]."""
    robot_data = env.scene["robot"].data
    joint_pos = robot_data.joint_pos.torch
    joint_vel = robot_data.joint_vel.torch
    joint_torque = robot_data.applied_torque.torch
    device = joint_pos.device
    batch = joint_pos.shape[0]

    global _body_obs_cache
    if _body_obs_cache["device"] != device or _body_obs_cache["idx_t"] is None:
        _body_obs_cache["idx_t"] = torch.tensor(G1_29DOF_BODY_JOINT_INDICES, dtype=torch.long, device=device)
        _body_obs_cache["device"] = device
        _body_obs_cache["batch"] = None

    idx_t = _body_obs_cache["idx_t"]
    n = idx_t.numel()

    if _body_obs_cache["batch"] != batch or _body_obs_cache["idx_batch"] is None:
        _body_obs_cache["idx_batch"] = idx_t.unsqueeze(0).expand(batch, n)
        _body_obs_cache["pos_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _body_obs_cache["vel_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _body_obs_cache["torque_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _body_obs_cache["combined_buf"] = torch.empty(batch, n * 3, device=device, dtype=joint_pos.dtype)
        _body_obs_cache["batch"] = batch

    idx_batch = _body_obs_cache["idx_batch"]
    pos_buf = _body_obs_cache["pos_buf"]
    vel_buf = _body_obs_cache["vel_buf"]
    torque_buf = _body_obs_cache["torque_buf"]
    combined_buf = _body_obs_cache["combined_buf"]

    torch.gather(joint_pos, 1, idx_batch, out=pos_buf)
    torch.gather(joint_vel, 1, idx_batch, out=vel_buf)
    torch.gather(joint_torque, 1, idx_batch, out=torque_buf)

    combined_buf[:, 0:n].copy_(pos_buf)
    combined_buf[:, n : 2 * n].copy_(vel_buf)
    combined_buf[:, 2 * n : 3 * n].copy_(torque_buf)
    return combined_buf


# Observation cache: index tensors + preallocated output buffers (Dex3 hand joints)
_dex3_obs_cache = {
    "device": None,
    "batch": None,
    "idx_t": None,
    "idx_batch": None,
    "pos_buf": None,
}


def get_robot_dex3_joint_states(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return Dex3 joint positions [batch, 14]."""
    joint_pos = env.scene["robot"].data.joint_pos.torch
    device = joint_pos.device
    batch = joint_pos.shape[0]

    global _dex3_obs_cache
    if _dex3_obs_cache["device"] != device or _dex3_obs_cache["idx_t"] is None:
        _dex3_obs_cache["idx_t"] = torch.tensor(G1_DEX3_JOINT_INDICES, dtype=torch.long, device=device)
        _dex3_obs_cache["device"] = device
        _dex3_obs_cache["batch"] = None

    idx_t = _dex3_obs_cache["idx_t"]
    n = idx_t.numel()

    if _dex3_obs_cache["batch"] != batch or _dex3_obs_cache["idx_batch"] is None:
        _dex3_obs_cache["idx_batch"] = idx_t.unsqueeze(0).expand(batch, n)
        _dex3_obs_cache["pos_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _dex3_obs_cache["batch"] = batch

    idx_batch = _dex3_obs_cache["idx_batch"]
    pos_buf = _dex3_obs_cache["pos_buf"]

    torch.gather(joint_pos, 1, idx_batch, out=pos_buf)

    return pos_buf
