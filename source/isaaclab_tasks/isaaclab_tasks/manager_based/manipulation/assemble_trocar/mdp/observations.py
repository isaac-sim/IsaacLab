# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
G1 29DOF (body) + Dex3 joint state helpers for the assemble_trocar task.

Notes:
- DDS has been removed (simulation-only observations).
- These functions are designed to be used as Isaac Lab observation terms.
"""

from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

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
    joint_pos = wp.to_torch(env.scene["robot"].data.joint_pos)
    joint_vel = wp.to_torch(env.scene["robot"].data.joint_vel)
    joint_torque = wp.to_torch(env.scene["robot"].data.applied_torque)
    device = joint_pos.device
    batch = joint_pos.shape[0]

    # Precompute and cache column indices
    global _body_obs_cache
    if _body_obs_cache["device"] != device or _body_obs_cache["idx_t"] is None:
        body_joint_indices = [
            0,
            3,
            6,
            9,
            13,
            17,
            1,
            4,
            7,
            10,
            14,
            18,
            2,
            5,
            8,
            11,
            15,
            19,
            21,
            23,
            25,
            27,
            12,
            16,
            20,
            22,
            24,
            26,
            28,
        ]
        _body_obs_cache["idx_t"] = torch.tensor(body_joint_indices, dtype=torch.long, device=device)
        _body_obs_cache["device"] = device
        _body_obs_cache["batch"] = None  # force re-init batch-shaped buffers

    idx_t = _body_obs_cache["idx_t"]
    n = idx_t.numel()

    # Preallocate/reuse batch-shaped indices and output buffers
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

    # Fill buffers using gather(out=...) to avoid new tensor allocations
    try:
        torch.gather(joint_pos, 1, idx_batch, out=pos_buf)
        torch.gather(joint_vel, 1, idx_batch, out=vel_buf)
        torch.gather(joint_torque, 1, idx_batch, out=torque_buf)
    except TypeError:
        pos_buf.copy_(torch.gather(joint_pos, 1, idx_batch))
        vel_buf.copy_(torch.gather(joint_vel, 1, idx_batch))
        torque_buf.copy_(torch.gather(joint_torque, 1, idx_batch))

    # Combine into a single buffer to avoid cat allocations
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
    joint_pos = wp.to_torch(env.scene["robot"].data.joint_pos)
    device = joint_pos.device
    batch = joint_pos.shape[0]

    global _dex3_obs_cache
    if _dex3_obs_cache["device"] != device or _dex3_obs_cache["idx_t"] is None:
        # Dex3 joint indices in the full robot joint vector (14 DOF)
        dex3_joint_indices = [31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38]
        _dex3_obs_cache["idx_t"] = torch.tensor(dex3_joint_indices, dtype=torch.long, device=device)
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

    try:
        torch.gather(joint_pos, 1, idx_batch, out=pos_buf)
    except TypeError:
        pos_buf.copy_(torch.gather(joint_pos, 1, idx_batch))

    return pos_buf
