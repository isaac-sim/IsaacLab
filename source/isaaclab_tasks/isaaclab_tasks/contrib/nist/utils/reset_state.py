# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-env reset-state read/write helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence
from contextlib import contextmanager

import numpy as np
import torch
from torch import Tensor


def get_reset_state(env, env_ids: Tensor, reset_assets: Sequence[str], is_relative: bool = False) -> Tensor:
    """Read and concatenate reset-state slices for scene assets."""
    states: list[Tensor] = []
    for name, articulation in env.scene._articulations.items():
        if name in reset_assets:
            root_state = articulation.data.root_state_w.torch[env_ids]
            if is_relative:
                root_state = root_state.clone()
                root_state[:, :3] -= env.scene.env_origins[env_ids]
            states.append(root_state)
            states.append(articulation.data.joint_pos.torch[env_ids])
            states.append(articulation.data.joint_vel.torch[env_ids])

    for name, rigid_object in env.scene._rigid_objects.items():
        if name in reset_assets:
            root_state = rigid_object.data.root_state_w.torch[env_ids]
            if is_relative:
                root_state = root_state.clone()
                root_state[:, :3] -= env.scene.env_origins[env_ids]
            if rigid_object.num_mesh_variants > 1:
                states.append(rigid_object.mesh_variant_ids.torch[env_ids, None].to(root_state.dtype))
            states.append(root_state)
    return torch.cat(states, dim=-1)


def set_reset_state(env, states: Tensor, env_ids: Tensor, reset_assets: Sequence[str], is_relative: bool = False):
    """Split ``states`` by scene asset and write reset-state slices."""
    offset = 0
    for name, articulation in env.scene._articulations.items():
        if name in reset_assets:
            n_joints, width = articulation.num_joints, 13 + 2 * articulation.num_joints
            state = states[:, offset : offset + width]
            root_state = state[:, :13]
            if is_relative:
                root_state = root_state.clone()
                root_state[:, :3] += env.scene.env_origins[env_ids]
            articulation.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
            if not articulation.is_fixed_base:
                articulation.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
            articulation.write_joint_position_to_sim_index(position=state[:, 13 : 13 + n_joints], env_ids=env_ids)
            articulation.write_joint_velocity_to_sim_index(
                velocity=state[:, 13 + n_joints : 13 + 2 * n_joints],
                env_ids=env_ids,
            )
            offset += width

    for name, rigid_object in env.scene._rigid_objects.items():
        if name in reset_assets:
            if rigid_object.num_mesh_variants > 1:
                variant_ids = states[:, offset].to(dtype=torch.int32)
                rigid_object.write_mesh_variant_to_sim(variant_ids, env_ids)
                offset += 1
            root_state = states[:, offset : offset + 13]
            if is_relative:
                root_state = root_state.clone()
                root_state[:, :3] += env.scene.env_origins[env_ids]
            rigid_object.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
            if not getattr(rigid_object.root_view, "is_fixed_base", False):
                rigid_object.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
            offset += 13

    if offset != states.shape[1]:
        raise ValueError(f"Reset state has {states.shape[1]} values, but the scene consumed {offset}.")


@contextmanager
def temporary_seed(seed: int, restore_numpy: bool = True, restore_python: bool = True):
    """Seed torch / numpy / python RNGs within a ``with`` block; restore on exit.

    Saves and restores torch (CPU + all CUDA), numpy, and python ``random``
    RNG state so deterministic build-time sampling does not perturb the
    global RNGs used elsewhere.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state() if restore_numpy else None
    py_state = random.getstate() if restore_python else None

    try:
        torch.manual_seed(seed)  # seeds CPU + all visible CUDA devices
        if restore_numpy:
            np.random.seed(seed)
        if restore_python:
            random.seed(seed)
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if np_state is not None:
            np.random.set_state(np_state)
        if py_state is not None:
            random.setstate(py_state)
