# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests between masked and index body-write kernels."""

from __future__ import annotations

import numpy as np
import warp as wp
from isaaclab_newton.assets import kernels as shared_kernels

_NUM_ENVS = 5
_NUM_BODIES = 4


def _selection(rng: np.random.Generator):
    """Sample matching boolean masks and compact indices for envs and bodies."""
    env_mask = rng.random(_NUM_ENVS) < 0.5
    body_mask = rng.random(_NUM_BODIES) < 0.5
    env_mask[0] = True
    body_mask[1] = True
    env_ids = np.nonzero(env_mask)[0].astype(np.int32)
    body_ids = np.nonzero(body_mask)[0].astype(np.int32)
    return env_mask, body_mask, env_ids, body_ids


def _random_transforms(rng: np.random.Generator) -> np.ndarray:
    positions = rng.uniform(-1.0, 1.0, size=(_NUM_ENVS, _NUM_BODIES, 3))
    quaternions = rng.normal(size=(_NUM_ENVS, _NUM_BODIES, 4))
    quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
    return np.concatenate([positions, quaternions], axis=-1).astype(np.float32)


def _random_spatial(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=(_NUM_ENVS, _NUM_BODIES, 6)).astype(np.float32)


def _pair(values: np.ndarray, dtype) -> tuple[wp.array, wp.array]:
    """Two independent device copies of the same initial buffer contents."""
    return wp.array(values, dtype=dtype, device="cpu"), wp.array(values, dtype=dtype, device="cpu")


def test_masked_body_link_pose_kernel_matches_index_kernel():
    rng = np.random.default_rng(7)
    env_mask_np, body_mask_np, env_ids_np, body_ids_np = _selection(rng)
    data = wp.array(_random_transforms(rng), dtype=wp.transformf, device="cpu")
    out_index, out_mask = _pair(_random_transforms(rng), wp.transformf)
    env_ids = wp.array(env_ids_np, dtype=wp.int32, device="cpu")
    body_ids = wp.array(body_ids_np, dtype=wp.int32, device="cpu")
    env_mask = wp.array(env_mask_np, dtype=wp.bool, device="cpu")
    body_mask = wp.array(body_mask_np, dtype=wp.bool, device="cpu")

    wp.launch(
        shared_kernels.set_body_link_pose_to_sim,
        dim=(env_ids_np.size, body_ids_np.size),
        inputs=[data, env_ids, body_ids, True],
        outputs=[out_index],
        device="cpu",
    )
    wp.launch(
        shared_kernels.set_body_link_pose_to_sim_mask,
        dim=(_NUM_ENVS, _NUM_BODIES),
        inputs=[data, env_mask, body_mask],
        outputs=[out_mask],
        device="cpu",
    )

    np.testing.assert_allclose(out_mask.numpy(), out_index.numpy())


def test_masked_body_com_pose_kernel_matches_index_kernel():
    rng = np.random.default_rng(11)
    env_mask_np, body_mask_np, env_ids_np, body_ids_np = _selection(rng)
    data = wp.array(_random_transforms(rng), dtype=wp.transformf, device="cpu")
    com_pos_b = wp.array(
        rng.uniform(-0.2, 0.2, size=(_NUM_ENVS, _NUM_BODIES, 3)).astype(np.float32), dtype=wp.vec3f, device="cpu"
    )
    com_index, com_mask = _pair(_random_transforms(rng), wp.transformf)
    link_index, link_mask = _pair(_random_transforms(rng), wp.transformf)
    env_ids = wp.array(env_ids_np, dtype=wp.int32, device="cpu")
    body_ids = wp.array(body_ids_np, dtype=wp.int32, device="cpu")
    env_mask = wp.array(env_mask_np, dtype=wp.bool, device="cpu")
    body_mask = wp.array(body_mask_np, dtype=wp.bool, device="cpu")

    wp.launch(
        shared_kernels.set_body_com_pose_to_sim,
        dim=(env_ids_np.size, body_ids_np.size),
        inputs=[data, com_pos_b, env_ids, body_ids, True],
        outputs=[com_index, link_index],
        device="cpu",
    )
    wp.launch(
        shared_kernels.set_body_com_pose_to_sim_mask,
        dim=(_NUM_ENVS, _NUM_BODIES),
        inputs=[data, com_pos_b, env_mask, body_mask],
        outputs=[com_mask, link_mask],
        device="cpu",
    )

    np.testing.assert_allclose(com_mask.numpy(), com_index.numpy())
    np.testing.assert_allclose(link_mask.numpy(), link_index.numpy())


def test_masked_body_com_velocity_kernel_matches_index_kernel():
    rng = np.random.default_rng(13)
    env_mask_np, body_mask_np, env_ids_np, body_ids_np = _selection(rng)
    data = wp.array(_random_spatial(rng), dtype=wp.spatial_vectorf, device="cpu")
    vel_index, vel_mask = _pair(_random_spatial(rng), wp.spatial_vectorf)
    acc_index, acc_mask = _pair(_random_spatial(rng), wp.spatial_vectorf)
    env_ids = wp.array(env_ids_np, dtype=wp.int32, device="cpu")
    body_ids = wp.array(body_ids_np, dtype=wp.int32, device="cpu")
    env_mask = wp.array(env_mask_np, dtype=wp.bool, device="cpu")
    body_mask = wp.array(body_mask_np, dtype=wp.bool, device="cpu")

    wp.launch(
        shared_kernels.set_body_com_velocity_to_sim,
        dim=(env_ids_np.size, body_ids_np.size),
        inputs=[data, env_ids, body_ids, True],
        outputs=[vel_index, acc_index],
        device="cpu",
    )
    wp.launch(
        shared_kernels.set_body_com_velocity_to_sim_mask,
        dim=(_NUM_ENVS, _NUM_BODIES),
        inputs=[data, env_mask, body_mask],
        outputs=[vel_mask, acc_mask],
        device="cpu",
    )

    np.testing.assert_allclose(vel_mask.numpy(), vel_index.numpy())
    np.testing.assert_allclose(acc_mask.numpy(), acc_index.numpy())


def test_masked_body_link_velocity_kernel_matches_index_kernel():
    rng = np.random.default_rng(17)
    env_mask_np, body_mask_np, env_ids_np, body_ids_np = _selection(rng)
    data = wp.array(_random_spatial(rng), dtype=wp.spatial_vectorf, device="cpu")
    com_pos_b = wp.array(
        rng.uniform(-0.2, 0.2, size=(_NUM_ENVS, _NUM_BODIES, 3)).astype(np.float32), dtype=wp.vec3f, device="cpu"
    )
    link_pose_w = wp.array(_random_transforms(rng), dtype=wp.transformf, device="cpu")
    link_vel_index, link_vel_mask = _pair(_random_spatial(rng), wp.spatial_vectorf)
    com_vel_index, com_vel_mask = _pair(_random_spatial(rng), wp.spatial_vectorf)
    acc_index, acc_mask = _pair(_random_spatial(rng), wp.spatial_vectorf)
    env_ids = wp.array(env_ids_np, dtype=wp.int32, device="cpu")
    body_ids = wp.array(body_ids_np, dtype=wp.int32, device="cpu")
    env_mask = wp.array(env_mask_np, dtype=wp.bool, device="cpu")
    body_mask = wp.array(body_mask_np, dtype=wp.bool, device="cpu")

    wp.launch(
        shared_kernels.set_body_link_velocity_to_sim,
        dim=(env_ids_np.size, body_ids_np.size),
        inputs=[data, com_pos_b, link_pose_w, env_ids, body_ids, True],
        outputs=[link_vel_index, com_vel_index, acc_index],
        device="cpu",
    )
    wp.launch(
        shared_kernels.set_body_link_velocity_to_sim_mask,
        dim=(_NUM_ENVS, _NUM_BODIES),
        inputs=[data, com_pos_b, link_pose_w, env_mask, body_mask],
        outputs=[link_vel_mask, com_vel_mask, acc_mask],
        device="cpu",
    )

    np.testing.assert_allclose(link_vel_mask.numpy(), link_vel_index.numpy())
    np.testing.assert_allclose(com_vel_mask.numpy(), com_vel_index.numpy())
    np.testing.assert_allclose(acc_mask.numpy(), acc_index.numpy())
