# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pytest
import torch
import trimesh

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

# Import after app launch
import warp as wp

from isaaclab.sensors.ray_caster.kernels import quat_yaw_only as _quat_yaw_only_func
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, random_orientation, yaw_quat
from isaaclab.utils.warp.kernels import raycast_mesh_masked_kernel as _raycast_mesh_masked_kernel
from isaaclab.utils.warp.ops import convert_to_warp_mesh, raycast_dynamic_meshes, raycast_mesh


@pytest.fixture(scope="module")
def raycast_setup():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Base trimesh cube and its Warp conversion
    trimesh_mesh = trimesh.creation.box([2, 2, 1])
    single_mesh = [
        convert_to_warp_mesh(
            trimesh_mesh.vertices,
            trimesh_mesh.faces,
            device,
        )
    ]
    single_mesh_id = single_mesh[0].id

    # Rays
    ray_starts = torch.tensor([[0, -0.35, -5], [0.25, 0.35, -5]], dtype=torch.float32, device=device).unsqueeze(0)
    ray_directions = torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0)
    expected_ray_hits = torch.tensor(
        [[0, -0.35, -0.5], [0.25, 0.35, -0.5]], dtype=torch.float32, device=device
    ).unsqueeze(0)

    return {
        "device": device,
        "trimesh_mesh": trimesh_mesh,
        "single_mesh_id": single_mesh_id,
        "wp_mesh": single_mesh[0],
        "ray_starts": ray_starts,
        "ray_directions": ray_directions,
        "expected_ray_hits": expected_ray_hits,
    }


def test_raycast_multi_cubes(raycast_setup):
    device = raycast_setup["device"]
    base_tm = raycast_setup["trimesh_mesh"]

    tm1 = base_tm.copy()
    wp_mesh_1 = convert_to_warp_mesh(tm1.vertices, tm1.faces, device)

    translation = np.eye(4)
    translation[:3, 3] = [0, 2, 0]
    tm2 = base_tm.copy().apply_transform(translation)
    wp_mesh_2 = convert_to_warp_mesh(tm2.vertices, tm2.faces, device)

    mesh_ids_wp = wp.array2d([[wp_mesh_1.id, wp_mesh_2.id]], dtype=wp.uint64, device=device)

    ray_directions = raycast_setup["ray_directions"]

    # Case 1
    ray_start = torch.tensor([[0, 0, -5], [0, 2.5, -5]], dtype=torch.float32, device=device).unsqueeze(0)
    ray_hits, ray_distance, ray_normal, ray_face_id, mesh_ids = raycast_dynamic_meshes(
        ray_start,
        ray_directions,
        mesh_ids_wp,
        return_distance=True,
        return_normal=True,
        return_face_id=True,
        return_mesh_id=True,
    )

    torch.testing.assert_close(ray_hits, torch.tensor([[[0, 0, -0.5], [0, 2.5, -0.5]]], device=device))
    torch.testing.assert_close(ray_distance, torch.tensor([[4.5, 4.5]], device=device))
    torch.testing.assert_close(ray_normal, torch.tensor([[[0, 0, -1], [0, 0, -1]]], device=device, dtype=torch.float32))
    assert torch.equal(mesh_ids, torch.tensor([[0, 1]], dtype=torch.int32, device=device))

    # Case 2 (explicit poses/orientations)
    ray_start = torch.tensor([[0, 0, -5], [0, 4.5, -5]], dtype=torch.float32, device=device).unsqueeze(0)
    ray_hits, ray_distance, ray_normal, ray_face_id, mesh_ids = raycast_dynamic_meshes(
        ray_start,
        ray_directions,
        mesh_ids_wp,
        return_distance=True,
        return_normal=True,
        return_face_id=True,
        mesh_positions_w=torch.tensor([[[0, 0, 0], [0, 2, 0]]], dtype=torch.float32, device=device),
        mesh_orientations_w=torch.tensor([[[0, 0, 0, 1], [0, 0, 0, 1]]], dtype=torch.float32, device=device),
        return_mesh_id=True,
    )

    torch.testing.assert_close(ray_hits, torch.tensor([[[0, 0, -0.5], [0, 4.5, -0.5]]], device=device))
    torch.testing.assert_close(ray_distance, torch.tensor([[4.5, 4.5]], device=device))
    torch.testing.assert_close(ray_normal, torch.tensor([[[0, 0, -1], [0, 0, -1]]], device=device, dtype=torch.float32))
    assert torch.equal(mesh_ids, torch.tensor([[0, 1]], dtype=torch.int32, device=device))


def test_raycast_single_cube(raycast_setup):
    device = raycast_setup["device"]
    ray_starts = raycast_setup["ray_starts"]
    ray_directions = raycast_setup["ray_directions"]
    mesh = raycast_setup["wp_mesh"]
    expected_ray_hits = raycast_setup["expected_ray_hits"]
    single_mesh_id = raycast_setup["single_mesh_id"]

    # Single-mesh helper
    ray_hits, ray_distance, ray_normal, ray_face_id = raycast_mesh(
        ray_starts,
        ray_directions,
        mesh,
        return_distance=True,
        return_normal=True,
        return_face_id=True,
    )
    torch.testing.assert_close(ray_hits, expected_ray_hits)
    torch.testing.assert_close(ray_distance, torch.tensor([[4.5, 4.5]], device=device))
    torch.testing.assert_close(ray_normal, torch.tensor([[[0, 0, -1], [0, 0, -1]]], device=device, dtype=torch.float32))
    torch.testing.assert_close(ray_face_id, torch.tensor([[3, 8]], dtype=torch.int32, device=device))

    # Multi-mesh API with one mesh
    ray_hits, ray_distance, ray_normal, ray_face_id, _ = raycast_dynamic_meshes(
        ray_starts,
        ray_directions,
        wp.array2d([[single_mesh_id]], dtype=wp.uint64, device=device),
        return_distance=True,
        return_normal=True,
        return_face_id=True,
    )
    torch.testing.assert_close(ray_hits, expected_ray_hits)
    torch.testing.assert_close(ray_distance, torch.tensor([[4.5, 4.5]], device=device))
    torch.testing.assert_close(ray_normal, torch.tensor([[[0, 0, -1], [0, 0, -1]]], device=device, dtype=torch.float32))
    torch.testing.assert_close(ray_face_id, torch.tensor([[3, 8]], dtype=torch.int32, device=device))


def test_raycast_moving_cube(raycast_setup):
    device = raycast_setup["device"]
    ray_starts = raycast_setup["ray_starts"]
    ray_directions = raycast_setup["ray_directions"]
    single_mesh_id = raycast_setup["single_mesh_id"]
    expected_ray_hits = raycast_setup["expected_ray_hits"]

    for distance in torch.linspace(0, 1, 10, device=device):
        ray_hits, ray_distance, ray_normal, ray_face_id, mesh_id = raycast_dynamic_meshes(
            ray_starts,
            ray_directions,
            wp.array2d([[single_mesh_id]], dtype=wp.uint64, device=device),
            return_distance=True,
            return_normal=True,
            return_face_id=True,
            return_mesh_id=True,
            mesh_positions_w=torch.tensor([[0, 0, distance.item()]], dtype=torch.float32, device=device),
        )
        offset = torch.tensor([[0, 0, distance.item()], [0, 0, distance.item()]], dtype=torch.float32, device=device)
        torch.testing.assert_close(ray_hits, expected_ray_hits + offset.unsqueeze(0))
        torch.testing.assert_close(ray_distance, distance + torch.tensor([[4.5, 4.5]], device=device))
        torch.testing.assert_close(
            ray_normal, torch.tensor([[[0, 0, -1], [0, 0, -1]]], device=device, dtype=torch.float32)
        )
        torch.testing.assert_close(ray_face_id, torch.tensor([[3, 8]], dtype=torch.int32, device=device))


def test_raycast_rotated_cube(raycast_setup):
    device = raycast_setup["device"]
    ray_starts = raycast_setup["ray_starts"]
    ray_directions = raycast_setup["ray_directions"]
    single_mesh_id = raycast_setup["single_mesh_id"]
    expected_ray_hits = raycast_setup["expected_ray_hits"]

    cube_rotation = quat_from_euler_xyz(
        torch.tensor([0.0], device=device), torch.tensor([0.0], device=device), torch.tensor([np.pi], device=device)
    )
    ray_hits, ray_distance, ray_normal, ray_face_id, _ = raycast_dynamic_meshes(
        ray_starts,
        ray_directions,
        wp.array2d([[single_mesh_id]], dtype=wp.uint64, device=device),
        return_distance=True,
        return_normal=True,
        return_face_id=True,
        mesh_orientations_w=cube_rotation.unsqueeze(0),
    )
    torch.testing.assert_close(ray_hits, expected_ray_hits)
    torch.testing.assert_close(ray_distance, torch.tensor([[4.5, 4.5]], device=device))
    torch.testing.assert_close(ray_normal, torch.tensor([[[0, 0, -1], [0, 0, -1]]], device=device, dtype=torch.float32))
    # Rotated cube swaps face IDs
    torch.testing.assert_close(ray_face_id, torch.tensor([[8, 3]], dtype=torch.int32, device=device))


def test_raycast_random_cube(raycast_setup):
    device = raycast_setup["device"]
    base_tm = raycast_setup["trimesh_mesh"]
    ray_starts = raycast_setup["ray_starts"]
    ray_directions = raycast_setup["ray_directions"]
    single_mesh_id = raycast_setup["single_mesh_id"]

    for orientation in random_orientation(10, device):
        pos = torch.tensor([[0.0, 0.0, torch.rand(1, device=device).item()]], dtype=torch.float32, device=device)

        tf_hom = np.eye(4)
        tf_hom[:3, :3] = matrix_from_quat(orientation).cpu().numpy()
        tf_hom[:3, 3] = pos.squeeze(0).cpu().numpy()

        tf_mesh = base_tm.copy().apply_transform(tf_hom)
        wp_mesh = convert_to_warp_mesh(tf_mesh.vertices, tf_mesh.faces, device)

        # Raycast transformed, static mesh
        ray_hits, ray_distance, ray_normal, ray_face_id, _ = raycast_dynamic_meshes(
            ray_starts,
            ray_directions,
            wp.array2d([[wp_mesh.id]], dtype=wp.uint64, device=device),
            return_distance=True,
            return_normal=True,
            return_face_id=True,
        )
        # Raycast original mesh with pose provided
        ray_hits_m, ray_distance_m, ray_normal_m, ray_face_id_m, _ = raycast_dynamic_meshes(
            ray_starts,
            ray_directions,
            wp.array2d([[single_mesh_id]], dtype=wp.uint64, device=device),
            return_distance=True,
            return_normal=True,
            return_face_id=True,
            mesh_positions_w=pos,
            mesh_orientations_w=orientation.view(1, 1, -1),
        )

        torch.testing.assert_close(ray_hits, ray_hits_m)
        torch.testing.assert_close(ray_distance, ray_distance_m)
        torch.testing.assert_close(ray_normal, ray_normal_m)
        torch.testing.assert_close(ray_face_id, ray_face_id_m)


# ---------------------------------------------------------------------------
# Tests for raycast_mesh_masked_kernel (new kernel in utils/warp/kernels.py)
# ---------------------------------------------------------------------------

_SENTINEL = -2.0  # value pre-filled into output buffers; chosen outside [-1, 1] so it cannot
# equal any component of a unit-length surface normal, making "not written" assertions unambiguous.


def _make_masked_buffers(device, n_envs, n_rays):
    """Allocate all warp buffers needed by raycast_mesh_masked_kernel.

    ray_dist_w and ray_normal_w are pre-filled with _SENTINEL so that tests can
    meaningfully assert those buffers were *not* written when the corresponding
    return flag is 0.
    """
    ray_starts_w = wp.zeros((n_envs, n_rays), dtype=wp.vec3f, device=device)
    ray_dirs_w = wp.zeros((n_envs, n_rays), dtype=wp.vec3f, device=device)
    ray_hits_w = wp.zeros((n_envs, n_rays), dtype=wp.vec3f, device=device)
    ray_dist_w = wp.zeros((n_envs, n_rays), dtype=wp.float32, device=device)
    wp.to_torch(ray_dist_w).fill_(_SENTINEL)
    ray_normal_w = wp.zeros((n_envs, n_rays), dtype=wp.vec3f, device=device)
    wp.to_torch(ray_normal_w).fill_(_SENTINEL)
    return ray_starts_w, ray_dirs_w, ray_hits_w, ray_dist_w, ray_normal_w


def test_raycast_mesh_masked_kernel_hits_only(raycast_setup):
    """return_distance=0, return_normal=0: only ray_hits are written on a hit."""
    device = raycast_setup["device"]
    mesh_id = raycast_setup["single_mesh_id"]
    expected_hits = raycast_setup["expected_ray_hits"]  # shape (1, 2, 3)

    n_envs, n_rays = 1, 2
    ray_starts_w, ray_dirs_w, ray_hits_w, ray_dist_w, ray_normal_w = _make_masked_buffers(device, n_envs, n_rays)
    env_mask = wp.array([True], dtype=wp.bool, device=device)

    wp.to_torch(ray_starts_w)[:] = torch.tensor([[[0, -0.35, -5], [0.25, 0.35, -5]]], device=device)
    wp.to_torch(ray_dirs_w)[:] = torch.tensor([[[0, 0, 1], [0, 0, 1]]], device=device)
    wp.to_torch(ray_hits_w).fill_(float("inf"))

    wp.launch(
        _raycast_mesh_masked_kernel,
        dim=(n_envs, n_rays),
        inputs=[mesh_id, env_mask, ray_starts_w, ray_dirs_w, float(1e6), 0, 0, ray_hits_w, ray_dist_w, ray_normal_w],
        device=device,
    )

    torch.testing.assert_close(wp.to_torch(ray_hits_w), expected_hits)
    assert torch.all(wp.to_torch(ray_dist_w) == _SENTINEL), "Distance buffer must not be written when return_distance=0"
    assert torch.all(wp.to_torch(ray_normal_w) == _SENTINEL), "Normal buffer must not be written when return_normal=0"


def test_raycast_mesh_masked_kernel_with_distance(raycast_setup):
    """return_distance=1: distances are written in addition to hits."""
    device = raycast_setup["device"]
    mesh_id = raycast_setup["single_mesh_id"]

    n_envs, n_rays = 1, 2
    ray_starts_w, ray_dirs_w, ray_hits_w, ray_dist_w, ray_normal_w = _make_masked_buffers(device, n_envs, n_rays)
    env_mask = wp.array([True], dtype=wp.bool, device=device)

    wp.to_torch(ray_starts_w)[:] = torch.tensor([[[0, -0.35, -5], [0.25, 0.35, -5]]], device=device)
    wp.to_torch(ray_dirs_w)[:] = torch.tensor([[[0, 0, 1], [0, 0, 1]]], device=device)
    wp.to_torch(ray_hits_w).fill_(float("inf"))

    wp.launch(
        _raycast_mesh_masked_kernel,
        dim=(n_envs, n_rays),
        inputs=[mesh_id, env_mask, ray_starts_w, ray_dirs_w, float(1e6), 1, 0, ray_hits_w, ray_dist_w, ray_normal_w],
        device=device,
    )

    # Cube bottom at z=-0.5, rays start at z=-5 going +z, distance = 4.5
    torch.testing.assert_close(wp.to_torch(ray_dist_w), torch.tensor([[4.5, 4.5]], device=device))
    assert torch.all(wp.to_torch(ray_normal_w) == _SENTINEL), "Normal buffer must not be written when return_normal=0"


def test_raycast_mesh_masked_kernel_with_normal(raycast_setup):
    """return_distance=1, return_normal=1: both distances and surface normals are written."""
    device = raycast_setup["device"]
    mesh_id = raycast_setup["single_mesh_id"]

    n_envs, n_rays = 1, 2
    ray_starts_w, ray_dirs_w, ray_hits_w, ray_dist_w, ray_normal_w = _make_masked_buffers(device, n_envs, n_rays)
    env_mask = wp.array([True], dtype=wp.bool, device=device)

    wp.to_torch(ray_starts_w)[:] = torch.tensor([[[0, -0.35, -5], [0.25, 0.35, -5]]], device=device)
    wp.to_torch(ray_dirs_w)[:] = torch.tensor([[[0, 0, 1], [0, 0, 1]]], device=device)
    wp.to_torch(ray_hits_w).fill_(float("inf"))

    wp.launch(
        _raycast_mesh_masked_kernel,
        dim=(n_envs, n_rays),
        inputs=[mesh_id, env_mask, ray_starts_w, ray_dirs_w, float(1e6), 1, 1, ray_hits_w, ray_dist_w, ray_normal_w],
        device=device,
    )

    # Cube bottom at z=-0.5, rays start at z=-5, distance = 4.5
    torch.testing.assert_close(wp.to_torch(ray_dist_w), torch.tensor([[4.5, 4.5]], device=device))
    torch.testing.assert_close(
        wp.to_torch(ray_normal_w),
        torch.tensor([[[0, 0, -1], [0, 0, -1]]], device=device, dtype=torch.float32),
    )


def test_raycast_mesh_masked_kernel_env_mask(raycast_setup):
    """Masked-out environments must not be written."""
    device = raycast_setup["device"]
    mesh_id = raycast_setup["single_mesh_id"]

    n_envs, n_rays = 2, 2
    ray_starts_w, ray_dirs_w, ray_hits_w, ray_dist_w, ray_normal_w = _make_masked_buffers(device, n_envs, n_rays)
    env_mask = wp.array([True, False], dtype=wp.bool, device=device)

    starts = torch.tensor([[[0, -0.35, -5], [0.25, 0.35, -5]], [[0, -0.35, -5], [0.25, 0.35, -5]]], device=device)
    dirs = torch.tensor([[[0, 0, 1], [0, 0, 1]], [[0, 0, 1], [0, 0, 1]]], device=device)
    wp.to_torch(ray_starts_w)[:] = starts
    wp.to_torch(ray_dirs_w)[:] = dirs
    wp.to_torch(ray_hits_w).fill_(float("inf"))

    wp.launch(
        _raycast_mesh_masked_kernel,
        dim=(n_envs, n_rays),
        inputs=[mesh_id, env_mask, ray_starts_w, ray_dirs_w, float(1e6), 1, 0, ray_hits_w, ray_dist_w, ray_normal_w],
        device=device,
    )

    hits = wp.to_torch(ray_hits_w)
    dist = wp.to_torch(ray_dist_w)

    assert not torch.isinf(hits[0]).any(), "Active env 0 should have valid hits"
    torch.testing.assert_close(dist[0], torch.tensor([4.5, 4.5], device=device))
    assert torch.isinf(hits[1]).all(), "Masked env 1 hits must remain inf"
    assert torch.all(dist[1] == _SENTINEL), "Masked env 1 distances must remain at sentinel"
    assert torch.all(wp.to_torch(ray_normal_w) == _SENTINEL), "Normal buffer must not be written when return_normal=0"


# ---------------------------------------------------------------------------
# Test quat_yaw_only correctness (regression for atan2-based fix)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _call_quat_yaw_only(q_in: wp.array(dtype=wp.quatf), q_out: wp.array(dtype=wp.quatf)):
    i = wp.tid()
    q_out[i] = _quat_yaw_only_func(q_in[i])


def test_quat_yaw_only_pure_yaw():
    """Pure yaw: quat_yaw_only should match the yaw_quat() reference for all yaw angles."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yaw_angles = torch.tensor([0.0, 0.5, 1.2, -0.8, np.pi], device=device)

    for yaw in yaw_angles:
        q_torch = quat_from_euler_xyz(
            torch.tensor([0.0], device=device),
            torch.tensor([0.0], device=device),
            yaw.unsqueeze(0),
        )  # shape (1, 4), xyzw

        expected = yaw_quat(q_torch)  # shape (1, 4)

        q_in = wp.from_torch(q_torch.contiguous(), dtype=wp.quatf)
        q_out = wp.zeros(1, dtype=wp.quatf, device=device)
        wp.launch(_call_quat_yaw_only, dim=1, inputs=[q_in, q_out], device=device)
        result = wp.to_torch(q_out)  # shape (1, 4)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


def test_quat_yaw_only_with_pitch_roll():
    """Non-zero pitch and roll: only the yaw component should be preserved.

    This is the regression test for the old bug where simply zeroing qx/qy and
    renormalizing gave the wrong answer when pitch or roll was non-zero.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Several combined pitch+roll+yaw orientations: (roll, pitch, yaw)
    test_cases = [
        (0.3, 0.4, 1.2),
        (0.5, 0.0, 0.7),
        (-0.2, 0.6, -1.0),
        (1.0, 1.0, 0.0),  # heavy pitch+roll, zero yaw → result should be identity
    ]

    for roll, pitch, yaw in test_cases:
        q_torch = quat_from_euler_xyz(
            torch.tensor([roll], device=device),
            torch.tensor([pitch], device=device),
            torch.tensor([yaw], device=device),
        )  # shape (1, 4), xyzw

        expected = yaw_quat(q_torch)  # shape (1, 4)

        q_in = wp.from_torch(q_torch.contiguous(), dtype=wp.quatf)
        q_out = wp.zeros(1, dtype=wp.quatf, device=device)
        wp.launch(_call_quat_yaw_only, dim=1, inputs=[q_in, q_out], device=device)
        result = wp.to_torch(q_out)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
