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

from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, random_orientation
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
        mesh_orientations_w=torch.tensor([[[1, 0, 0, 0], [1, 0, 0, 0]]], dtype=torch.float32, device=device),
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
