# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from isaaclab.app import AppLauncher

# launch omniverse app. Used for warp.
app_launcher = AppLauncher(headless=True)

import numpy as np
import pytest
import torch
import trimesh
import warp as wp

from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, random_orientation
from isaaclab.utils.warp.ops import convert_to_warp_mesh, raycast_dynamic_meshes, raycast_single_mesh


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def rays(device):
    ray_starts = torch.tensor([[0, -0.35, -5], [0.25, 0.35, -5]], dtype=torch.float32, device=device).unsqueeze(0)
    ray_directions = torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0)
    expected_ray_hits = torch.tensor(
        [[0, -0.35, -0.5], [0.25, 0.35, -0.5]], dtype=torch.float32, device=device
    ).unsqueeze(0)
    return ray_starts, ray_directions, expected_ray_hits


@pytest.fixture
def trimesh_box():
    return trimesh.creation.box([2, 2, 1])


@pytest.fixture
def single_mesh(trimesh_box, device):
    wp_mesh = convert_to_warp_mesh(trimesh_box.vertices, trimesh_box.faces, device)
    return wp_mesh, wp_mesh.id


def test_raycast_multi_cubes(device, trimesh_box, rays):
    """Test raycasting against two cubes."""
    ray_starts, ray_directions, _ = rays

    trimesh_1 = trimesh_box.copy()
    wp_mesh_1 = convert_to_warp_mesh(trimesh_1.vertices, trimesh_1.faces, device)

    translation = np.eye(4)
    translation[:3, 3] = [0, 2, 0]
    trimesh_2 = trimesh_box.copy().apply_transform(translation)
    wp_mesh_2 = convert_to_warp_mesh(trimesh_2.vertices, trimesh_2.faces, device)

    # get mesh id array
    mesh_ids_wp = wp.array2d([[wp_mesh_1.id, wp_mesh_2.id]], dtype=wp.uint64, device=device)

    # Static positions (no transforms passed)
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

    torch.testing.assert_close(
        ray_hits, torch.tensor([[[0, 0, -0.5], [0, 2.5, -0.5]]], dtype=torch.float32, device=device)
    )
    torch.testing.assert_close(ray_distance, torch.tensor([[4.5, 4.5]], dtype=torch.float32, device=device))
    torch.testing.assert_close(ray_normal, torch.tensor([[[0, 0, -1], [0, 0, -1]]], dtype=torch.float32, device=device))
    assert torch.equal(mesh_ids, torch.tensor([[0, 1]], dtype=torch.int32, device=device))

    # Dynamic positions/orientations
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

    torch.testing.assert_close(
        ray_hits, torch.tensor([[[0, 0, -0.5], [0, 4.5, -0.5]]], dtype=torch.float32, device=device)
    )
    torch.testing.assert_close(ray_distance, torch.tensor([[4.5, 4.5]], dtype=torch.float32, device=device))
    torch.testing.assert_close(ray_normal, torch.tensor([[[0, 0, -1], [0, 0, -1]]], dtype=torch.float32, device=device))
    assert torch.equal(mesh_ids, torch.tensor([[0, 1]], dtype=torch.int32, device=device))


def test_raycast_single_cube(device, single_mesh, rays):
    """Test raycasting against a single cube."""
    ray_starts, ray_directions, expected_ray_hits = rays
    _, single_mesh_id = single_mesh

    ray_hits, ray_distance, ray_normal, ray_face_id = raycast_single_mesh(
        ray_starts,
        ray_directions,
        single_mesh_id,
        return_distance=True,
        return_normal=True,
        return_face_id=True,
    )
    torch.testing.assert_close(ray_hits, expected_ray_hits)
    torch.testing.assert_close(ray_distance, torch.tensor([[4.5, 4.5]], dtype=torch.float32, device=device))
    torch.testing.assert_close(
        ray_normal,
        torch.tensor([[[0, 0, -1], [0, 0, -1]]], dtype=torch.float32, device=device),
    )
    torch.testing.assert_close(ray_face_id, torch.tensor([[3, 8]], dtype=torch.int32, device=device))

    # check multiple meshes implementation
    ray_hits, ray_distance, ray_normal, ray_face_id, _ = raycast_dynamic_meshes(
        ray_starts,
        ray_directions,
        wp.array2d([[single_mesh_id]], dtype=wp.uint64, device=device),
        return_distance=True,
        return_normal=True,
        return_face_id=True,
    )
    torch.testing.assert_close(ray_hits, expected_ray_hits)
    torch.testing.assert_close(ray_distance, torch.tensor([[4.5, 4.5]], dtype=torch.float32, device=device))
    torch.testing.assert_close(ray_normal, torch.tensor([[[0, 0, -1], [0, 0, -1]]], dtype=torch.float32, device=device))
    torch.testing.assert_close(ray_face_id, torch.tensor([[3, 8]], dtype=torch.int32, device=device))


@pytest.mark.parametrize("num_samples", [10])
def test_raycast_moving_cube(device, single_mesh, rays, num_samples):
    r"""Test raycasting against a single cube with different distances.
    |-------------|
    |\            |
    | \           |
    |  \     8    |
    |   \         |
    |    \    x_1 |
    |     \       |
    |      \      |
    |       \     |
    |        \    |
    |         \   |
    |   3  x_2 \  |
    |           \ |
    |            \|
    |-------------|

    """
    ray_starts, ray_directions, expected_ray_hits = rays
    _, single_mesh_id = single_mesh

    # move the cube along the z axis
    for distance in torch.linspace(0, 1, num_samples, device=device):
        ray_hits, ray_distance, ray_normal, ray_face_id, mesh_id = raycast_dynamic_meshes(
            ray_starts,
            ray_directions,
            wp.array2d([[single_mesh_id]], dtype=wp.uint64, device=device),
            return_distance=True,
            return_normal=True,
            return_face_id=True,
            return_mesh_id=True,
            mesh_positions_w=torch.tensor([[0, 0, distance]], dtype=torch.float32, device=device),
        )
        torch.testing.assert_close(
            ray_hits,
            expected_ray_hits
            + torch.tensor([[0, 0, distance], [0, 0, distance]], dtype=torch.float32, device=device).unsqueeze(0),
        )
        torch.testing.assert_close(
            ray_distance, distance + torch.tensor([[4.5, 4.5]], dtype=torch.float32, device=device)
        )
        torch.testing.assert_close(
            ray_normal, torch.tensor([[[0, 0, -1], [0, 0, -1]]], dtype=torch.float32, device=device)
        )
        torch.testing.assert_close(ray_face_id, torch.tensor([[3, 8]], dtype=torch.int32, device=device))


def test_raycast_rotated_cube(device, single_mesh, rays):
    """Test raycasting against a single cube with different 90deg. orientations."""
    ray_starts, ray_directions, expected_ray_hits = rays
    _, single_mesh_id = single_mesh

    cube_rotation = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([np.pi])).to(device)
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
    torch.testing.assert_close(ray_distance, torch.tensor([[4.5, 4.5]], dtype=torch.float32, device=device))
    torch.testing.assert_close(ray_normal, torch.tensor([[[0, 0, -1], [0, 0, -1]]], dtype=torch.float32, device=device))
    # Make sure the face ids are correct. The cube is rotated by 90deg. so the face ids are different.
    torch.testing.assert_close(ray_face_id, torch.tensor([[8, 3]], dtype=torch.int32, device=device))


@pytest.mark.parametrize("num_random", [10])
def test_raycast_random_cube(device, trimesh_box, single_mesh, rays, num_random):
    """Test raycasting against a single cube with random poses."""
    ray_starts, ray_directions, _ = rays
    _, single_mesh_id = single_mesh

    for orientation in random_orientation(num_random, device):
        pos = torch.tensor([[0, 0, torch.rand(1)]], dtype=torch.float32, device=device)
        tf_hom = np.eye(4)
        tf_hom[:3, :3] = matrix_from_quat(orientation).cpu().numpy()
        tf_hom[:3, 3] = pos.cpu().numpy()
        tf_mesh = trimesh_box.copy().apply_transform(tf_hom)

        # get raycast for transformed, static mesh
        wp_mesh = convert_to_warp_mesh(tf_mesh.vertices, tf_mesh.faces, device)
        ray_hits, ray_distance, ray_normal, ray_face_id, _ = raycast_dynamic_meshes(
            ray_starts,
            ray_directions,
            wp.array2d([[wp_mesh.id]], dtype=wp.uint64, device=device),
            return_distance=True,
            return_normal=True,
            return_face_id=True,
        )
        # get raycast for modified mesh
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
