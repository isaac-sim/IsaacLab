# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Warp ray-casting operations used by the ray caster sensors.

The single-mesh and dynamic multi-mesh raycast helpers are exercised against a unit cube. Sensor-level
behavior is covered in ``test_ray_caster_sensor.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import trimesh
import warp as wp

from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, random_orientation
from isaaclab.utils.warp.ops import convert_to_warp_mesh, raycast_dynamic_meshes, raycast_mesh, raycast_single_mesh

pytestmark = pytest.mark.unit

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# two rays from z=-5 pointing +z at a 2x2x1 box centered at the origin: both hit the bottom face at z=-0.5
RAY_STARTS = torch.tensor([[[0.0, -0.35, -5.0], [0.25, 0.35, -5.0]]], device=DEVICE)
RAY_DIRECTIONS = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], device=DEVICE)
EXPECTED_HITS = torch.tensor([[[0.0, -0.35, -0.5], [0.25, 0.35, -0.5]]], device=DEVICE)
EXPECTED_DISTANCE = torch.tensor([[4.5, 4.5]], device=DEVICE)
EXPECTED_NORMAL = torch.tensor([[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]], device=DEVICE)
EXPECTED_FACE_ID = torch.tensor([[3, 8]], dtype=torch.int32, device=DEVICE)


@pytest.fixture(scope="module")
def box() -> trimesh.Trimesh:
    return trimesh.creation.box([2, 2, 1])


@pytest.fixture(scope="module")
def box_mesh(box) -> wp.Mesh:
    return convert_to_warp_mesh(box.vertices, box.faces, DEVICE)


def _mesh_ids(*meshes: wp.Mesh) -> wp.array:
    return wp.array2d([[mesh.id for mesh in meshes]], dtype=wp.uint64, device=DEVICE)


def _raycast(mesh_ids: wp.array, ray_starts=RAY_STARTS, **kwargs):
    return raycast_dynamic_meshes(
        ray_starts,
        RAY_DIRECTIONS,
        mesh_ids,
        return_distance=True,
        return_normal=True,
        return_face_id=True,
        return_mesh_id=True,
        **kwargs,
    )


def _assert_box_hit(hits, distance, normal, face_id, offset=None):
    expected_hits = EXPECTED_HITS if offset is None else EXPECTED_HITS + offset
    expected_distance = EXPECTED_DISTANCE if offset is None else EXPECTED_DISTANCE + offset[2]
    torch.testing.assert_close(hits, expected_hits)
    torch.testing.assert_close(distance, expected_distance)
    torch.testing.assert_close(normal, EXPECTED_NORMAL)
    torch.testing.assert_close(face_id, EXPECTED_FACE_ID)


def test_raycast_single_cube(box_mesh):
    """The single-mesh helpers and the multi-mesh helper agree on hits, distances, normals, and face ids."""
    _assert_box_hit(
        *raycast_mesh(
            RAY_STARTS, RAY_DIRECTIONS, box_mesh, return_distance=True, return_normal=True, return_face_id=True
        )
    )
    _assert_box_hit(
        *raycast_single_mesh(
            RAY_STARTS, RAY_DIRECTIONS, box_mesh.id, return_distance=True, return_normal=True, return_face_id=True
        )
    )
    hits, distance, normal, face_id, mesh_id = _raycast(_mesh_ids(box_mesh))
    _assert_box_hit(hits, distance, normal, face_id)
    assert torch.equal(mesh_id, torch.zeros((1, 2), dtype=torch.int32, device=DEVICE))


def test_raycast_multi_cubes(box, box_mesh):
    """Rays hitting two different cubes report the matching mesh ids, with and without explicit mesh poses."""
    translation = np.eye(4)
    translation[:3, 3] = [0, 2, 0]
    shifted = box.copy().apply_transform(translation)
    shifted_mesh = convert_to_warp_mesh(shifted.vertices, shifted.faces, DEVICE)
    mesh_ids = _mesh_ids(box_mesh, shifted_mesh)

    # baked translation: second ray at y=2.5 hits the shifted cube
    ray_starts = torch.tensor([[[0.0, 0.0, -5.0], [0.0, 2.5, -5.0]]], device=DEVICE)
    hits, distance, normal, _, mesh_id = _raycast(mesh_ids, ray_starts)
    torch.testing.assert_close(hits, torch.tensor([[[0.0, 0.0, -0.5], [0.0, 2.5, -0.5]]], device=DEVICE))
    torch.testing.assert_close(distance, EXPECTED_DISTANCE)
    torch.testing.assert_close(normal, EXPECTED_NORMAL)
    assert torch.equal(mesh_id, torch.tensor([[0, 1]], dtype=torch.int32, device=DEVICE))

    # explicit poses translate the shifted cube by another 2 m
    ray_starts = torch.tensor([[[0.0, 0.0, -5.0], [0.0, 4.5, -5.0]]], device=DEVICE)
    hits, distance, normal, _, mesh_id = _raycast(
        mesh_ids,
        ray_starts,
        mesh_positions_w=torch.tensor([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], device=DEVICE),
        mesh_orientations_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]], device=DEVICE),
    )
    torch.testing.assert_close(hits, torch.tensor([[[0.0, 0.0, -0.5], [0.0, 4.5, -0.5]]], device=DEVICE))
    torch.testing.assert_close(distance, EXPECTED_DISTANCE)
    torch.testing.assert_close(normal, EXPECTED_NORMAL)
    assert torch.equal(mesh_id, torch.tensor([[0, 1]], dtype=torch.int32, device=DEVICE))


def test_raycast_moved_and_rotated_cube(box_mesh):
    """Mesh poses translate the hits along the ray and a 180 deg yaw swaps the hit face ids."""
    for distance in torch.linspace(0, 1, 4, device=DEVICE):
        offset = torch.tensor([0.0, 0.0, distance], device=DEVICE)
        hits, ray_distance, normal, face_id, _ = _raycast(_mesh_ids(box_mesh), mesh_positions_w=offset.reshape(1, 1, 3))
        _assert_box_hit(hits, ray_distance, normal, face_id, offset=offset)

    yaw_180 = quat_from_euler_xyz(
        torch.tensor([0.0], device=DEVICE), torch.tensor([0.0], device=DEVICE), torch.tensor([np.pi], device=DEVICE)
    )
    hits, ray_distance, normal, face_id, _ = _raycast(_mesh_ids(box_mesh), mesh_orientations_w=yaw_180.unsqueeze(0))
    torch.testing.assert_close(hits, EXPECTED_HITS)
    torch.testing.assert_close(ray_distance, EXPECTED_DISTANCE)
    torch.testing.assert_close(normal, EXPECTED_NORMAL)
    torch.testing.assert_close(face_id, EXPECTED_FACE_ID.flip(1))


def test_raycast_random_pose_matches_baked_mesh(box, box_mesh):
    """Passing a mesh pose gives the same result as baking that pose into the mesh vertices."""
    for orientation in random_orientation(5, DEVICE):
        pos = torch.tensor([[0.0, 0.0, torch.rand(1).item()]], device=DEVICE)
        transform = np.eye(4)
        transform[:3, :3] = matrix_from_quat(orientation).cpu().numpy()
        transform[:3, 3] = pos.squeeze(0).cpu().numpy()
        baked = box.copy().apply_transform(transform)
        baked_mesh = convert_to_warp_mesh(baked.vertices, baked.faces, DEVICE)

        expected = _raycast(_mesh_ids(baked_mesh))
        actual = _raycast(_mesh_ids(box_mesh), mesh_positions_w=pos, mesh_orientations_w=orientation.view(1, 1, -1))
        for expected_out, actual_out in zip(expected[:4], actual[:4]):
            torch.testing.assert_close(expected_out, actual_out)
