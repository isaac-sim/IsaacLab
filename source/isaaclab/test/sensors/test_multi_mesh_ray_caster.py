# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from omni.isaac.lab.app import AppLauncher

# # launch omniverse app. Used for warp.
app_launcher = AppLauncher(headless=True)

import numpy as np
import torch
import trimesh
import unittest

import warp as wp
from omni.isaac.lab.utils.math import matrix_from_quat, quat_from_euler_xyz, random_orientation
from omni.isaac.lab.utils.warp.ops import convert_to_warp_mesh, raycast_dynamic_meshes, raycast_single_mesh


class TestRaycast(unittest.TestCase):
    """Test fixture for the raycast ops."""

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Create static mesh to raycast against
        self._trimesh_meshes = [trimesh.creation.box([2, 2, 1])]
        self.single_mesh = [
            convert_to_warp_mesh(
                self._trimesh_meshes[0].vertices,
                self._trimesh_meshes[0].faces,
                self.device,
            )
        ]
        self.single_mesh_id = self.single_mesh[0].id

        self.ray_starts = torch.Tensor([[0, -0.35, -5], [0.25, 0.35, -5]]).to(self.device).unsqueeze(0)
        self.ray_directions = torch.Tensor([[0, 0, 1], [0, 0, 1]]).to(self.device).unsqueeze(0)
        self.expected_ray_hits = torch.Tensor([[0, -0.35, -0.5], [0.25, 0.35, -0.5]]).to(self.device).unsqueeze(0)

    def test_raycast_multi_cubes(self):
        """Test raycasting against two cubes."""
        trimesh_1 = self._trimesh_meshes[0].copy()
        wp_mesh_1 = convert_to_warp_mesh(trimesh_1.vertices, trimesh_1.faces, self.device)

        translation = np.eye(4)
        translation[:3, 3] = [0, 2, 0]
        trimesh_2 = self._trimesh_meshes[0].copy().apply_transform(translation)
        wp_mesh_2 = convert_to_warp_mesh(trimesh_2.vertices, trimesh_2.faces, self.device)

        # get mesh id array
        mesh_ids_wp = wp.array2d([[wp_mesh_1.id, wp_mesh_2.id]], dtype=wp.uint64, device=self.device)

        ray_start = torch.Tensor([[0, 0, -5], [0, 2.5, -5]]).unsqueeze(0).to(self.device)
        ray_hits, ray_distance, ray_normal, ray_face_id, mesh_ids = raycast_dynamic_meshes(
            ray_start,
            self.ray_directions,
            mesh_ids_wp,
            return_distance=True,
            return_normal=True,
            return_face_id=True,
            return_mesh_id=True,
        )

        torch.testing.assert_close(ray_hits, torch.Tensor([[[0, 0, -0.5], [0, 2.5, -0.5]]]).to(self.device))
        torch.testing.assert_close(ray_distance, torch.Tensor([[4.5, 4.5]]).to(self.device))
        torch.testing.assert_close(ray_normal, torch.Tensor([[[0, 0, -1], [0, 0, -1]]]).to(self.device))
        self.assertTrue(torch.equal(mesh_ids, torch.Tensor([[0, 1]]).to(self.device, dtype=torch.int32)))

        ray_start = torch.Tensor([[0, 0, -5], [0, 4.5, -5]]).unsqueeze(0).to(self.device)
        ray_hits, ray_distance, ray_normal, ray_face_id, mesh_ids = raycast_dynamic_meshes(
            ray_start,
            self.ray_directions,
            mesh_ids_wp,
            return_distance=True,
            return_normal=True,
            return_face_id=True,
            mesh_positions_w=torch.Tensor([[[0, 0, 0], [0, 2, 0]]]),
            mesh_orientations_w=torch.Tensor([[[1, 0, 0, 0], [1, 0, 0, 0]]]),
            return_mesh_id=True,
        )

        torch.testing.assert_close(ray_hits, torch.Tensor([[0, 0, -0.5], [0, 4.5, -0.5]]).unsqueeze(0).to(self.device))
        torch.testing.assert_close(ray_distance, torch.Tensor([[4.5, 4.5]]).to(self.device))
        torch.testing.assert_close(ray_normal, torch.Tensor([[0, 0, -1], [0, 0, -1]]).unsqueeze(0).to(self.device))
        self.assertTrue(torch.equal(mesh_ids, torch.Tensor([[0, 1]]).to(self.device, dtype=torch.int32)))

    def test_raycast_single_cube(self):
        """Test raycasting against a single cube."""
        # Convert meshes to warp formatay_hits.to(device).view(shape), ray_distance, ray_normal, ray_face_id
        ray_hits, ray_distance, ray_normal, ray_face_id = raycast_single_mesh(
            self.ray_starts,
            self.ray_directions,
            self.single_mesh_id,
            return_distance=True,
            return_normal=True,
            return_face_id=True,
        )
        torch.testing.assert_close(ray_hits, self.expected_ray_hits)
        torch.testing.assert_close(ray_distance, torch.Tensor([[4.5, 4.5]]).to(self.device))
        torch.testing.assert_close(ray_normal, torch.Tensor([[0, 0, -1], [0, 0, -1]]).to(self.device).unsqueeze(0))
        torch.testing.assert_close(ray_face_id, torch.Tensor([[3, 8]]).to(self.device, dtype=torch.int32))

        # check multiple meshes implementation
        ray_hits, ray_distance, ray_normal, ray_face_id, _ = raycast_dynamic_meshes(
            self.ray_starts,
            self.ray_directions,
            wp.array2d([[self.single_mesh_id]], dtype=wp.uint64, device=self.device),
            return_distance=True,
            return_normal=True,
            return_face_id=True,
        )
        torch.testing.assert_close(ray_hits, self.expected_ray_hits)
        torch.testing.assert_close(ray_distance, torch.Tensor([[4.5, 4.5]]).to(self.device))
        torch.testing.assert_close(ray_normal, torch.Tensor([[[0, 0, -1], [0, 0, -1]]]).to(self.device))
        torch.testing.assert_close(ray_face_id, torch.Tensor([[3, 8]]).to(self.device, dtype=torch.int32))

    def test_raycast_moving_cube(self):
        r"""Test raycasting against a single cube with different distances.

        The faces that are raycasted against are the bottom two, it will look similar to:


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
        # move the cube along the z axis
        for distance in torch.linspace(0, 1, 10):
            ray_hits, ray_distance, ray_normal, ray_face_id, mesh_id = raycast_dynamic_meshes(
                self.ray_starts,
                self.ray_directions,
                wp.array2d([[self.single_mesh_id]], dtype=wp.uint64, device=self.device),
                return_distance=True,
                return_normal=True,
                return_face_id=True,
                return_mesh_id=True,
                mesh_positions_w=torch.Tensor([[0, 0, distance]]).to(self.device),
            )
            torch.testing.assert_close(
                ray_hits,
                self.expected_ray_hits + torch.Tensor([[0, 0, distance], [0, 0, distance]]).to(self.device),
            )
            torch.testing.assert_close(ray_distance, distance + torch.Tensor([[4.5, 4.5]]).to(self.device))
            torch.testing.assert_close(ray_normal, torch.Tensor([[[0, 0, -1], [0, 0, -1]]]).to(self.device))
            torch.testing.assert_close(ray_face_id, torch.Tensor([[3, 8]]).to(self.device, dtype=torch.int32))

    def test_raycast_rotated_cube(self):
        """Test raycasting against a single cube with different 90deg. orientations."""
        cube_rotation = quat_from_euler_xyz(torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([np.pi]))
        ray_hits, ray_distance, ray_normal, ray_face_id, _ = raycast_dynamic_meshes(
            self.ray_starts,
            self.ray_directions,
            wp.array2d([[self.single_mesh_id]], dtype=wp.uint64, device=self.device),
            return_distance=True,
            return_normal=True,
            return_face_id=True,
            mesh_orientations_w=cube_rotation.unsqueeze(0),
        )
        torch.testing.assert_close(ray_hits, self.expected_ray_hits)
        torch.testing.assert_close(ray_distance, torch.Tensor([[4.5, 4.5]]).to(self.device))
        torch.testing.assert_close(ray_normal, torch.Tensor([[[0, 0, -1], [0, 0, -1]]]).to(self.device))
        # Make sure the face ids are correct. The cube is rotated by 90deg. so the face ids are different.
        torch.testing.assert_close(ray_face_id, torch.Tensor([[8, 3]]).to(self.device, dtype=torch.int32))

    def test_Raycast_random_cube(self):
        """Test raycasting against a single cube with random poses."""
        for orientation in random_orientation(10, self.device):
            pos = torch.Tensor([[0, 0, torch.rand(1)]]).to(self.device)
            tf_hom = np.eye(4)
            tf_hom[:3, :3] = matrix_from_quat(orientation).cpu().numpy()
            tf_hom[:3, 3] = pos.cpu().numpy()
            tf_mesh = self._trimesh_meshes[0].copy().apply_transform(tf_hom)

            # get raycast for transformed, static mesh
            wp_mesh = convert_to_warp_mesh(tf_mesh.vertices, tf_mesh.faces, self.device)
            ray_hits, ray_distance, ray_normal, ray_face_id, _ = raycast_dynamic_meshes(
                self.ray_starts,
                self.ray_directions,
                wp.array2d([[wp_mesh.id]], dtype=wp.uint64, device=self.device),
                return_distance=True,
                return_normal=True,
                return_face_id=True,
            )
            # get raycast for modified mesh
            ray_hits_m, ray_distance_m, ray_normal_m, ray_face_id_m, _ = raycast_dynamic_meshes(
                self.ray_starts,
                self.ray_directions,
                wp.array2d([[self.single_mesh_id]], dtype=wp.uint64, device=self.device),
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


if __name__ == "__main__":
    unittest.main()
