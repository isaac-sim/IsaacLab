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


##
# RayCaster sensor-level tests
##


def test_raycaster_offset_does_not_affect_pos_w():
    """Verify that cfg.offset.pos shifts ray starts but NOT data.pos_w.

    data.pos_w must reflect the parent body position so that downstream
    observations like height_scan (pos_w_z - hit_z - 0.5) produce values
    relative to the body, not relative to the offset sensor frame.

    Regression test: previously the offset was baked into the FrameView's
    Xform local transform, causing data.pos_w to include the 20m offset
    and breaking height-scan observations during training.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
    from isaaclab.terrains.trimesh.utils import make_plane
    from isaaclab.terrains.utils import create_prim_from_mesh

    sim_utils.create_new_stage()

    # ground plane at z=0
    mesh = make_plane(size=(100, 100), height=0.0, center_zero=True)
    create_prim_from_mesh("/World/ground", mesh)

    # parent body at known position
    body_pos = (0.0, 0.0, 0.6)
    sim_utils.create_prim("/World/Robot", "Xform", translation=body_pos)

    # large z-offset to make the regression obvious
    offset_z = 20.0
    cfg = RayCasterCfg(
        prim_path="/World/Robot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, offset_z)),
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.5, size=[1.0, 1.0]),
        ray_alignment="yaw",
    )

    dt = 0.01
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt))

    sensor = RayCaster(cfg)
    sim.reset()
    sensor.update(dt)

    pos_w = sensor.data.pos_w[0].cpu()

    # pos_w.z should be near the body height, NOT body_height + offset
    assert abs(pos_w[2].item() - body_pos[2]) < 1.0, (
        f"data.pos_w.z = {pos_w[2].item():.2f}, expected near body height {body_pos[2]}."
        f" If pos_w.z ≈ {body_pos[2] + offset_z}, the offset was incorrectly baked into the FrameView."
    )

    # ray_hits should be near z=0 (ground plane)
    hits_z = sensor.data.ray_hits_w[0, :, 2].cpu()
    valid = hits_z[~torch.isinf(hits_z)]
    if len(valid) > 0:
        assert valid.abs().max().item() < 2.0, (
            f"Ray hits z range [{valid.min().item():.2f}, {valid.max().item():.2f}] — expected near ground (z≈0)."
        )

    # height_scan observation: pos_w_z - hit_z - 0.5 should be small, not ~20
    if len(valid) > 0:
        height_obs = pos_w[2].item() - valid.mean().item() - 0.5
        assert abs(height_obs) < 5.0, (
            f"height_scan observation = {height_obs:.2f}, expected near 0."
            f" If ≈{offset_z}, the offset leaked into data.pos_w."
        )

    sim.stop()
    sim.clear_instance()
