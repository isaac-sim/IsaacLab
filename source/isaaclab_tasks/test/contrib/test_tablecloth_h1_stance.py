# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical regression for the H1 tablecloth expert's stance and grasp."""

import sys

import gymnasium as gym
import pytest
import torch
import trimesh

from pxr import Gf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.app import launch_simulation
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_mul, subtract_frame_transforms

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.tablecloth.h1_env_cfg import HAND_OFFSETS
from isaaclab_tasks.utils import parse_env_cfg


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_h1_stays_grounded_and_retains_cloth(device, monkeypatch):
    """The scripted trick retains both physical grips without destabilizing the robot."""
    monkeypatch.setattr(sys, "argv", ["tablecloth_h1.py"])
    from scripts.environments.state_machine import tablecloth_h1 as expert

    cfg = parse_env_cfg("IsaacContrib-Tablecloth-H1", device=device, num_envs=1)
    cfg.sim.visualizer_cfgs = []
    success_term = cfg.terminations.success
    cfg.terminations.success = None
    cfg.terminations.tableware_fallen = None
    cfg.rewards.success = None
    with launch_simulation(cfg=cfg):
        env = gym.make("IsaacContrib-Tablecloth-H1", cfg=cfg).unwrapped
        try:
            env.reset()
            robot = env.scene["robot"]
            pelvis_ids, _ = robot.find_bodies("pelvis")
            foot_ids, foot_names = robot.find_bodies(".*ankle_link")
            leg_ids, _ = robot.find_joints(".*(hip|knee|ankle).*", preserve_order=True)
            initial_pelvis = robot.data.body_pos_w.torch[:, pelvis_ids].clone()
            initial_legs = robot.data.joint_pos.torch[:, leg_ids].clone()

            bbox_cache = UsdGeom.BBoxCache(0, ["default", "render", "proxy"])
            for foot_id, foot_name in zip(foot_ids, foot_names, strict=True):
                foot_prims = sim_utils.get_all_matching_child_prims(
                    "/World/envs/env_0/Robot",
                    predicate=lambda prim: prim.GetName() == foot_name and prim.HasAPI(UsdPhysics.RigidBodyAPI),
                    stage=env.sim.stage,
                )
                assert len(foot_prims) == 1
                bounds = bbox_cache.ComputeUntransformedBound(foot_prims[0]).ComputeAlignedRange()
                corners = torch.tensor([tuple(bounds.GetCorner(i)) for i in range(8)], device=device)
                corners_w = quat_apply(robot.data.body_quat_w.torch[0, foot_id].expand(8, -1), corners)
                corners_w += robot.data.body_pos_w.torch[0, foot_id]
                assert 0.0 <= corners_w[:, 2].min().item() <= 0.002, "H1's feet must start at ground level"

            body_ids, _ = robot.find_bodies(["left_hand_link", "right_hand_link", "torso_link"], preserve_order=True)
            positions = robot.data.body_pos_w.torch[:, body_ids].clone()
            rotations = robot.data.body_quat_w.torch[:, body_ids]
            initial_torso_position = positions[:, 2].clone()
            up_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device)
            initial_torso_up = quat_apply(rotations[:, 2], up_axis)
            offsets = torch.tensor([*HAND_OFFSETS, (0.0, 0.0, 0.0)], device=device)
            positions += quat_apply(rotations, offsets.unsqueeze(0))
            positions, rotations = subtract_frame_transforms(
                robot.data.root_pos_w.torch[:, None].expand_as(positions),
                robot.data.root_quat_w.torch[:, None].expand_as(rotations),
                positions,
                rotations,
            )
            torso_pose = torch.cat((positions[:, 2], rotations[:, 2]), dim=-1)
            machine = expert.H1TableclothStateMachine(env.step_dt, torso_pose, env.action_manager.total_action_dim, 2.0)
            closed_thumbs = torch.tensor(
                [expert.FINGER_CLOSED_VALUES[0], expert.FINGER_CLOSED_VALUES[12]], device=device
            )
            grasp_pose = torch.tensor(expert.HAND_KEYFRAMES[-2:], device=device)
            cloth_mesh = UsdGeom.Mesh(env.sim.stage.GetPrimAtPath("/World/envs/env_0/Cloth/sim_mesh"))
            assert all(count == 3 for count in cloth_mesh.GetFaceVertexCountsAttr().Get())
            cloth_faces = torch.tensor(cloth_mesh.GetFaceVertexIndicesAttr().Get(), device=device).reshape(-1, 3)
            mesh_transform = UsdGeom.XformCache().GetLocalToWorldTransform(cloth_mesh.GetPrim())
            rest_points = torch.tensor(
                [tuple(mesh_transform.Transform(Gf.Vec3d(*point))) for point in cloth_mesh.GetPointsAttr().Get()],
                device=device,
            )
            torch.testing.assert_close(
                rest_points, env.scene["cloth"].data.default_nodal_state_w.torch[0, :, :3], atol=1.0e-6, rtol=0.0
            )
            grasped_vertices = None
            for _ in range(312):
                with torch.inference_mode():
                    actions = machine.compute()
                    env.step(actions)
                torch.testing.assert_close(
                    robot.data.body_pos_w.torch[:, pelvis_ids], initial_pelvis, atol=1.0e-6, rtol=0.0
                )
                torch.testing.assert_close(robot.data.joint_pos.torch[:, leg_ids], initial_legs, atol=0.005, rtol=0.0)
                torch.testing.assert_close(
                    robot.data.body_pos_w.torch[:, body_ids[2]], initial_torso_position, atol=0.001, rtol=0.0
                )
                torch.testing.assert_close(
                    quat_apply(robot.data.body_quat_w.torch[:, body_ids[2]], up_axis),
                    initial_torso_up,
                    atol=0.002,
                    rtol=0.0,
                )

                hand_positions = robot.data.body_pos_w.torch[0, body_ids[:2]] + quat_apply(
                    robot.data.body_quat_w.torch[0, body_ids[:2]], offsets[:2]
                )
                cloth_positions = env.scene["cloth"].data.nodal_pos_w.torch[0]
                thumb_targets = actions[0, [expert.ARM_ACTION_DIM, expert.ARM_ACTION_DIM + 12]]
                hand_actions = actions[0, :14].reshape(2, 7)
                hand_targets = hand_actions[:, :3]
                if (
                    grasped_vertices is None
                    and torch.allclose(thumb_targets, closed_thumbs)
                    and torch.allclose(hand_targets, grasp_pose)
                ):
                    surface = trimesh.Trimesh(
                        vertices=cloth_positions.cpu().numpy(), faces=cloth_faces.cpu().numpy(), process=False
                    )
                    points, _, face_ids = trimesh.proximity.closest_point_naive(surface, hand_positions.cpu().numpy())
                    weights = trimesh.triangles.points_to_barycentric(surface.triangles[face_ids], points)
                    # Keep fixed material coordinates, not nearby vertices that depend on mesh resolution.
                    grasped_vertices = cloth_faces[torch.as_tensor(face_ids, device=device)]
                    grasp_weights = torch.as_tensor(weights, dtype=cloth_positions.dtype, device=device)
                    assert torch.isfinite(grasp_weights).all()
                    assert torch.all(
                        torch.linalg.vector_norm(torch.as_tensor(points, device=device) - hand_positions, dim=-1)
                        < 0.005
                    )
                if grasped_vertices is not None:
                    grasp_points = (cloth_positions[grasped_vertices] * grasp_weights[..., None]).sum(dim=1)
                    separation = (grasp_points - hand_positions).norm(dim=-1)
                    assert torch.all(separation < 0.015), f"H1 lost its cloth grasp: {separation.tolist()} m"
                    desired_rotations = quat_mul(robot.data.root_quat_w.torch.expand(2, -1), hand_actions[:, 3:])
                    attitude_error = torch.rad2deg(
                        quat_error_magnitude(robot.data.body_quat_w.torch[0, body_ids[:2]], desired_rotations)
                    )
                    assert torch.all(attitude_error < 5.0), f"H1's wrists drifted: {attitude_error.tolist()} deg"

            assert grasped_vertices is not None, "The expert never closed its grasp"
            assert success_term.func(env, **success_term.params).all(), "The tablecloth trick failed"
        finally:
            env.close()
