# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

newton = pytest.importorskip("newton")

from isaaclab_newton.assets.mpm_object import MPMObjectCfg
from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg, NewtonMPMManager
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg

from isaaclab.assets import RigidObjectCfg
from isaaclab.cloner.query import iter_sources
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils import configclass


def test_mpm_object_initializes_from_interactive_scene():
    @configclass
    class MPMSceneCfg(InteractiveSceneCfg):
        media = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(
                lower=(0.0, 0.0, 0.0),
                upper=(0.1, 0.1, 0.1),
                voxel_size=0.1,
                particle_placement="cell_center",
                visible=False,
            ),
        )

    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(MPMSceneCfg(num_envs=2, env_spacing=1.0))
        sim.reset()

        media = scene["media"]
        assert media.num_instances == 2
        assert media.particles_per_object == 1
        assert media.data.particle_pos_w.torch.shape == (2, 1, 3)
        assert not sim.get_scene_data_provider().get_geometry_points()

        default_state = media.data.default_particle_state_w.torch.clone()
        shifted_state = default_state[0:1].clone()
        shifted_state[..., 2] += 0.05

        media.write_particle_state_to_sim_index(
            shifted_state,
            env_ids=torch.tensor([0], device=sim.device, dtype=torch.int32),
        )
        torch.testing.assert_close(media.data.particle_state_w.torch[0:1], shifted_state)

        media.reset(env_ids=[0])
        torch.testing.assert_close(media.data.particle_state_w.torch[0], default_state[0])


def test_mpm_solver_refreshes_kinematic_rigid_body_transforms():
    from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg  # noqa: PLC0415

    import isaaclab.sim as sim_utils  # noqa: PLC0415

    @configclass
    class MPMSceneCfg(InteractiveSceneCfg):
        collider = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/KinematicBox",
            spawn=sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.1),
                rigid_props=[
                    sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
                    PhysxRigidBodyCfg(disable_gravity=True),
                ],
                collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
        )
        media = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(lower=(-0.05, -0.05, 0.3), upper=(0.05, 0.05, 0.4), voxel_size=0.05),
        )

    sim_cfg = SimulationCfg(
        dt=1.0 / 60.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(MPMSceneCfg(num_envs=1, env_spacing=0.0))
        sim.reset()

        collider = scene["collider"]
        angle = 0.5
        root_pose = torch.tensor(
            [[0.1, 0.0, 0.25, 0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle)]],
            dtype=torch.float32,
            device=collider.device,
        )
        collider.write_root_link_pose_to_sim_index(root_pose=root_pose)
        sim.step(render=False)

        body_labels = list(NewtonMPMManager.get_model().body_label)
        body_idx = body_labels.index("/World/envs/env_0/KinematicBox")
        body_q = NewtonMPMManager.get_state_0().body_q.numpy()[body_idx]

        np.testing.assert_allclose(body_q, root_pose.detach().cpu().numpy()[0], rtol=1.0e-5, atol=1.0e-6)


def test_mpm_object_publishes_points_without_kit_visualizer():
    @configclass
    class MPMSceneCfg(InteractiveSceneCfg):
        media = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sand",
            spawn=MPMGridCfg(
                lower=(0.0, 0.0, 0.0),
                upper=(0.1, 0.1, 0.1),
                voxel_size=0.1,
                visual_color=(0.1, 0.2, 0.3),
            ),
        )

    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(MPMSceneCfg(num_envs=2, env_spacing=1.0))

        from pxr import UsdGeom  # noqa: PLC0415

        sim.reset()

        media = scene["media"]
        provider = sim.get_scene_data_provider()
        publication = provider.get_geometry_points()
        expected_paths = [f"/World/envs/env_{env_idx}/Sand/Particles" for env_idx in range(media.num_instances)]
        assert list(publication) == expected_paths

        for _, _, source_path, _ in iter_sources(sim.get_clone_plan(), media.cfg.prim_path):
            points_prim = media.stage.GetPrimAtPath(source_path + "/Particles")
            assert points_prim.IsValid()
            points = UsdGeom.Points(points_prim)
            assert not points.GetResetXformStack()
            assert len(points.GetPointsAttr().Get()) == media.particles_per_object
            assert len(points.GetWidthsAttr().Get()) == media.particles_per_object
            assert tuple(points.GetDisplayColorAttr().Get()[0]) == pytest.approx((0.1, 0.2, 0.3))

        before = np.stack([values.numpy() for values in publication.values()])
        for _ in range(3):
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
            publication = provider.get_geometry_points()
        actual = np.stack([values.numpy() for values in publication.values()])
        expected = media.data.particle_pos_w.torch.cpu().numpy()
        assert np.any(actual != before)
        np.testing.assert_allclose(actual, expected, atol=1.0e-6)

        state = media.data.particle_state_w.torch.clone()
        state[..., 2] += 0.05
        media.write_particle_state_to_sim_index(state)
        actual = np.stack([values.numpy() for values in provider.get_geometry_points().values()])
        np.testing.assert_allclose(actual, state[..., :3].cpu().numpy(), atol=1.0e-6)
