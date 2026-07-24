# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton USD/Fabric body synchronization."""

from __future__ import annotations

import pytest

from isaaclab.app import AppLauncher

pytestmark = pytest.mark.requires_kit

# Launch Isaac Sim before importing Newton modules so USD schema bindings are initialized.
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import torch
import warp as wp
from isaaclab_newton.physics import NewtonCfg, XPBDSolverCfg

from usdrt import Rt

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.configclass import configclass


@configclass
class _RenderSceneCfg(InteractiveSceneCfg):
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


def _fabric_position(body_path: str) -> torch.Tensor:
    """Read the world position consumed by Kit/RTX from the real Fabric stage."""
    stage = sim_utils.get_current_stage(fabric=True)
    assert stage is not None, "The rendering-side Fabric stage is unavailable"

    prim = stage.GetPrimAtPath(body_path)
    assert prim.IsValid(), f"Fabric body prim does not exist: {body_path}"
    world_matrix = Rt.Xformable(prim).GetFabricHierarchyWorldMatrixAttr().Get()
    assert world_matrix is not None, f"Fabric body prim has no world matrix: {body_path}"
    translation = world_matrix.ExtractTranslation()
    return torch.tensor([float(translation[i]) for i in range(3)])


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_root_pose_write_is_visible_on_next_render_without_step():
    """A reset-time pose write must reach Kit/RTX on the next render.

    This reproduces the application sequence that used to render one-frame-old
    transforms: write asset state, then render without an intervening physics
    step or asset-data read. The assertion reads Fabric's world matrix, which is
    the transform consumed by Kit/RTX; USD is intentionally not written back.
    """
    device = "cuda:0"
    sim_cfg = SimulationCfg(
        device=device,
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(solver_cfg=XPBDSolverCfg(), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(_RenderSceneCfg(num_envs=1, env_spacing=2.0))
        sim.register_interactive_scene(scene)
        try:
            sim.reset()
            scene.reset()
            sim.render()

            cube = scene["cube"]
            body_path = "/World/envs/env_0/Cube"
            target_pose = torch.tensor(
                [[1.5, -0.75, 2.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=device,
            )

            cube.write_root_link_pose_to_sim_index(root_pose=target_pose)

            physics_steps = sim.get_physics_step_count()
            sim.render()
            wp.synchronize_device(device)

            assert sim.get_physics_step_count() == physics_steps
            torch.testing.assert_close(
                _fabric_position(body_path),
                target_pose[0, :3].cpu(),
                rtol=0.0,
                atol=1.0e-4,
            )

            # Exercise the same application boundary with a production
            # graph-safe writer. Clearing the capture-time host flag before
            # replay ensures each render depends on replayed device-side
            # invalidation rather than Python code that ran during capture.
            env_mask = wp.ones(1, dtype=wp.bool, device=device)
            pose_buffer = target_pose.clone()
            cube.write_root_link_pose_to_sim_mask(root_pose=pose_buffer, env_mask=env_mask)
            sim.render()

            torch.cuda.synchronize(device)
            with wp.ScopedCapture(device=device) as capture:
                cube.write_root_link_pose_to_sim_mask(root_pose=pose_buffer, env_mask=env_mask)

            sim.render()

            replay_targets = (
                torch.tensor([2.5, 0.5, 1.25, 0.0, 0.0, 0.0, 1.0], device=device),
                torch.tensor([-1.25, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0], device=device),
            )
            for replay_target in replay_targets:
                pose_buffer.copy_(replay_target.unsqueeze(0))
                torch.cuda.synchronize(device)
                wp.capture_launch(capture.graph)
                wp.synchronize_device(device)

                physics_steps = sim.get_physics_step_count()
                sim.render()
                wp.synchronize_device(device)

                assert sim.get_physics_step_count() == physics_steps
                torch.testing.assert_close(
                    _fabric_position(body_path),
                    replay_target[:3].cpu(),
                    rtol=0.0,
                    atol=1.0e-4,
                )
        finally:
            sim.register_interactive_scene(None)
