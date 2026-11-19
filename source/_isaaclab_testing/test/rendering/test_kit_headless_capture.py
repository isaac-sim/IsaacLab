# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless Kit sensor captures must refresh after scene and camera preparation."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True, visualizer="kit", headless=True)

import pytest
import torch
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_visualizers.kit import KitBackendCfg
from rendering_runtime import build_rendering_scene
from rendering_scene_cfgs import CartpoleRenderingSceneCfg
from visualizer_checks import changed_pixels

from isaaclab.visualizers import SceneCameraCfg, VisualizerCfg

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]


@pytest.mark.rendering
def test_headless_kit_streaming_captures_motion_without_follow():
    """A fixed sensor view stays live when headless Kit presentation does not step the app."""
    cfg = CartpoleRenderingSceneCfg(num_envs=1, env_spacing=2.0)
    cfg.robot.init_state.joint_pos = {"slider_to_cart": 0.0, "cart_to_pole": 0.6}
    cfg.camera.renderer_cfg = IsaacRtxRendererCfg()
    viewer = VisualizerCfg(
        headless=True,
        backend=KitBackendCfg(),
        streaming_envs=[0],
        cameras=[SceneCameraCfg(prim_path=cfg.camera.prim_path)],
    )
    with build_rendering_scene(cfg, "physx", data_types=("rgb",), visualizer_cfgs=[viewer]) as runtime:
        visualizer = runtime.sim.visualizers[0]
        robot = runtime.scene["robot"]
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch)
        robot.write_joint_velocity_to_sim_index(velocity=torch.full_like(robot.data.default_joint_vel.torch, 0.5))
        initial_positions = robot.data.joint_pos.torch.clone()
        frames = []
        for _ in range(16):
            runtime.step()
            step = runtime.sim.get_physics_step_count()
            positions = robot.data.joint_pos.torch.clone()
            frames.append(visualizer.render_tiled_rgb_array().copy())
            assert runtime.sim.get_physics_step_count() == step, "Capture advanced physics"
            torch.testing.assert_close(robot.data.joint_pos.torch, positions, rtol=0.0, atol=0.0)
        assert torch.max(torch.abs(robot.data.joint_pos.torch - initial_positions)) > 1e-3
        assert any(changed_pixels(a, b) > 10 for a, b in zip(frames[3:], frames[4:], strict=False)), (
            "Headless streaming frames stayed stale while Cartpole moved"
        )
