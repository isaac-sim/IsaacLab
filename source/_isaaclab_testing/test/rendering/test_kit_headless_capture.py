# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless Kit sensor captures must refresh after scene and camera preparation."""

import sys
from contextlib import closing
from types import ModuleType, SimpleNamespace

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True, visualizer="kit", headless=True)

import pytest
import torch
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_visualizers.kit import KitBackendCfg
from rendering_runtime import build_rendering_scene
from rendering_scene_cfgs import CartpoleRenderingSceneCfg
from visualizer_checks import changed_pixels

import omni.kit.app

from isaaclab.envs.utils.video_recorder import VideoRecorder
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.visualizers import SceneCameraCfg, VisualizerCfg

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]


@pytest.mark.rendering
def test_headless_kit_streaming_captures_motion_without_follow(monkeypatch, tmp_path):
    """A fixed sensor view records live frames while Kit stays idle between clips."""
    cfg = CartpoleRenderingSceneCfg(num_envs=1, env_spacing=2.0)
    cfg.robot.init_state.joint_pos = {"slider_to_cart": 0.0, "cart_to_pole": 0.6}
    cfg.camera.renderer_cfg = IsaacRtxRendererCfg()
    viewer = VisualizerCfg(
        headless=True,
        backend=KitBackendCfg(),
        streaming_envs=[0],
        cameras=[SceneCameraCfg(prim_path=cfg.camera.prim_path)],
    )
    clips = []

    class CountClip:
        def __init__(self, frames, **_kwargs):
            clips.append([frame.copy() for frame in frames])

        def write_videofile(self, *_args, **_kwargs):
            pass

    encoder_module = ModuleType("moviepy.video.io.ImageSequenceClip")
    encoder_module.ImageSequenceClip = CountClip
    monkeypatch.setitem(sys.modules, encoder_module.__name__, encoder_module)

    with build_rendering_scene(cfg, "physx", data_types=("rgb",), visualizer_cfgs=[viewer]) as runtime:
        previous_video_flag = runtime.sim.get_setting("/isaaclab/video/enabled")
        runtime.sim.set_setting("/isaaclab/video/enabled", True)
        try:
            robot = runtime.scene["robot"]
            robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch)
            robot.write_joint_velocity_to_sim_index(velocity=torch.full_like(robot.data.default_joint_vel.torch, 0.5))
            runtime.stabilize_camera()
            initial_positions = robot.data.joint_pos.torch.clone()
            recorder_cfg = VideoRecorderCfg(
                source="viz:kit:streaming_view",
                video_length=4,
                video_interval=8,
                step_offset=4,
                fps=60,
                output_dir=str(tmp_path),
            )
            env = SimpleNamespace(sim=runtime.sim, scene=runtime.scene, step_dt=runtime.sim.get_physics_dt())
            app = omni.kit.app.get_app()
            with closing(VideoRecorder(recorder_cfg, env)) as recorder:
                for index in range(16):
                    ticks = app.get_update_number()
                    runtime.step()
                    step = runtime.sim.get_physics_step_count()
                    positions = robot.data.joint_pos.torch.clone()
                    recorder.step()
                    # The offset and recurring gap leave steps 1-4 and 9-12 idle.
                    if index in (0, 1, 2, 3, 8, 9, 10, 11):
                        assert app.get_update_number() == ticks, "Kit updated between recording windows"
                    assert runtime.sim.get_physics_step_count() == step, "Capture advanced physics"
                    torch.testing.assert_close(robot.data.joint_pos.torch, positions, rtol=0.0, atol=0.0)
            assert [len(clip) for clip in clips] == [4, 4]
            frames = [frame for clip in clips for frame in clip]
            assert torch.max(torch.abs(robot.data.joint_pos.torch - initial_positions)) > 1e-3
            assert any(changed_pixels(a, b) > 10 for a, b in zip(frames[3:], frames[4:], strict=False)), (
                "Headless streaming frames stayed stale while Cartpole moved"
            )
        finally:
            runtime.sim.set_setting("/isaaclab/video/enabled", previous_video_flag)
