# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-scene Kit/GL presentation, pause controls, and remote-viewer startup."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True, visualizer="kit")

import logging
from pathlib import Path

import numpy as np
import omni.timeline
import pytest
import torch
from golden_image import compare_to_golden
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_visualizers.kit import KitBackendCfg
from isaaclab_visualizers.newton import NewtonGLBackendCfg
from isaaclab_visualizers.rerun import RerunBackendCfg
from isaaclab_visualizers.viser import ViserBackendCfg
from PIL import Image
from rendering_runtime import build_rendering_scene
from rendering_scene_cfgs import AnymalRenderingSceneCfg, make_rendering_scene_spec
from visualizer_checks import changed_pixels, check_motion_and_pause

from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import (
    create_rotation_matrix_from_view,
    quat_apply_inverse,
    quat_conjugate,
    quat_from_matrix,
    quat_mul,
)
from isaaclab.utils.seed import configure_seed
from isaaclab.visualizers import PerspectiveCameraCfg, SceneCameraCfg, VisualizerCfg, WindowCfg

pytestmark = pytest.mark.isaacsim_ci


@pytest.mark.parametrize("physics", ("physx", "newton"))
@pytest.mark.parametrize("scene", ("cartpole", "anymal_d", "shadow_hand"))
def test_visualizer_images_and_motion(physics, scene, request, caplog, capsys):
    """One scene supplies both viewports and four-camera grids, with independent golden references."""
    configure_seed(42, torch_deterministic=True)
    cfg = (
        AnymalRenderingSceneCfg(num_envs=4, env_spacing=4.0)
        if scene == "anymal_d"
        else make_rendering_scene_spec(scene, physics).cfg
    )
    cfg.env_spacing = 4.0
    eye, lookat = {
        "cartpole": ((2.25, 0.0, 3.5), (0.0, 0.0, 2.25)),
        "shadow_hand": ((0.4, 0.4, 1.0), (0.0, 0.0, 0.6)),
        "anymal_d": ((2.5, 2.5, 1.5), (0.0, 0.0, 0.5)),
    }[scene]
    # Attach the tiled camera to a physical body so the view follows its imported frame.
    body = {"cartpole": "cart", "shadow_hand": "Geometry/rh_forearm", "anymal_d": "base"}[scene]
    cfg.camera.prim_path = f"{{ENV_REGEX_NS}}/Robot/{body}/Camera"
    cfg.camera.width = cfg.camera.height = 200
    offset = torch.tensor([[e - t for e, t in zip(eye, lookat)]])
    orientation = quat_from_matrix(create_rotation_matrix_from_view(offset, torch.zeros_like(offset)))
    body_rotation = (2**-0.5, 0.0, 2**-0.5, 0.0) if scene == "shadow_hand" else (0.0, 0.0, 0.0, 1.0)
    parent = quat_mul(torch.tensor([cfg.robot.init_state.rot]), torch.tensor([body_rotation]))
    cfg.camera.offset = CameraCfg.OffsetCfg(
        pos=tuple(quat_apply_inverse(parent, offset)[0].tolist()),
        rot=tuple(quat_mul(quat_conjugate(parent), orientation)[0].tolist()),
        convention="opengl",
    )
    cfg.gl_camera = cfg.camera.replace(
        prim_path=cfg.camera.prim_path + "GL", renderer_cfg=NewtonWarpRendererCfg(), data_types=["rgb"]
    )
    if scene == "cartpole":
        cfg.robot.init_state.joint_pos = {"slider_to_cart": 0.0, "cart_to_pole": np.pi / 4}
        cfg.robot.init_state.joint_vel = {"slider_to_cart": 0.0, "cart_to_pole": 0.5}
    viewers = [
        VisualizerCfg(
            headless=True,
            window=WindowCfg(size=(400, 400)),
            eye=eye,
            lookat=lookat,
            background_color=(0.3, 0.55, 0.82),
            streaming_envs=list(range(4)),
            cameras=[PerspectiveCameraCfg(eye=eye, lookat=lookat), SceneCameraCfg(prim_path=camera.prim_path)],
            backend=backend,
        )
        for backend, camera in ((KitBackendCfg(), cfg.camera), (NewtonGLBackendCfg(), cfg.gl_camera))
    ]
    with build_rendering_scene(
        cfg, physics, renderer="isaac_rtx", data_types=("rgb",), visualizer_cfgs=viewers
    ) as runtime:
        assert runtime.sim.is_playing()
        assert omni.timeline.get_timeline_interface().is_playing()
        kit, gl = runtime.sim.visualizers
        origin = runtime.scene.env_origins[0].cpu().numpy()
        for visualizer in (kit, gl):
            visualizer.set_camera_view(origin + eye, origin + lookat)
        for _ in range(12):
            runtime.render_camera()
            kit.render_rgb_array()

        def capture_tiled(visualizer):
            # Kit's headless window owns app updates. Complete queued frames before reading its sensor.
            for _ in range(3):
                kit.step(0.0)
                kit.render_rgb_array()
                visualizer.current_image_view.camera.update(0.0, force_recompute=True)
            visualizer.current_image_view.invalidate()
            return visualizer.render_tiled_rgb_array().copy()

        for visualizer, name in ((kit, "kit"), (gl, "newton_gl")):
            for mode in ("viewport", "tiled"):
                if name == "newton_gl":
                    visualizer._select_camera(int(mode == "tiled"))
                    if mode == "viewport":
                        visualizer.set_camera_view(origin + eye, origin + lookat)
                frame = visualizer.render_rgb_array() if mode == "viewport" else capture_tiled(visualizer)
                assert frame is not None and np.ptp(frame) > 20
                label = f"{physics}-{name}-{mode}"
                # Preserve measured cross-GPU gates from the original viewer tests.
                max_diff_pct, min_ssim = {
                    ("cartpole", "kit", "viewport"): (7.0, 0.87),
                    ("cartpole", "kit", "tiled"): (3.0, 0.96),
                    ("cartpole", "newton_gl", "viewport"): (0.5, 0.99),
                    ("cartpole", "newton_gl", "tiled"): (0.5, 0.99),
                    ("anymal_d", "kit", "viewport"): (4.5, 0.96),
                    ("anymal_d", "kit", "tiled"): (12.0, 0.91),
                    ("anymal_d", "newton_gl", "viewport"): (0.5, 0.99),
                    ("anymal_d", "newton_gl", "tiled"): (1.0, 0.98),
                    ("shadow_hand", "kit", "viewport"): (2.0, 0.95),
                    ("shadow_hand", "kit", "tiled"): (7.0, 0.935),
                    ("shadow_hand", "newton_gl", "viewport"): (2.0, 0.97),
                    ("shadow_hand", "newton_gl", "tiled"): (2.0, 0.97),
                }[scene, name, mode]
                result = compare_to_golden(
                    Image.fromarray(frame),
                    Path(__file__).parent / "golden_images/visualizers" / scene / f"{label}.png",
                    label=f"{scene}-{label}",
                    artifact_dir="tests/comparison-images/images",
                    max_diff_pct=max_diff_pct,
                    min_ssim=min_ssim,
                    pixel_l2_threshold=20.0,
                )
                result.record(request)
                assert result.passed, result.error
        if scene == "cartpole":
            # A global perspective camera must see all partitions, independently of the tiled sensor.
            if physics == "physx":
                center = runtime.scene.env_origins.mean(dim=0).cpu().numpy() + (0, 0, 2)
                kit.set_camera_view(center + (9, 9, 7), center)
                for _ in range(8):
                    runtime.render_camera()
                    frame = kit.render_rgb_array()
                result = compare_to_golden(
                    Image.fromarray(frame),
                    Path(__file__).parent / "golden_images/visualizers/cartpole_all_envs/physx-kit-viewport.png",
                    label="cartpole-all-envs",
                    artifact_dir="tests/comparison-images/images",
                    max_diff_pct=1.0,
                    min_ssim=0.985,
                    pixel_l2_threshold=20.0,
                )
                result.record(request)
                assert result.passed, result.error
                kit.set_camera_view(origin + eye, origin + lookat)
            before = kit.render_rgb_array().copy()
            for _ in range(12):
                runtime.step()
            playing = kit.render_rgb_array().copy()
            assert changed_pixels(before, playing) > 100
            runtime.sim.pause()
            for _ in range(20):
                kit.step(0.0)
                paused = kit.render_rgb_array().copy()
            for _ in range(3):
                kit.step(0.0)
            assert changed_pixels(paused, kit.render_rgb_array(), threshold=80) <= 100  # NVBUG 6570125
            runtime.sim.play()
            for _ in range(12):
                runtime.step()
            assert changed_pixels(paused, kit.render_rgb_array()) > 100
            gl._select_camera(0)
            gl.set_camera_view(origin + eye, origin + lookat)
            check_motion_and_pause(runtime, gl)
            # Fresh renders while physics is paused must be stable; never compare two cached tiled frames.
            for visualizer in (kit, gl):
                robot = runtime.scene["robot"]
                robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch)
                robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch)
                runtime.scene.write_data_to_sim()
                runtime.sim.forward()
                runtime.scene.update(0.0)
                if visualizer is gl:
                    gl._select_camera(1)
                before = capture_tiled(visualizer)
                for _ in range(12):
                    runtime.step()
                playing = capture_tiled(visualizer)
                assert changed_pixels(before, playing, threshold=5) > 25
                runtime.sim.pause()
                paused = []
                for _ in range(2):
                    paused.append(capture_tiled(visualizer))
                # Kit temporal noise while paused: NVBUG 6570125. GL uses deterministic Warp sensor output.
                threshold = (160 if physics == "newton" else 80) if visualizer is kit else 50
                assert changed_pixels(*paused, threshold=threshold) <= 100
                runtime.sim.play()
                for _ in range(12):
                    runtime.step()
                assert changed_pixels(paused[-1], capture_tiled(visualizer), threshold=5) > 25
        assert runtime.sim.visualizers == [kit, gl]
    assert not [
        record
        for record in caplog.records
        if record.levelno >= logging.ERROR
        and ("visualizer" in record.name or record.name == "isaaclab.sim.simulation_context")
    ]
    captured = capsys.readouterr()
    assert "imgui_bundle not found" not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize("physics", ("physx", "newton"))
def test_remote_visualizers_step(physics, caplog):
    """Native remote servers start and accept simulation steps without a task or browser."""
    cfg = make_rendering_scene_spec("cartpole", physics).cfg
    viewers = [
        VisualizerCfg(backend=backend)
        for backend in (
            RerunBackendCfg(grpc_port=0, web_port=0, bind_address="127.0.0.1"),
            ViserBackendCfg(port=0, bind_address="127.0.0.1"),
        )
    ]
    with build_rendering_scene(
        cfg, physics, renderer="isaac_rtx", data_types=("rgb",), visualizer_cfgs=viewers
    ) as runtime:
        for _ in range(5):
            runtime.step()
        assert len(runtime.sim.visualizers) == 2
        assert all(visualizer._viewer is not None for visualizer in runtime.sim.visualizers)
    assert not [
        record
        for record in caplog.records
        if record.levelno >= logging.ERROR
        and ("visualizer" in record.name or record.name == "isaaclab.sim.simulation_context")
    ]
