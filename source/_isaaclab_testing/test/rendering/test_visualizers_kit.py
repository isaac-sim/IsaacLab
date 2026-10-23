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
import pytest
import torch
from golden_image import compare_to_golden
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_visualizers.kit import KitBackendCfg
from isaaclab_visualizers.newton import NewtonGLBackendCfg
from isaaclab_visualizers.rerun import RerunBackendCfg
from isaaclab_visualizers.viser import ViserBackendCfg
from PIL import Image
from rendering_runtime import build_rendering_scene
from rendering_scene_cfgs import AnymalRenderingSceneCfg, make_rendering_scene_spec
from visualizer_checks import changed_pixels, check_motion_and_pause

import omni.timeline

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.sensors import CameraCfg
from isaaclab.utils import instantiate
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
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


@pytest.mark.parametrize(
    "scene,mode,physics",
    [
        (scene, mode, physics)
        for scene in ("cartpole", "anymal_d", "shadow_hand")
        for mode in ("viewport", "tiled")
        for physics in ("physx", "newton")
    ]
    + [("cartpole", "all_envs", "physx")],
)
def test_visualizer_images_and_motion(scene, mode, physics, request, caplog, capsys):
    """Render the original scenes and poses through both backends without task managers.

    scene_states.npz records the original captures at 8c9b995: root/joint poses, cube poses,
    and marker transforms. Fixed inputs preserve the subject without replaying RL resets.
    Viewports retain one environment; tiled cameras retain four independent environments.
    """
    configure_seed(42, torch_deterministic=True)
    cfg = (
        AnymalRenderingSceneCfg(num_envs=4, env_spacing=2.5)
        if scene == "anymal_d"
        else make_rendering_scene_spec(scene, physics).cfg
    )
    cfg.num_envs = 1 if mode == "viewport" else 4
    cfg.env_spacing = {"cartpole": 20.0, "anymal_d": 2.5, "shadow_hand": 0.75}[scene]
    eye, lookat = {
        "cartpole": ((2.25, 0.0, 3.5), (0.0, 0.0, 2.25)),
        "shadow_hand": ((0.4, 0.4, 1.0), (0.0, 0.0, 0.6)),
        "anymal_d": ((2.5, 2.5, 1.5), (0.0, 0.0, 0.5)),
    }[scene]
    if mode == "all_envs":
        cfg.env_spacing = 8.0
        eye, lookat = (9.0, 9.0, 10.0), (0.0, 0.0, 2.25)
    if scene == "shadow_hand":
        cfg.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(), collision_group=-1)
        cfg.goal_object = VisualizationMarkersCfg(
            prim_path="/Visuals/goal_marker",
            markers={
                "goal": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
                )
            },
        )
    elif scene == "anymal_d":
        cfg.fill_light.spawn.texture_file = retrieve_file_path(cfg.fill_light.spawn.texture_file)
    body = {"cartpole": "slider", "shadow_hand": "Geometry/rh_forearm", "anymal_d": "base"}[scene]
    cfg.camera.prim_path = f"{{ENV_REGEX_NS}}/Robot/{body}/Camera"
    cfg.camera.width = cfg.camera.height = 200
    cfg.camera.spawn = sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    )
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
            headless=isinstance(backend, NewtonGLBackendCfg),
            window=WindowCfg(size=(400, 400)),
            eye=eye,
            lookat=lookat,
            background_color=None if isinstance(backend, KitBackendCfg) else (0.3, 0.55, 0.82),
            streaming_envs=list(range(cfg.num_envs)),
            cameras=[PerspectiveCameraCfg(eye=eye, lookat=lookat), SceneCameraCfg(prim_path=camera.prim_path)],
            backend=backend,
        )
        for backend, camera in ((KitBackendCfg(), cfg.camera), (NewtonGLBackendCfg(), cfg.gl_camera))
    ]
    cfg.camera.renderer_cfg = IsaacRtxRendererCfg()
    with build_rendering_scene(cfg, physics, data_types=("rgb",), visualizer_cfgs=viewers) as runtime:
        assert runtime.sim.is_playing()
        assert omni.timeline.get_timeline_interface().is_playing()
        kit, gl = runtime.sim.visualizers
        if scene == "cartpole":
            robot = runtime.scene["robot"]
            robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch)
            robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch)
            runtime.scene.write_data_to_sim()
            runtime.sim.forward()
            runtime.scene.update(0.0)
        markers = {}
        if scene == "shadow_hand":
            markers["goal_marker"] = runtime.scene["goal_object"]
        elif scene == "anymal_d":
            for name, marker_cfg in (
                ("velocity_goal", GREEN_ARROW_X_MARKER_CFG),
                ("velocity_current", BLUE_ARROW_X_MARKER_CFG),
            ):
                marker_cfg = marker_cfg.replace(prim_path=f"/Visuals/Command/{name}")
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                markers[name] = instantiate(marker_cfg)

        def capture_tiled(visualizer):
            # Complete queued Kit frames before reading the sensor's current state.
            for _ in range(3):
                kit.step(0.0)
                kit.render_rgb_array()
                visualizer.current_image_view.camera.update(0.0, force_recompute=True)
            visualizer.current_image_view.invalidate()
            return visualizer.render_tiled_rgb_array().copy()

        states = np.load(Path(__file__).parent / "golden_images/visualizers/scene_states.npz")
        captures = ((kit, "kit"),) if mode == "all_envs" else ((kit, "kit"), (gl, "newton_gl"))
        for visualizer, name in captures:
            if visualizer is gl:
                gl._select_camera(int(mode == "tiled"))
            if scene != "cartpole":
                key = f"{scene}/{physics}/{'kit' if visualizer is kit else 'newton'}/{mode}"
                robot = runtime.scene["robot"]
                assert robot.joint_names == states[f"{key}/joint_names"].tolist()
                robot.write_root_pose_to_sim_index(
                    root_pose=torch.tensor(states[f"{key}/root_pose"], device=runtime.sim.device)
                )
                positions = torch.tensor(states[f"{key}/joint_pos"], device=runtime.sim.device)
                robot.write_joint_position_to_sim_index(position=positions)
                robot.actuators.target_command.set_position_index(value=positions)
                if scene == "shadow_hand":
                    runtime.scene["object"].write_root_pose_to_sim_index(
                        root_pose=torch.tensor(states[f"{key}/object_pose"], device=runtime.sim.device)
                    )
                for marker_name, marker in markers.items():
                    marker.visualize(
                        translations=states[f"{key}/{marker_name}/translations"],
                        orientations=states[f"{key}/{marker_name}/orientations"],
                        scales=states[f"{key}/{marker_name}/scales"] if scene == "anymal_d" else None,
                        environment_ids=torch.arange(cfg.num_envs, device=runtime.sim.device),
                    )
                runtime.scene.write_data_to_sim()
                runtime.sim.forward()
                runtime.scene.update(0.0)
                if scene == "anymal_d" and mode == "tiled":
                    runtime.render_camera()
                    camera = visualizer.current_image_view.camera
                    targets = robot.data.root_pos_w.torch
                    camera.set_world_poses_from_view(
                        targets + torch.tensor((2.5, 2.5, 1.0), device=runtime.sim.device), targets
                    )
            for _ in range(60 if scene == "shadow_hand" else 20):
                runtime.render_camera()
                if mode != "tiled":
                    visualizer.set_camera_view(eye, lookat)
                kit.render_rgb_array()
            frame = capture_tiled(visualizer) if mode == "tiled" else visualizer.render_rgb_array()
            assert frame is not None and np.ptp(frame) > 20
            image_scene = "cartpole_all_envs" if mode == "all_envs" else scene
            image_mode = "viewport" if mode == "all_envs" else mode
            max_diff_pct, min_ssim = {
                ("cartpole", "kit", "viewport"): (7.0, 0.87),
                ("cartpole", "kit", "tiled"): (3.0, 0.96),
                ("cartpole", "newton_gl", "viewport"): (0.5, 0.99),
                ("cartpole", "newton_gl", "tiled"): (0.5, 0.99),
                ("cartpole_all_envs", "kit", "viewport"): (1.0, 0.985),
                ("anymal_d", "kit", "viewport"): (4.5, 0.96),
                ("anymal_d", "kit", "tiled"): (12.0, 0.91),
                ("anymal_d", "newton_gl", "viewport"): (0.5, 0.99),
                ("anymal_d", "newton_gl", "tiled"): (1.0, 0.98),
                ("shadow_hand", "kit", "viewport"): (2.0, 0.95),
                ("shadow_hand", "kit", "tiled"): (7.0, 0.935),
                ("shadow_hand", "newton_gl", "viewport"): (2.0, 0.97),
                ("shadow_hand", "newton_gl", "tiled"): (2.0, 0.97),
            }[image_scene, name, image_mode]
            label = f"{physics}-{name}-{image_mode}"
            result = compare_to_golden(
                Image.fromarray(frame),
                Path(__file__).parent / "golden_images/visualizers" / image_scene / f"{label}.png",
                label=f"{image_scene}-{label}",
                artifact_dir="tests/comparison-images/images",
                max_diff_pct=max_diff_pct,
                min_ssim=min_ssim,
                pixel_l2_threshold=20.0,
            )
            result.record(request)
            assert result.passed, result.error
        states.close()
        if scene == "cartpole" and mode == "viewport":
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
            gl.set_camera_view(eye, lookat)
            check_motion_and_pause(runtime, gl)
        if scene == "cartpole" and mode == "tiled":
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
    cfg.camera.renderer_cfg = IsaacRtxRendererCfg()
    with build_rendering_scene(cfg, physics, data_types=("rgb",), visualizer_cfgs=viewers) as runtime:
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
