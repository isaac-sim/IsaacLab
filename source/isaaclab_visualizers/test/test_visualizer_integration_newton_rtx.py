# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared RTX motion, background, camera, and resource-lifetime checks.

Kept apart from ``test_visualizer_integration_newton``: OVRTX cannot share a process with Kit, which
that module launches at import.
"""

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxManager

from isaaclab.test.utils import launch_test_simulation
from isaaclab.visualizers import VisualizerCfg

launch_test_simulation(NewtonCfg(solver_cfg=MJWarpSolverCfg()), visualizer=["newton_rtx"])
# This process exercises both physics backends; register schemas before the first native USD population.
OvPhysxManager._prepare_stage_creation()


import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab_visualizers.newton import NewtonRTXBackendCfg  # noqa: E402

from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.cloner import ReplicateSession  # noqa: E402
from isaaclab.sim import create_new_stage  # noqa: E402
from isaaclab.utils import instantiate  # noqa: E402
from isaaclab.visualizers import SceneCameraCfg, WindowCfg  # noqa: E402

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

import visualizer_integration_utils as _viz_utils  # noqa: E402


@pytest.mark.rendering
@pytest.mark.kitless
def test_newton_rtx_mpm_points_follow_world_state():
    """Authored particle geometry follows native state updates without repeating its parent transform."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.physics import MPMSolverCfg
    from isaaclab_newton.sim.spawners.mpm import MPMGridCfg

    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils import configclass
    from isaaclab.visualizers import PerspectiveCameraCfg

    @configclass
    class ParticleSceneCfg(InteractiveSceneCfg):
        fluid = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Fluid",
            spawn=MPMGridCfg(
                lower=(-0.04, -0.04, -0.04),
                upper=(0.04, 0.04, 0.04),
                voxel_size=0.04,
                particle_placement="cell_center",
                radius=0.015,
                visual_color=(0.1, 0.4, 1.0),
            ),
            init_state=MPMObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1000.0))

    cfg = SimulationCfg(
        dt=1.0 / 120.0,
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05), use_cuda_graph=False),
        visualizer_cfgs=[
            VisualizerCfg(
                headless=True,
                window=WindowCfg(size=(256, 256)),
                cameras=[PerspectiveCameraCfg(eye=(0.0, -1.0, 1.1), lookat=(0.0, 0.0, 1.1))],
                backend=NewtonRTXBackendCfg(render_settings={"omni:rtx:quality": ("Int", 0)}),
            )
        ],
    )
    scene_cfg = ParticleSceneCfg(num_envs=1, env_spacing=0.0)
    with launch_simulation(SimpleNamespace(sim=cfg, scene=scene_cfg), {"visualizer": ["newton_rtx"]}):
        sim = SimulationContext(cfg)
        scene = instantiate(scene_cfg)
        sim.reset()
        fluid, visualizer = scene["fluid"], sim.visualizers[0]
        _viz_utils._warm_newton_viewer(visualizer)
        initial = fluid.data.particle_state_w.torch.clone()
        centers = []
        for offset in (0.0, 0.10):
            state = initial.clone()
            state[..., 2] += offset
            fluid.write_particle_state_to_sim_index(state)
            # Allow Newton's in-flight perspective frame to complete at each held pose.
            for _ in range(3):
                sim.step(render=False)
                scene.update(sim.get_physics_dt())
                pixels = visualizer.render_rgb_array()
            color = pixels.astype(np.int16)
            blue = (color[..., 2] - color[..., 0] > 50) & (color[..., 2] - color[..., 1] > 20)
            assert blue.any(), "The authored particle cloud was not rendered"
            centers.append(np.argwhere(blue)[:, 0].mean())
        assert centers[1] < centers[0] - 5.0, "The rendered particle cloud did not move upward"


@pytest.mark.rendering
@pytest.mark.kitless
def test_directly_spawned_camera_drives_rtx_window():
    """A standalone script declares its scene through ReplicateSession and displays an ordinary Camera."""
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_ov.renderers import OVRTXRendererCfg

    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation
    from isaaclab.sensors import CameraCfg

    camera_cfg = CameraCfg(
        prim_path="/World/Camera", width=96, height=64, data_types=["rgba", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(), renderer_cfg=OVRTXRendererCfg(),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 5.0), convention="opengl"),
        render_settings={"omni:rtx:quality": ("Int", 0)},
    )  # fmt: skip
    sim_cfg = sim_utils.SimulationCfg(
        physics=OvPhysxCfg(),
        visualizer_cfgs=[VisualizerCfg(
            headless=True, cameras=[SceneCameraCfg(prim_path=camera_cfg.prim_path)], window=WindowCfg(size=(96, 64)),
            backend=NewtonRTXBackendCfg(),
        )],
    )  # fmt: skip
    target_cfg = AssetBaseCfg(prim_path="/World/Target", spawn=sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0)))
    light_cfg = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1000.0))
    with launch_simulation(sim_cfg, {"visualizer": ["newton_rtx"], "device": "cuda:0"}):
        sim = sim_utils.SimulationContext(sim_cfg)
        with ReplicateSession((target_cfg, light_cfg, camera_cfg), num_clones=3, env_spacing=0.0):
            instantiate(target_cfg)
            instantiate(light_cfg)
            camera = instantiate(camera_cfg)
        sim.reset()
        visualizer = sim.visualizers[0]
        assert visualizer.current_image_view.camera is camera
        assert visualizer.render_rgb_array().shape == (64, 96, 3)
        assert np.ptp(camera.data.output["rgba"].warp.numpy()[..., :3]) > 20
        # This world camera has no per-environment rows; partial scene resets reset its one frame.
        camera.reset([2])
        np.testing.assert_array_equal(camera.frame.warp.numpy(), [0])
        camera.data
        camera.reset(env_mask=wp.array([False, False, True], dtype=wp.bool, device=sim.device))
        np.testing.assert_array_equal(camera.frame.warp.numpy(), [0])
        focal = float(camera._data.intrinsic_matrices.torch[0, 0, 0])
        output_ptr = camera.data.output["rgba"].warp.ptr
        depth = camera.data.output["distance_to_image_plane"].torch[0, 32, :, 0]
        assert depth[48].item() == pytest.approx(4.5, abs=0.02)
        initial_width = ((depth > 4.4) & (depth < 4.6)).sum().item()
        visualizer._viewer.camera.fov *= 0.5
        visualizer.step(0.0)
        visualizer.render_rgb_array()
        assert float(camera._data.intrinsic_matrices.torch[0, 0, 0]) > 1.5 * focal
        assert camera.data.output["rgba"].warp.ptr == output_ptr
        assert ((depth > 4.4) & (depth < 4.6)).sum().item() > 1.8 * initial_width
        visualizer.close()
        camera.update(sim_cfg.dt)
        assert np.ptp(camera.data.output["rgba"].warp.numpy()[..., :3]) > 20


@pytest.mark.arm_ci
def test_newton_rtx_motion_pause_and_markers(caplog: pytest.LogCaptureFixture, monkeypatch) -> None:
    """RTX preserves motion, pause controls, markers, and device-updated surface meshes."""
    env = None
    try:
        _viz_utils._prepare_visualizer_test_process()
        create_new_stage()
        env = _viz_utils._make_cartpole_camera_env(visualizer_kind=("newton_rtx",), backend_kind="newton")
        env.cfg.initial_pole_angle_range = _viz_utils._INTEGRATION_TEST_POLE_ANGLE_RANGE
        env.cfg.initial_pole_velocity_range = _viz_utils._INTEGRATION_TEST_POLE_VELOCITY_RANGE
        _viz_utils._configure_sim_for_visualizer_test(env)
        with caplog.at_level(logging.WARNING):
            env.reset()
            visualizer = env.sim.visualizers[0]
            actions = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
            # The first capture must succeed before warm-up can suppress its error.
            first_frame = visualizer.render_rgb_array()
            width, height = _viz_utils._CARTPOLE_NEWTON_INTEGRATION_WINDOW_SIZE
            assert first_frame is not None and first_frame.shape == (height, width, 3)
            _viz_utils._run_newton_viewer_frame_motion_test(
                env,
                visualizer._viewer,
                visualizer=visualizer,
                step_hook=lambda: env.step(action=actions),
                get_physics_step_count=env.sim.get_physics_step_count,
                physics_kind="newton",
                viz_kind="newton_rtx",
            )
            _viz_utils._assert_newton_rtx_markers_drawn(env, visualizer, case_label="newton_rtx/newton")
            points = wp.array([(0.5, -0.8, 1.7), (1.3, -0.8, 1.7), (0.5, 0.8, 1.7)], dtype=wp.vec3, device=env.device)
            indices = wp.array([0, 1, 2], dtype=wp.uint32, device=env.device)
            visualizer.log_mesh("surface", points, indices, color=(0.0, 0.0, 1.0), dynamic=True)
            _viz_utils._warm_newton_viewer(visualizer)
            initial_mesh = visualizer.render_rgb_array()
            points.assign(np.array([(0.5, -0.8, 2.5), (1.3, -0.8, 2.5), (0.5, 0.8, 2.5)], np.float32))
            with monkeypatch.context() as gpu_update:
                gpu_update.setattr(wp.array, "numpy", Mock(side_effect=AssertionError("Geometry readback")))
                visualizer.log_mesh("surface", points, indices, dynamic=True)
            # Recording a shared view must see geometry updates without drawing the window.
            updated_mesh = visualizer.current_image_view.read_rgb(env.sim.get_physics_step_count())
            blue_rows = []
            for pixels in (initial_mesh, updated_mesh):
                blue = (pixels[..., 2] > 50) & (pixels[..., 2] > 1.5 * pixels[..., :2].max(axis=2))
                assert blue.any(), "The runtime mesh must be visible in RTX output."
                blue_rows.append(np.nonzero(blue)[0].mean())
            assert blue_rows[1] < blue_rows[0] - 10, "Moving the mesh upward must move its rendered pixels."
            visualizer.log_mesh("surface", points, indices, color=(0.0, 1.0, 0.0), roughness=0.9)
            recolored = visualizer.current_image_view.read_rgb(env.sim.get_physics_step_count())
            green = (recolored[..., 1] > 50) & (recolored[..., 1] > 1.5 * recolored[..., (0, 2)].max(axis=2))
            assert (green & blue).sum() > blue.sum() / 2, "A later material update must reach recorded pixels."
            uvs = wp.array([(0, 0), (1, 0), (0, 1)], dtype=wp.vec2, device=env.device)
            for color in ((255, 0, 0), (0, 0, 255)):
                texture = np.full((2, 2, 3), color, np.uint8)
                visualizer.log_mesh("surface", points, indices, uvs=uvs, texture=texture, color=(1.0, 1.0, 1.0))
                textured = visualizer.current_image_view.read_rgb(env.sim.get_physics_step_count())
                assert np.argmax(textured[blue].mean(axis=0)) == np.argmax(color)
        _viz_utils._assert_no_visualizer_log_issues(caplog)
    finally:
        _viz_utils._cleanup_visualizer_test_process(env)


@pytest.mark.rendering
@pytest.mark.kitless
@pytest.mark.parametrize("lighting", [True, False])
@pytest.mark.parametrize("background_color", [None, (0.0, 0.0, 1.0)])
def test_newton_rtx_scene_sky_and_background_override(tmp_path, monkeypatch, lighting, background_color):
    """The window shares sensor rendering, preserves lighting, and owns no sensor resources."""
    import gymnasium as gym
    from isaaclab_newton.renderers import NewtonWarpRendererCfg
    from isaaclab_ov.renderers import OVRTXRendererCfg

    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation
    from isaaclab.assets import VisualMaterialCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.visualizers import PerspectiveCameraCfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    monkeypatch.setattr(
        "newton.viewer.ViewerRTX.__init__",
        Mock(side_effect=AssertionError("RTX windows must use the simulation renderer, not construct ViewerRTX")),
    )
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")

    texture = tmp_path / "red.hdr"
    texture.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n" + bytes((128, 0, 0, 129)) * 8)
    cfg, _ = resolve_task_config("Isaac-Cartpole", "", overrides=("physics=newton_mjwarp",))
    cfg.scene.num_envs = 2
    cfg.seed = 0
    cfg.scene.distant_light = AssetBaseCfg(
        prim_path="/World/Sky", spawn=sim_utils.DomeLightCfg(texture_file=str(texture), intensity=750.0)
    )
    if not lighting:
        cfg.scene.distant_light = None
    cfg.scene.target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0), visual_material=sim_utils.PreviewSurfaceCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -2.0, 2.0)),
    )
    cfg.scene.material = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Target/geometry/material", spawn=None)
    cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        width=128,
        height=128,
        data_types=["rgba"],
        renderer_cfg=OVRTXRendererCfg() if lighting else NewtonWarpRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=15.2908),
        offset=CameraCfg.OffsetCfg(pos=(0.0, -6.0, 2.0), rot=(0.0, 0.0, 2**-0.5, 2**-0.5), convention="world"),
    )
    cfg.sim.visualizer_cfgs = [
        VisualizerCfg(
            headless=True,
            window=WindowCfg(size=(128, 128)),
            streaming_envs=[0],
            background_color=background_color,
            cameras=[
                PerspectiveCameraCfg(eye=(0.0, -6.0, 2.0), lookat=(0.0, -2.0, 2.0), focal_length=24.0),
                SceneCameraCfg(prim_path="{ENV_REGEX_NS}/Camera"),
            ],
            backend=NewtonRTXBackendCfg(),
        )
    ]
    with launch_simulation(cfg, {"visualizer": ["newton_rtx"], "device": "cuda:0"}):
        env = gym.make("Isaac-Cartpole", cfg=cfg)
        try:
            env.reset()
            visualizer = env.unwrapped.sim.visualizers[0]
            camera = env.unwrapped.scene["camera"]
            viewer = visualizer._viewer
            renderer = visualizer.current_image_view.camera._renderer
            native_renderer = renderer.backend.renderer
            if lighting:
                assert renderer is camera._renderer
            assert len(renderer._camera_render_data) == (2 if lighting else 1)
            assert visualizer.current_image_view.camera.num_instances == 1
            assert len(env.unwrapped.scene.sensors) == 2
            origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
            visualizer.set_camera_view(origin + (0.0, -6.0, 2.0), origin + (0.0, -2.0, 2.0))
            for _ in range(40):
                env.unwrapped.sim.step(render=False)
                pixels = visualizer.render_rgb_array()
            background = pixels[8:24, 8:24].mean(axis=(0, 1))
            color = pixels[56:72, 56:72].mean(axis=(0, 1))
            channel = 0 if background_color is None else 2
            if lighting or background_color is not None:
                assert background[channel] > 180 and np.delete(background, channel).max() < 10, background
            else:
                assert background.max() < 10, background
            if lighting:
                assert color[0] > 80 and color[0] > 2 * color[1:].max(), color
            else:
                assert color.max() < 10, color

            output = camera.data.output["rgba"].warp
            sensor_frame = camera.frame.torch.clone()
            sensor_ptr = output.ptr
            captured_output = wp.empty_like(output)
            perspective = visualizer.current_image_view.read(env.unwrapped.sim.get_physics_step_count())
            perspective_ptr = perspective.ptr
            captured_perspective = wp.empty_like(perspective)
            with wp.ScopedCapture(device=output.device) as capture:
                wp.copy(captured_output, output)
                wp.copy(captured_perspective, perspective)
            viewer.renderer.window.set_size(160, 96)
            viewer.renderer.window.dispatch_events()
            pixels = visualizer.render_rgb_array()
            assert pixels.shape == (128, 128, 3)
            assert camera.image_shape == (128, 128)
            assert camera.data.output["rgba"].warp.ptr == sensor_ptr
            assert visualizer.current_image_view.read(env.unwrapped.sim.get_physics_step_count()).ptr == perspective_ptr
            torch.testing.assert_close(camera.frame.torch, sensor_frame)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(captured_output.numpy(), output.numpy())
            np.testing.assert_array_equal(captured_perspective.numpy(), perspective.numpy())

            visualizer._select_camera(1)
            image = visualizer.render_tiled_rgba_array()
            np.testing.assert_array_equal(image.numpy(), camera.data.output["rgba"].warp.numpy()[0])
            assert image.device.is_cuda
            with monkeypatch.context() as gpu_display:
                gpu_display.setattr(wp.array, "numpy", Mock(side_effect=AssertionError("Display downloaded pixels")))
                visualizer._select_camera(0)
                visualizer._render_frame()
                visualizer._select_camera(1)
                visualizer._render_frame()
            if lighting and background_color is None:
                env.unwrapped.sim.reset()
                visualizer._select_camera(0)
                assert visualizer.render_rgb_array().shape == (128, 128, 3)
            visualizer.close()
            assert renderer.backend.renderer is native_renderer
            camera.update(cfg.sim.dt)
            assert np.ptp(camera.data.output["rgba"].warp.numpy()) > 0
        finally:
            env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
