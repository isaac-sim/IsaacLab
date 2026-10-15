# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native Newton RTX motion, background, camera, and resource-lifetime checks.

Kept apart from ``test_visualizer_integration_newton``: OVRTX cannot share a process with Kit, which
that module launches at import.
"""

import logging
import os
import sys
from pathlib import Path
from unittest.mock import Mock

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(NewtonCfg(solver_cfg=MJWarpSolverCfg()), visualizer=["newton_rtx"])


import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg  # noqa: E402

from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.sim import create_new_stage  # noqa: E402
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
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, build_simulation_context
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
            NewtonRTXVisualizerCfg(
                headless=True,
                window=WindowCfg(size=(256, 256)),
                cameras=[PerspectiveCameraCfg(eye=(0.0, -1.0, 1.1), lookat=(0.0, 0.0, 1.1))],
                render_settings={"omni:rtx:quality": ("Int", 0)},
            )
        ],
    )
    with build_simulation_context(sim_cfg=cfg) as sim:
        scene = InteractiveScene(ParticleSceneCfg(num_envs=1, env_spacing=0.0))
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


@pytest.mark.arm_ci
def test_newton_rtx_motion_pause_and_markers(caplog: pytest.LogCaptureFixture) -> None:
    """Newton RTX frames are non-flat, move while playing, freeze while paused, and draw a marker."""
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
            # The first async capture must succeed before warm-up can suppress its error.
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
        _viz_utils._assert_no_visualizer_log_issues(caplog)
    finally:
        _viz_utils._cleanup_visualizer_test_process(env)


@pytest.mark.rendering
@pytest.mark.kitless
@pytest.mark.parametrize("lighting", [True, False])
@pytest.mark.parametrize("background_color", [None, (0.0, 0.0, 1.0)])
def test_newton_rtx_scene_sky_and_background_override(tmp_path, monkeypatch, lighting, background_color):
    """The native viewer preserves authored lighting and borrows a stage without owning sensors."""
    if lighting:
        pytest.skip(
            "Two OVRTX renderers can hang; re-enable after adopting https://github.com/newton-physics/newton/pull/4627."
        )
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
        NewtonRTXVisualizerCfg(
            headless=True,
            window=WindowCfg(size=(128, 128)),
            streaming_envs=[0],
            background_color=background_color,
            cameras=[
                PerspectiveCameraCfg(eye=(0.0, -6.0, 2.0), lookat=(0.0, -2.0, 2.0), focal_length=24.0),
                SceneCameraCfg(prim_path="{ENV_REGEX_NS}/Camera"),
            ],
        )
    ]
    with launch_simulation(cfg, {"visualizer": ["newton_rtx"], "device": "cuda:0"}):
        env = gym.make("Isaac-Cartpole", cfg=cfg)
        try:
            env.reset()
            visualizer = env.unwrapped.sim.visualizers[0]
            camera = env.unwrapped.scene["camera"]
            viewer = visualizer._viewer
            native_stage = viewer._borrowed_stage
            # Exercise the native presentation path through an EGL window in headless CI.
            viewer._headless = False
            assert visualizer.current_image_view.camera is None
            assert len(env.unwrapped.scene.sensors) == 1
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
            with wp.ScopedCapture(device=output.device) as capture:
                wp.copy(captured_output, output)
            viewer._window.set_size(160, 96)
            viewer._window.dispatch_events()
            pixels = visualizer.render_rgb_array()
            assert pixels.shape == (128, 128, 3)
            assert camera.image_shape == (128, 128)
            assert camera.data.output["rgba"].warp.ptr == sensor_ptr
            torch.testing.assert_close(camera.frame.torch, sensor_frame)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(captured_output.numpy(), output.numpy())

            visualizer._select_camera(1)
            image = visualizer.render_tiled_rgba_array()
            np.testing.assert_array_equal(image.numpy(), camera.data.output["rgba"].warp.numpy()[0])
            assert image.device.is_cuda
            with monkeypatch.context() as gpu_display:
                gpu_display.setattr(wp.array, "numpy", Mock(side_effect=AssertionError("Display downloaded pixels")))
                visualizer._render_frame()
            if lighting and background_color is None:
                env.unwrapped.sim.reset()
                visualizer._select_camera(0)
                assert visualizer.render_rgb_array().shape == (128, 128, 3)
            visualizer.close()
            query = native_stage.get_attribute_write_floor()
            try:
                assert native_stage.fetch_ordinal(query) > 0
            finally:
                native_stage.release_ordinal_query(query).wait()
            camera.update(cfg.sim.dt)
            assert np.ptp(camera.data.output["rgba"].warp.numpy()) > 0
        finally:
            env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
