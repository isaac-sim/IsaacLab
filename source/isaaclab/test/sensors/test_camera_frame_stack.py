# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

NUM_CAMERAS = 4
CAMERA_HEIGHT = 64
CAMERA_WIDTH = 64


def _make_camera_cfg(frame_stack: int = 1) -> CameraCfg:
    return CameraCfg(
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 4.0), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
        prim_path="/World/Origin_.*/CameraSensor",
        update_period=0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        frame_stack=frame_stack,
    )


def _populate_scene():
    """Add minimal prims to the scene."""
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.SphereLightCfg()
    cfg.func("/World/Light", cfg, translation=(0.0, 0.0, 10.0))
    for i in range(NUM_CAMERAS):
        sim_utils.create_prim(f"/World/Origin_{i}", "Xform")
        sim_utils.create_prim(
            f"/World/Origin_{i}/Cube",
            "Cube",
            translation=(0.0, 0.0, 1.0),
            scale=(0.5, 0.5, 0.5),
        )


@pytest.fixture(scope="function")
def setup_scene():
    """Set up a minimal simulation scene and tear it down after the test."""
    sim_utils.create_new_stage()
    dt = 0.01
    sim_cfg = sim_utils.SimulationCfg(dt=dt)
    sim = sim_utils.SimulationContext(sim_cfg)
    _populate_scene()
    sim_utils.update_stage()
    yield sim
    sim.clear_instance()


# -- Imports needed by _populate_scene but deferred until after AppLauncher --
from pxr import Gf, UsdGeom  # noqa: E402


@pytest.mark.isaacsim_ci
def test_frame_stack_1_preserves_shape(setup_scene):
    """frame_stack=1 should produce the standard (N, H, W, 3) output."""
    sim = setup_scene
    camera = Camera(_make_camera_cfg(frame_stack=1))
    sim.reset()
    camera.update(dt=0.01)

    rgb = camera.data.output["rgb"]
    assert rgb.shape == (NUM_CAMERAS, CAMERA_HEIGHT, CAMERA_WIDTH, 3), f"Expected (N, H, W, 3), got {rgb.shape}"


@pytest.mark.isaacsim_ci
def test_frame_stack_2_doubles_channels(setup_scene):
    """frame_stack=2 should produce (N, H, W, 6) output."""
    sim = setup_scene
    camera = Camera(_make_camera_cfg(frame_stack=2))
    sim.reset()
    # Step twice to fill the buffer
    camera.update(dt=0.01)
    camera.update(dt=0.01)

    rgb = camera.data.output["rgb"]
    assert rgb.shape == (NUM_CAMERAS, CAMERA_HEIGHT, CAMERA_WIDTH, 6), f"Expected (N, H, W, 6), got {rgb.shape}"


@pytest.mark.isaacsim_ci
def test_frame_stack_init_fills_history(setup_scene):
    """On first update, all history slots should be filled with the same frame."""
    sim = setup_scene
    camera = Camera(_make_camera_cfg(frame_stack=2))
    sim.reset()
    camera.update(dt=0.01)

    rgb = camera.data.output["rgb"]
    # With 2-frame stack, channels 0:3 and 3:6 should be identical (same frame copied)
    first_frame = rgb[..., :3]
    second_frame = rgb[..., 3:6]
    assert torch.equal(first_frame, second_frame), "After first update, all history slots should contain the same frame"


@pytest.mark.isaacsim_ci
def test_frame_stack_ring_buffer_shifts_correctly(setup_scene):
    """After a camera move, the ring buffer should shift: old newest becomes new oldest."""
    sim = setup_scene
    camera = Camera(_make_camera_cfg(frame_stack=2))
    sim.reset()
    camera.update(dt=0.01)

    # Capture the pre-move newest frame (channels 3:6)
    rgb_before = camera.data.output["rgb"].clone()
    pre_move_newest = rgb_before[..., 3:6].clone()

    # Move the camera to guarantee a different rendered frame
    for prim in camera._sensor_prims:
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(2.0, 0.0, 4.0))

    sim.step()
    camera.update(dt=0.01)

    rgb_after = camera.data.output["rgb"]
    post_move_oldest = rgb_after[..., :3]
    post_move_newest = rgb_after[..., 3:6]

    # The pre-move newest frame should now be the post-move oldest frame
    assert torch.equal(pre_move_newest, post_move_oldest), (
        "Ring buffer should shift: previous newest frame becomes the new oldest"
    )
    # The new newest frame should differ (camera moved)
    assert not torch.equal(post_move_oldest, post_move_newest), (
        "After moving the camera, the newest frame should differ from the oldest"
    )


@pytest.mark.isaacsim_ci
def test_frame_stack_reset_clears_history(setup_scene):
    """After reset, history should be re-initialized with the current frame."""
    sim = setup_scene
    camera = Camera(_make_camera_cfg(frame_stack=2))
    sim.reset()

    # Step a few times to build up different history
    for _ in range(5):
        sim.step()
        camera.update(dt=0.01)

    # Reset specific envs
    env_ids = torch.tensor([0], dtype=torch.long)
    camera.reset(env_ids=env_ids)
    camera.update(dt=0.01)

    # After reset + update, env 0's history should be all identical (re-filled)
    rgb = camera.data.output["rgb"]
    first_frame_env0 = rgb[0, :, :, :3]
    second_frame_env0 = rgb[0, :, :, 3:6]
    assert torch.equal(first_frame_env0, second_frame_env0), (
        "After reset, env 0's history slots should contain the same frame"
    )


@pytest.mark.isaacsim_ci
def test_frame_stack_partial_reset_preserves_others(setup_scene):
    """Resetting env 0 should not affect env 1's history."""
    sim = setup_scene
    camera = Camera(_make_camera_cfg(frame_stack=2))
    sim.reset()

    # Step to build history
    for _ in range(3):
        sim.step()
        camera.update(dt=0.01)

    # Capture env 1's stacked output before reset
    rgb_before = camera.data.output["rgb"][1].clone()

    # Reset only env 0
    camera.reset(env_ids=torch.tensor([0], dtype=torch.long))
    camera.update(dt=0.01)

    # env 1 should still have progressed (new frame), not be reset
    rgb_after = camera.data.output["rgb"][1]
    # Shape should be preserved
    assert rgb_after.shape == rgb_before.shape


@pytest.mark.isaacsim_ci
def test_frame_stack_default_is_one():
    """CameraCfg should default to frame_stack=1."""
    cfg = CameraCfg()
    assert cfg.frame_stack == 1


# -- Newton warning and preset resolution tests --
# These require isaaclab_tasks for launch_simulation and preset configs.

import logging

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_tasks.utils.sim_launcher import launch_simulation

_warning_capture: list[str] = []


@pytest.fixture(autouse=True, scope="session")
def _install_launcher_warning_capture():
    """Capture warnings from sim_launcher logger."""

    class _Handler(logging.Handler):
        def handle(self, record):
            _warning_capture.append(record.getMessage())
            return True

    handler = _Handler(level=logging.WARNING)
    logger = logging.getLogger("isaaclab_tasks.utils.sim_launcher")
    logger.addHandler(handler)
    yield
    logger.removeHandler(handler)


@pytest.mark.isaacsim_ci
def test_newton_warning_fires_with_stack_1():
    """Newton physics + camera with frame_stack=1 should trigger warning."""
    import sys

    old_argv = sys.argv
    sys.argv = [sys.argv[0], "presets=newton"]
    try:
        env_cfg, _ = resolve_task_config("Isaac-Cartpole-Camera-Presets-Direct-v0", "skrl_cfg_entry_point")
    finally:
        sys.argv = old_argv

    # Override frame_stack back to 1 (newton preset sets it to 2)
    if hasattr(env_cfg.tiled_camera, "frame_stack"):
        env_cfg.tiled_camera.frame_stack = 1

    _warning_capture.clear()
    with launch_simulation(env_cfg, {"enable_cameras": True}):
        pass

    assert any("frame_stack" in w for w in _warning_capture), "Expected Newton + frame_stack<=1 warning, got: " + str(
        _warning_capture
    )


@pytest.mark.isaacsim_ci
def test_newton_warning_does_not_fire_with_stack_2():
    """Newton physics + camera with frame_stack=2 should NOT trigger warning."""
    import sys

    old_argv = sys.argv
    sys.argv = [sys.argv[0], "presets=newton"]
    try:
        env_cfg, _ = resolve_task_config("Isaac-Cartpole-Camera-Presets-Direct-v0", "skrl_cfg_entry_point")
    finally:
        sys.argv = old_argv

    # Newton preset auto-sets frame_stack=2 via MultiBackendCameraCfg
    _warning_capture.clear()
    with launch_simulation(env_cfg, {"enable_cameras": True}):
        pass

    assert not any("frame_stack" in w for w in _warning_capture), (
        "Unexpected Newton frame_stack warning with frame_stack=2: " + str(_warning_capture)
    )


@pytest.mark.isaacsim_ci
def test_preset_newton_sets_frame_stack_2():
    """presets=newton should resolve frame_stack to 2 on MultiBackendCameraCfg."""
    import sys

    old_argv = sys.argv
    sys.argv = [sys.argv[0], "presets=newton"]
    try:
        env_cfg, _ = resolve_task_config("Isaac-Cartpole-Camera-Presets-Direct-v0", "skrl_cfg_entry_point")
    finally:
        sys.argv = old_argv

    cam = env_cfg.tiled_camera
    assert hasattr(cam, "frame_stack"), "Resolved camera should have frame_stack"
    assert cam.frame_stack == 2, f"Expected frame_stack=2 for newton, got {cam.frame_stack}"


@pytest.mark.isaacsim_ci
def test_preset_physx_keeps_frame_stack_1():
    """presets=physx should keep frame_stack at default 1."""
    import sys

    # Temporarily inject presets=physx into argv for resolve_task_config
    old_argv = sys.argv
    sys.argv = [sys.argv[0], "presets=physx"]
    try:
        env_cfg, _ = resolve_task_config("Isaac-Cartpole-Camera-Presets-Direct-v0", "skrl_cfg_entry_point")
    finally:
        sys.argv = old_argv

    cam = env_cfg.tiled_camera
    assert hasattr(cam, "frame_stack"), "Resolved camera should have frame_stack"
    assert cam.frame_stack == 1, f"Expected frame_stack=1 for physx, got {cam.frame_stack}"


@pytest.mark.isaacsim_ci
def test_physx_no_regression_output_shape(setup_scene):
    """PhysX with default frame_stack=1 should produce standard (N, H, W, 3) output."""
    sim = setup_scene
    camera = Camera(_make_camera_cfg(frame_stack=1))
    sim.reset()
    camera.update(dt=0.01)

    rgb = camera.data.output["rgb"]
    assert rgb.shape == (NUM_CAMERAS, CAMERA_HEIGHT, CAMERA_WIDTH, 3), (
        f"PhysX default should produce (N, H, W, 3), got {rgb.shape}"
    )
