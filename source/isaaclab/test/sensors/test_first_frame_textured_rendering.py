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

import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# resolution
HEIGHT = 256
WIDTH = 256

# grey default-material detection: channels within this tolerance and mean below threshold
GREY_CHANNEL_TOLERANCE = 3.0
GREY_MEAN_THRESHOLD = 85.0

# number of extra sim steps before capturing the stabilised reference frame
STABILISATION_STEPS = 5

# max allowed per-channel mean difference between first and stabilised frames
FRAME_CONSISTENCY_THRESHOLD = 15.0

# scene: dome light
DOME_LIGHT_INTENSITY = 3000.0

# scene: textured cube pose
CUBE_TRANSLATION = (0.0, 0.0, 0.6)
CUBE_ORIENTATION = (0.7071, 0.0, 0.7071, 0.0)  # rotate DexCube with its yellow "E" face texture up
CUBE_SCALE = (0.9, 0.9, 0.9)


def _is_grey(mean_rgb: torch.Tensor) -> bool:
    """Return True if mean_rgb looks like the grey default material."""
    channels_equal = (mean_rgb[1] - mean_rgb[0]).abs() < GREY_CHANNEL_TOLERANCE and (
        mean_rgb[2] - mean_rgb[0]
    ).abs() < GREY_CHANNEL_TOLERANCE
    all_low = mean_rgb.mean() < GREY_MEAN_THRESHOLD
    return bool(channels_equal and all_low)


@pytest.fixture(scope="function")
def setup_sim(device):
    """Fixture to set up and tear down the textured rendering test environment."""
    # Create a new stage
    sim_utils.create_new_stage()
    # Simulation time-step
    dt = 0.01
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # populate scene
    _populate_scene()
    # load stage
    sim_utils.update_stage()
    yield sim, dt
    # Teardown
    rep.vp_manager.destroy_hydra_textures("Replicator")
    sim.stop()
    sim.clear_instance()


def _assert_first_frame_textured(first_frame: torch.Tensor, stable_frame: torch.Tensor):
    """Verify that first_frame shows loaded textures and is consistent with stable_frame."""
    mean_first = first_frame.mean(dim=(0, 1))
    mean_stable = stable_frame.mean(dim=(0, 1))
    # Guard 1: not the grey default material
    assert not _is_grey(mean_first), (
        f"First frame looks like the grey default material "
        f"(mean RGB: {mean_first[0]:.1f}, {mean_first[1]:.1f}, {mean_first[2]:.1f}). "
        "The renderer's streaming wait (ensure_isaac_rtx_render_update) "
        "may not have completed texture loading before the first capture."
    )

    # Guard 2: first frame and stabilised frame are broadly consistent
    per_channel_diff = (mean_first - mean_stable).abs()
    assert per_channel_diff.max().item() < FRAME_CONSISTENCY_THRESHOLD, (
        f"First and stabilised frames differ too much per-channel "
        f"(max delta {per_channel_diff.max():.1f}, means: "
        f"first=({mean_first[0]:.1f}, {mean_first[1]:.1f}, {mean_first[2]:.1f}), "
        f"stable=({mean_stable[0]:.1f}, {mean_stable[1]:.1f}, {mean_stable[2]:.1f})). "
        "The first frame may not be fully textured."
    )


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.isaacsim_ci
def test_first_frame_is_textured_camera(setup_sim, device):
    """First RTX frame from a USD Camera must show loaded textures, not a grey placeholder."""
    sim, dt = setup_sim
    camera_cfg = CameraCfg(
        height=HEIGHT,
        width=WIDTH,
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.75), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
        prim_path="/World/Camera",
        update_period=0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    # Create camera
    camera = Camera(camera_cfg)

    sim.reset()

    # The first sim step + camera update should produce textured output
    sim.step()
    camera.update(dt)
    first_frame = camera.data.output["rgb"][0].clone().to(dtype=torch.float32)

    # Let the renderer stabilise, then capture the reference frame
    for _ in range(STABILISATION_STEPS):
        sim.step()
    camera.update(dt)
    stable_frame = camera.data.output["rgb"][0].clone().to(dtype=torch.float32)

    del camera

    _assert_first_frame_textured(first_frame, stable_frame)


"""
Helper functions.
"""


def _populate_scene():
    """Add prims to the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=DOME_LIGHT_INTENSITY, color=(1.0, 1.0, 1.0))
    cfg.func("/World/Light/Dome", cfg)
    # Textured cube rotated so yellow "E" face is visible
    sim_utils.create_prim(
        "/World/Objects/ReferenceCube",
        "Xform",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        translation=CUBE_TRANSLATION,
        orientation=CUBE_ORIENTATION,
        scale=CUBE_SCALE,
    )
