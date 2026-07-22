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
from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg

import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

pytestmark = [pytest.mark.integration, pytest.mark.rendering]

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


@pytest.fixture(scope="function")
def setup_pose_sim(device):
    """Set up a local-only RTX scene for the tensor-pose regression test."""
    sim_utils.create_new_stage()
    dt = 0.01
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=device, gravity=(0.0, 0.0, 0.0)))
    light_cfg = sim_utils.DomeLightCfg(intensity=DOME_LIGHT_INTENSITY, color=(1.0, 1.0, 1.0))
    light_cfg.func("/World/Light", light_cfg)
    sim_utils.update_stage()
    yield sim, dt
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


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.isaacsim_ci
def test_tensor_pose_is_visible_in_first_camera_frame(setup_pose_sim, device):
    """A tensor pose write after reset must be visible without a public physics step."""
    sim, dt = setup_pose_sim
    cube = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Objects/MovingCube",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=PhysxRigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=PhysxCollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 0.2)),
        )
    )
    camera = Camera(
        CameraCfg(
            height=HEIGHT,
            width=WIDTH,
            offset=CameraCfg.OffsetCfg(pos=(0.5, 0.0, 2.0), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
            prim_path="/World/PoseCamera",
            update_period=0,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
        )
    )

    sim.reset()

    root_pose = cube.data.default_root_pose.torch.clone()
    root_pose[:, :3] = torch.tensor((0.5, 0.0, 0.2), device=sim.device)
    cube.write_root_pose_to_sim_index(root_pose=root_pose)
    sim.render()

    camera.update(dt)
    first_frame = camera.data.output["rgb"].torch[0].clone().to(dtype=torch.float32)
    first_frame_step = sim.get_physics_step_count()

    sim.step()
    camera.update(dt)
    stable_frame = camera.data.output["rgb"].torch[0].clone().to(dtype=torch.float32)

    def red_fraction(frame: torch.Tensor) -> float:
        red, green, blue = frame.unbind(dim=-1)
        red_pixels = (red > 40.0) & (red > 1.5 * green) & (red > 1.5 * blue)
        return red_pixels.float().mean().item()

    first_red_fraction = red_fraction(first_frame)
    stable_red_fraction = red_fraction(stable_frame)

    assert first_frame_step == 0
    assert stable_red_fraction > 0.005, "The moved red cube is not visible in the stable reference frame."
    assert first_red_fraction > 0.5 * stable_red_fraction, (
        f"The tensor pose is stale in the first frame: first red fraction={first_red_fraction:.6f}, "
        f"stable red fraction={stable_red_fraction:.6f}."
    )

    del camera
    del cube


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
