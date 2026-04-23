# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Tests for the deprecated TiledCamera / TiledCameraCfg aliases.

TiledCamera is now a thin subclass of Camera that emits a DeprecationWarning.
All substantive Camera tests live in ``test_camera.py``. This file only verifies
that the deprecation mechanism works correctly and that TiledCamera remains a
functional Camera alias.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import random
import warnings

import numpy as np
import pytest
import torch

import omni.replicator.core as rep
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg, TiledCamera, TiledCameraCfg


@pytest.fixture(scope="function")
def setup_camera(device) -> tuple[sim_utils.SimulationContext, CameraCfg, float]:
    """Fixture to set up and tear down the camera simulation environment."""
    camera_cfg = CameraCfg(
        height=128,
        width=256,
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 4.0), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
        prim_path="/World/Camera",
        update_period=0,
        data_types=["rgb", "distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    # Create a new stage
    sim_utils.create_new_stage()
    # Simulation time-step
    dt = 0.01
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
    sim: sim_utils.SimulationContext = sim_utils.SimulationContext(sim_cfg)
    # populate scene
    _populate_scene()
    # load stage
    sim_utils.update_stage()
    yield sim, camera_cfg, dt
    # Teardown
    rep.vp_manager.destroy_hydra_textures("Replicator")
    sim.stop()
    sim.clear_instance()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_tiled_camera_deprecation_warning(setup_camera, device):
    """TiledCamera instantiation emits a DeprecationWarning."""
    sim, camera_cfg, dt = setup_camera
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        camera = TiledCamera(camera_cfg)
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "TiledCamera is deprecated" in str(deprecation_warnings[0].message)
    del camera


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_tiled_camera_cfg_deprecation_warning(setup_camera, device):
    """TiledCameraCfg instantiation emits a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _cfg = TiledCameraCfg(
            height=128,
            width=256,
            prim_path="/World/Camera",
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
        )
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "TiledCameraCfg is deprecated" in str(deprecation_warnings[0].message)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_tiled_camera_is_camera_subclass(setup_camera, device):
    """TiledCamera is a subclass of Camera, so isinstance checks work."""
    sim, camera_cfg, dt = setup_camera
    camera = TiledCamera(camera_cfg)
    assert isinstance(camera, Camera)
    assert isinstance(camera, TiledCamera)
    del camera


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_tiled_camera_basic_functionality(setup_camera, device):
    """TiledCamera produces correct output (proving it delegates to Camera)."""
    sim, camera_cfg, dt = setup_camera
    # Create camera
    camera = TiledCamera(camera_cfg)
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[0].GetPath().pathString == camera_cfg.prim_path
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (1, 3)
    assert camera.data.intrinsic_matrices.shape == (1, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(5):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for im_type, im_data in camera.data.output.items():
            if im_type == "rgb":
                assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 3)
                assert (im_data / 255.0).mean() > 0.0
            elif im_type == "distance_to_camera":
                assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 1)
                assert im_data.mean() > 0.0
    del camera


"""
Helper functions.
"""


def _populate_scene():
    """Add prims to the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.SphereLightCfg()
    cfg.func("/World/Light/GreySphere", cfg, translation=(4.5, 3.5, 10.0))
    cfg.func("/World/Light/WhiteSphere", cfg, translation=(-4.5, 3.5, 10.0))
    # Random objects
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    for i in range(10):
        # sample random position
        position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        position *= np.asarray([1.5, 1.5, 0.5])
        # create prim
        prim_type = random.choice(["Cube", "Sphere", "Cylinder"])
        prim = sim_utils.create_prim(
            f"/World/Objects/Obj_{i:02d}",
            prim_type,
            translation=position,
            scale=(0.25, 0.25, 0.25),
            semantic_label=prim_type,
        )
        # cast to geom prim
        geom_prim = getattr(UsdGeom, prim_type)(prim)
        # set random color
        color = Gf.Vec3f(random.random(), random.random(), random.random())
        geom_prim.CreateDisplayColorAttr()
        geom_prim.GetDisplayColorAttr().Set([color])
        # add rigid body and collision properties using Isaac Lab schemas
        prim_path = f"/World/Objects/Obj_{i:02d}"
        sim_utils.define_rigid_body_properties(prim_path, sim_utils.RigidBodyPropertiesCfg())
        sim_utils.define_mass_properties(prim_path, sim_utils.MassPropertiesCfg(mass=5.0))
        sim_utils.define_collision_properties(prim_path, sim_utils.CollisionPropertiesCfg())


# ------------------------------------------------------------------
# Camera pose → render validation (PrepareForReuse / Fabric path)
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "device, camera_cls",
    [
        pytest.param("cpu", TiledCamera, id="cpu-tiled"),
        pytest.param("cpu", Camera, id="cpu-non_tiled"),
        pytest.param("cuda:0", TiledCamera, id="cuda:0-tiled"),
        pytest.param("cuda:0", Camera, id="cuda:0-non_tiled"),
    ],
)
def test_camera_pose_update_reflected_in_render(setup_camera, device, camera_cls):
    """Camera pose changes via FrameView should be visible in rendered depth.

    Moves camera close then far, renders depth, and verifies that the mean
    valid depth from the far position is significantly larger (>1.5×) than
    the close position.  This validates that Fabric-side pose writes
    (via PrepareForReuse) or USD writes are correctly propagated to the
    RTX renderer.
    """
    sim, _unused_cam_cfg, dt = setup_camera

    cam_cfg = CameraCfg(
        prim_path="/World/PoseTestCam",
        height=128,
        width=256,
        update_period=0,
        update_latest_camera_pose=True,
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    camera = camera_cls(cam_cfg)
    sim.reset()

    target = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=camera.device)
    max_range = cam_cfg.spawn.clipping_range[1]

    # -- close position --
    eyes_close = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32, device=camera.device)
    camera.set_world_poses_from_view(eyes_close, target)
    sim.step()
    camera.update(dt)
    depth_close = camera.data.output["distance_to_camera"].clone()

    # -- far position --
    eyes_far = torch.tensor([[8.0, 8.0, 8.0]], dtype=torch.float32, device=camera.device)
    camera.set_world_poses_from_view(eyes_far, target)
    sim.step()
    camera.update(dt)
    depth_far = camera.data.output["distance_to_camera"].clone()

    # -- validate --
    valid_close = depth_close[depth_close < max_range]
    valid_far = depth_far[depth_far < max_range]

    assert valid_close.numel() > 0, "No valid close-range depth pixels"
    assert valid_far.numel() > 0, "No valid far-range depth pixels"

    mean_close = valid_close.mean().item()
    mean_far = valid_far.mean().item()

    assert mean_far > mean_close * 1.5, (
        f"Far depth ({mean_far:.2f}) should be > 1.5× close depth ({mean_close:.2f}). "
        "Camera pose change may not be reaching the renderer."
    )
    del camera
