# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Tests for the deprecated TiledCamera / TiledCameraCfg aliases.

TiledCamera is now a thin subclass of Camera that emits a DeprecationWarning.
All substantive Camera tests live in ``test_camera.py``. This file only verifies
that the deprecation mechanism works correctly and that TiledCamera remains an
initializable Camera alias.
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
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = [pytest.mark.integration, pytest.mark.rendering]


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


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
@pytest.mark.isaacsim_ci
def test_tiled_camera_deprecation_warning(setup_camera, device):
    """TiledCamera instantiation emits a DeprecationWarning and yields a working Camera."""
    sim, camera_cfg, dt = setup_camera
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        camera = TiledCamera(camera_cfg)
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "TiledCamera is deprecated" in str(deprecation_warnings[0].message)
    assert isinstance(camera, Camera)
    # the alias must still run Camera initialization
    sim.reset()
    assert camera.is_initialized
    del camera


def test_tiled_camera_cfg_deprecation_warning():
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
        sim_utils.apply_rigid_body_properties(prim_path, [sim_utils.UsdPhysicsRigidBodyCfg()], create_if_missing=True)
        sim_utils.apply_mass_properties(prim_path, [sim_utils.MassCfg(mass=5.0)], create_if_missing=True)
        sim_utils.apply_collision_properties(prim_path, [sim_utils.UsdPhysicsCollisionCfg()], create_if_missing=True)
