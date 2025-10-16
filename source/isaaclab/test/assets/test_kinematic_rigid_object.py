# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils
import pytest

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


def generate_test_scene(
    num_cubes: int = 1,
    elongated_cube=False,
    height=1.0,
    spawn_robot=False,
    device: str = "cpu",
) -> tuple[RigidObject, torch.Tensor, Articulation | None]:
    """Build a minimal scene: N kinematic cubes (optionally tall) and an optional Franka.

    Notes:
    - Cubes are **kinematic**.
    - Each env lives under `/World/envs_{i}` which aligns with regex prim paths used by assets.
    - When `elongated_cube=True`, the Y scale is 12x to create visible vertical extent for camera tests.
    """
    origins = torch.tensor([(i * 1.0, 0, height) for i in range(num_cubes)]).to(device)
    # Create Top-level Xforms, one for each cube
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/envs_{i}", "Xform", translation=origin)

    # Create rigid object
    cube_object_cfg = RigidObjectCfg(
        prim_path="/World/envs_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            scale=(1.0, 12.0, 1.0) if elongated_cube else (1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height)),
    )
    cube_object = RigidObject(cfg=cube_object_cfg)

    robot = None
    if spawn_robot:
        robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs_.*/Robot")
        robot_cfg.init_state.pos = (-0.75, -0.75, 0.0)
        robot = Articulation(robot_cfg)

    return cube_object, origins, robot


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("spawn_robot", [True, False])
def test_kinematic_enabled_rigidbody_scales_correctly(device, spawn_robot):
    """Rendering sanity for scaled geometry:

    With an elongated cube, the top row of the image (same column as the optical center)
    should have **finite** distance (i.e., the object actually occupies those pixels).

    We check this via `depth < 1.0`.
    """
    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:

        NUM_CUBES = 2
        sim._app_control_on_stop_handle = None

        # Generate cubes scene
        cube_object, _, _ = generate_test_scene(
            num_cubes=NUM_CUBES, device=device, spawn_robot=spawn_robot, elongated_cube=True
        )
        camera = Camera(
            CameraCfg(
                height=120,
                width=160,
                prim_path="/World/envs_.*/Camera",
                update_period=0,
                data_types=["distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
                ),
            )
        )
        # Play sim
        sim.reset()

        # Local helper to step + update + fetch depth
        def _render_depth():
            sim.step()
            cube_object.update(sim.cfg.dt)
            camera.update(sim.cfg.dt)
            return camera.data.output["distance_to_image_plane"]

        # 1) object rendered at center of camera FOV, but because it is elongated, we should see finite depth
        # at top pixel in camera.
        depth = _render_depth()
        assert torch.all(depth[:, 0, camera.cfg.width // 2, 0] < 1.0)


@pytest.mark.parametrize(
    "device, spawn_robot",
    [
        ("cpu", True),
        ("cpu", False),
        ("cuda", True),
        pytest.param("cuda", False, marks=pytest.mark.skip(reason="Fails until illegal memory access issue is fixed")),
    ],
)
def test_kinematic_enabled_rigidbody_set_transform_correctly_rendered(device, spawn_robot):
    """Spatial coverage sanity:

    Move the cube within the image plane and verify depth at the center/left/right pixels is finite.
    This ensures:
      1) set_transforms() actually changes pose for each env instance,
      2) rendering pipeline works well to produce expected image reflected by change in transform,
    """
    with build_simulation_context(device=device, add_ground_plane=True, auto_add_lighting=True) as sim:
        CAMERA_FAR_LEFT_XY = [-0.425, 0.0]
        CAMERA_FAR_RIGHT_XY = [0.425, 0.0]
        NUM_CUBES = 2
        sim._app_control_on_stop_handle = None

        # Generate cubes scene
        cube_object, _, _ = generate_test_scene(num_cubes=NUM_CUBES, device=device, spawn_robot=spawn_robot)
        camera = Camera(
            CameraCfg(
                height=120,
                width=160,
                prim_path="/World/envs_.*/Camera",
                update_period=0,
                data_types=["distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
                ),
            )
        )
        sanity_tensor = torch.ones(10, device=device)
        env_idx = torch.arange(NUM_CUBES, device=device)

        # Play sim
        sim.reset()
        center_transform = cube_object.root_physx_view.get_transforms().clone()

        # Local helper to step + update + fetch depth
        def _render_depth():
            sim.step()
            cube_object.update(sim.cfg.dt)
            camera.update(sim.cfg.dt)
            return camera.data.output["distance_to_image_plane"]

        # 1) object rendered at center of camera FOV
        depth = _render_depth()
        assert torch.all(depth[:, camera.cfg.height // 2, camera.cfg.width // 2, 0].tanh() < 1.0)

        # 2) object rendered at far left of camera FOV
        far_left_transform = center_transform.clone()
        far_left_transform[:, :2] += torch.tensor(CAMERA_FAR_LEFT_XY, device=device).repeat(NUM_CUBES, 1)
        cube_object.root_physx_view.set_transforms(far_left_transform, indices=env_idx)
        assert torch.all(sanity_tensor == 1.0)  # torch not in bad state
        depth = _render_depth()
        assert torch.all(depth[:, camera.cfg.height // 2, 0, 0].tanh() < 1.0)

        # 3) object rendered at far right of camera FOV
        far_right_transform = center_transform.clone()
        far_right_transform[:, :2] += torch.tensor(CAMERA_FAR_RIGHT_XY, device=device).repeat(NUM_CUBES, 1)
        cube_object.root_physx_view.set_transforms(far_right_transform, indices=env_idx)
        assert torch.all(sanity_tensor == 1.0)  # torch not in bad state
        depth = _render_depth()
        assert torch.all(depth[:, camera.cfg.height // 2, -1, 0].tanh() < 1.0)
