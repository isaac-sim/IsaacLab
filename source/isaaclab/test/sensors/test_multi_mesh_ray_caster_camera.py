# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import copy
from collections.abc import Callable

import numpy as np
import pytest
import torch

import omni.replicator.core as rep
from pxr import Gf

import isaaclab.cloner as lab_cloner
import isaaclab.sim as sim_utils
from isaaclab.cloner import ClonePlan
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCamera, MultiMeshRayCasterCameraCfg, patterns
from isaaclab.sim import PinholeCameraCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.terrains.utils import create_prim_from_mesh

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG
from isaaclab_assets.robots.spot import SPOT_CFG

# sample camera poses (quaternions in xyzw format)
POSITION = [2.5, 2.5, 2.5]
QUAT_ROS = [0.33985114, 0.82047325, -0.42470819, -0.17591989]
QUAT_OPENGL = [0.17591988, 0.42470818, 0.82047324, 0.33985113]
QUAT_WORLD = [-0.27984815, -0.1159169, 0.88047623, -0.3647052]
MESH_ID_GROUND = 0
MESH_ID_OBJECT = 1
MESH_ID_ROBOT_MIN = 2


def _assert_quat_close(actual, expected, **kwargs):
    """Assert quaternions match while allowing the equivalent negated representation."""
    if hasattr(actual, "torch"):
        actual = actual.torch
    if hasattr(expected, "torch"):
        expected = expected.torch
    actual = torch.as_tensor(actual)
    expected = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
    expected = torch.where((actual * expected).sum(dim=-1, keepdim=True) < 0.0, -expected, expected)
    torch.testing.assert_close(actual, expected, **kwargs)


@pytest.fixture(scope="function")
def setup_simulation():
    """Fixture to set up and tear down the simulation environment."""
    # Create a new stage
    sim_utils.create_new_stage()
    # Simulation time-step
    dt = 0.01
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=dt)
    sim: sim_utils.SimulationContext = sim_utils.SimulationContext(sim_cfg)
    # Ground-plane
    mesh = make_plane(size=(100, 100), height=0.0, center_zero=True)
    create_prim_from_mesh("/World/defaultGroundPlane", mesh)
    # load stage
    sim_utils.update_stage()

    camera_cfg = MultiMeshRayCasterCameraCfg(
        prim_path="/World/Camera",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        update_period=0,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        debug_vis=False,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=480,
            width=640,
        ),
        data_types=["distance_to_image_plane"],
    )

    # create xform because placement of camera directly under world is not supported
    sim_utils.create_prim("/World/Camera", "Xform")

    yield sim, dt, camera_cfg

    # Cleanup
    # close all the opened viewport from before.
    rep.vp_manager.destroy_hydra_textures("Replicator")
    # stop simulation
    sim.stop()
    # clear the stage
    sim.clear_instance()


@pytest.mark.parametrize(
    "convention,quat",
    [
        ("ros", QUAT_ROS),
        ("opengl", QUAT_OPENGL),
        ("world", QUAT_WORLD),
    ],
)
@pytest.mark.isaacsim_ci
def test_camera_init_offset(setup_simulation, convention, quat):
    """Test camera initialization with offset using different conventions."""
    sim, dt, camera_cfg = setup_simulation

    # Create camera config with specific convention
    cam_cfg_offset = copy.deepcopy(camera_cfg)
    cam_cfg_offset.offset = MultiMeshRayCasterCameraCfg.OffsetCfg(
        pos=POSITION,
        rot=quat,
        convention=convention,
    )
    sim_utils.create_prim(f"/World/CameraOffset{convention.capitalize()}", "Xform")
    cam_cfg_offset.prim_path = f"/World/CameraOffset{convention.capitalize()}"

    camera = MultiMeshRayCasterCamera(cam_cfg_offset)

    # play sim
    sim.reset()

    # update camera
    camera.update(dt)

    # check that transform is set correctly
    np.testing.assert_allclose(camera.data.pos_w.torch[0].cpu().numpy(), cam_cfg_offset.offset.pos)

    del camera


@pytest.mark.isaacsim_ci
def test_camera_init(setup_simulation):
    """Test camera initialization."""
    sim, dt, camera_cfg = setup_simulation

    # Create camera
    camera = MultiMeshRayCasterCamera(cfg=camera_cfg)
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.torch.shape == (1, 3)
    assert camera.data.quat_w_ros.torch.shape == (1, 4)
    assert camera.data.quat_w_world.torch.shape == (1, 4)
    assert camera.data.quat_w_opengl.torch.shape == (1, 4)
    assert camera.data.intrinsic_matrices.torch.shape == (1, 3, 3)
    assert camera.data.image_shape == (camera_cfg.pattern_cfg.height, camera_cfg.pattern_cfg.width)
    assert camera.data.info == {camera_cfg.data_types[0]: None}
    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for im_data in camera.data.output.values():
            assert im_data.shape == (1, camera_cfg.pattern_cfg.height, camera_cfg.pattern_cfg.width, 1)

    del camera


@pytest.mark.isaacsim_ci
def test_camera_resolution(setup_simulation):
    """Test camera resolution is correctly set."""
    sim, dt, camera_cfg = setup_simulation

    # Create camera
    camera = MultiMeshRayCasterCamera(cfg=camera_cfg)
    # Play sim
    sim.reset()
    camera.update(dt)
    # access image data and compare shapes
    for im_data in camera.data.output.values():
        assert im_data.shape == (1, camera_cfg.pattern_cfg.height, camera_cfg.pattern_cfg.width, 1)

    del camera


@pytest.mark.isaacsim_ci
def test_camera_init_intrinsic_matrix(setup_simulation):
    """Test camera initialization from intrinsic matrix."""
    sim, dt, camera_cfg = setup_simulation

    # get the first camera
    camera_1 = MultiMeshRayCasterCamera(cfg=camera_cfg)
    # get intrinsic matrix
    sim.reset()
    intrinsic_matrix = camera_1.data.intrinsic_matrices.torch[0].cpu().flatten().tolist()

    # initialize from intrinsic matrix
    intrinsic_camera_cfg = MultiMeshRayCasterCameraCfg(
        prim_path="/World/Camera",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        update_period=0,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        debug_vis=False,
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic_matrix,
            height=camera_cfg.pattern_cfg.height,
            width=camera_cfg.pattern_cfg.width,
            focal_length=camera_cfg.pattern_cfg.focal_length,
        ),
        data_types=["distance_to_image_plane"],
    )
    camera_2 = MultiMeshRayCasterCamera(cfg=intrinsic_camera_cfg)

    # play sim
    sim.reset()
    sim.play()

    # update cameras
    camera_1.update(dt)
    camera_2.update(dt)

    # check image data
    torch.testing.assert_close(
        camera_1.data.output["distance_to_image_plane"].torch,
        camera_2.data.output["distance_to_image_plane"].torch,
    )
    # check that both intrinsic matrices are the same
    torch.testing.assert_close(
        camera_1.data.intrinsic_matrices.torch[0],
        camera_2.data.intrinsic_matrices.torch[0],
    )

    del camera_1, camera_2


@pytest.mark.isaacsim_ci
def test_multi_camera_init(setup_simulation):
    """Test multi-camera initialization."""
    sim, dt, camera_cfg = setup_simulation

    # -- camera 1
    cam_cfg_1 = copy.deepcopy(camera_cfg)
    cam_cfg_1.prim_path = "/World/Camera_0"
    sim_utils.create_prim("/World/Camera_0", "Xform")
    # Create camera
    cam_1 = MultiMeshRayCasterCamera(cam_cfg_1)

    # -- camera 2
    cam_cfg_2 = copy.deepcopy(camera_cfg)
    cam_cfg_2.prim_path = "/World/Camera_1"
    sim_utils.create_prim("/World/Camera_1", "Xform")
    # Create camera
    cam_2 = MultiMeshRayCasterCamera(cam_cfg_2)

    # play sim
    sim.reset()

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        cam_1.update(dt)
        cam_2.update(dt)
        # check image data
        for cam in [cam_1, cam_2]:
            for im_data in cam.data.output.values():
                assert im_data.shape == (1, camera_cfg.pattern_cfg.height, camera_cfg.pattern_cfg.width, 1)

    del cam_1, cam_2


@pytest.mark.isaacsim_ci
def test_camera_set_world_poses(setup_simulation):
    """Test camera function to set specific world pose."""
    sim, dt, camera_cfg = setup_simulation

    camera = MultiMeshRayCasterCamera(camera_cfg)
    # play sim
    sim.reset()

    # convert to torch tensors
    position = torch.tensor([POSITION], dtype=torch.float32, device=camera.device)
    orientation = torch.tensor([QUAT_WORLD], dtype=torch.float32, device=camera.device)
    # set new pose
    camera.set_world_poses(position.clone(), orientation.clone(), convention="world")

    # check if transform correctly set in output
    torch.testing.assert_close(camera.data.pos_w.torch, position)
    torch.testing.assert_close(camera.data.quat_w_world.torch, orientation)

    del camera


@pytest.mark.isaacsim_ci
def test_camera_set_world_poses_from_view(setup_simulation):
    """Test camera function to set specific world pose from view."""
    sim, dt, camera_cfg = setup_simulation

    camera = MultiMeshRayCasterCamera(camera_cfg)
    # play sim
    sim.reset()

    # convert to torch tensors
    eyes = torch.tensor([POSITION], dtype=torch.float32, device=camera.device)
    targets = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=camera.device)
    quat_ros_gt = torch.tensor([QUAT_ROS], dtype=torch.float32, device=camera.device)
    # set new pose
    camera.set_world_poses_from_view(eyes.clone(), targets.clone())

    # check if transform correctly set in output
    torch.testing.assert_close(camera.data.pos_w.torch, eyes)
    _assert_quat_close(camera.data.quat_w_ros.torch, quat_ros_gt)

    del camera


@pytest.mark.parametrize("height,width", [(240, 320), (480, 640)])
@pytest.mark.isaacsim_ci
def test_intrinsic_matrix(setup_simulation, height, width):
    """Checks that the camera's set and retrieve methods work for intrinsic matrix."""
    sim, dt, camera_cfg = setup_simulation

    camera_cfg_copy = copy.deepcopy(camera_cfg)
    camera_cfg_copy.pattern_cfg.height = height
    camera_cfg_copy.pattern_cfg.width = width
    camera = MultiMeshRayCasterCamera(camera_cfg_copy)
    # play sim
    sim.reset()
    # Desired properties (obtained from realsense camera at 320x240 resolution)
    rs_intrinsic_matrix = [229.31640625, 0.0, 164.810546875, 0.0, 229.826171875, 122.1650390625, 0.0, 0.0, 1.0]
    rs_intrinsic_matrix = torch.tensor(rs_intrinsic_matrix, device=camera.device).reshape(3, 3).unsqueeze(0)
    # Set matrix into simulator
    camera.set_intrinsic_matrices(rs_intrinsic_matrix.clone())
    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # Check that matrix is correct
        torch.testing.assert_close(rs_intrinsic_matrix, camera.data.intrinsic_matrices.torch)

    del camera


@pytest.mark.parametrize(
    "data_types",
    [
        ["distance_to_image_plane", "distance_to_camera", "normals"],
        ["distance_to_image_plane"],
        ["distance_to_camera"],
    ],
)
@pytest.mark.isaacsim_ci
def test_output_equal_to_usdcamera(setup_simulation, data_types):
    """Test that ray caster camera output equals USD camera output."""
    sim, dt, camera_cfg = setup_simulation

    camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
        height=240,
        width=320,
    )
    sim_utils.create_prim("/World/Camera_warp", "Xform")
    camera_cfg_warp = MultiMeshRayCasterCameraCfg(
        prim_path="/World/Camera_warp",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        update_period=0,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
        debug_vis=False,
        pattern_cfg=camera_pattern_cfg,
        data_types=data_types,
    )

    camera_warp = MultiMeshRayCasterCamera(camera_cfg_warp)

    # create usd camera
    camera_cfg_usd = CameraCfg(
        height=240,
        width=320,
        prim_path="/World/Camera_usd",
        update_period=0,
        data_types=data_types,
        spawn=PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-4, 1.0e5)
        ),
    )
    camera_usd = Camera(camera_cfg_usd)

    # play sim
    sim.reset()
    sim.play()

    eyes_np = np.asarray([[2.5, 2.5, 4.5]], dtype=np.float32)
    targets_np = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    eyes = torch.tensor(eyes_np, dtype=torch.float32, device=camera_warp.device)
    targets = torch.tensor(targets_np, dtype=torch.float32, device=camera_warp.device)
    # set views
    camera_warp.set_world_poses_from_view(eyes, targets)
    camera_usd.set_world_poses_from_view(eyes_np, targets_np)

    # perform steps
    for _ in range(5):
        sim.step()

    # update camera
    camera_usd.update(dt)
    camera_warp.update(dt)

    # check the intrinsic matrices
    torch.testing.assert_close(
        camera_usd.data.intrinsic_matrices.torch,
        camera_warp.data.intrinsic_matrices.torch,
    )

    # check the apertures
    torch.testing.assert_close(
        camera_usd._sensor_prims[0].GetHorizontalApertureAttr().Get(),
        camera_cfg_warp.pattern_cfg.horizontal_aperture,
    )

    # check image data
    for data_type in data_types:
        if data_type in camera_usd.data.output and data_type in camera_warp.data.output:
            if data_type == "distance_to_camera" or data_type == "distance_to_image_plane":
                torch.testing.assert_close(
                    camera_usd.data.output[data_type].torch,
                    camera_warp.data.output[data_type].torch,
                    atol=5e-5,
                    rtol=5e-6,
                )
            elif data_type == "normals":
                # NOTE: floating point issues of ~1e-5, so using atol and rtol in this case
                torch.testing.assert_close(
                    camera_usd.data.output[data_type].torch[..., :3],
                    camera_warp.data.output[data_type].torch,
                    rtol=1e-5,
                    atol=1e-4,
                )
            else:
                torch.testing.assert_close(
                    camera_usd.data.output[data_type].torch,
                    camera_warp.data.output[data_type].torch,
                )

    del camera_usd, camera_warp


def _create_heterogeneous_clone_scene(sim: sim_utils.SimulationContext, num_envs: int) -> torch.Tensor:
    """Create alternating Spot/ANYmal and cube/sphere cloned environments."""
    stage = sim_utils.get_current_stage()
    env_fmt = "/World/envs/env_{}"
    env_ids = torch.arange(num_envs, dtype=torch.long, device=sim.device)
    env_origins, _ = lab_cloner.grid_transforms(num_envs, spacing=4.0, device=sim.device)

    sim_utils.create_prim("/World/envs", "Xform", stage=stage)
    for env_id, origin in enumerate(env_origins.cpu().tolist()):
        sim_utils.create_prim(env_fmt.format(env_id), "Xform", translation=tuple(origin), stage=stage)
        sim_utils.create_prim(env_fmt.format(env_id) + "/RayCasterCamera", "Xform", stage=stage)

    robot_mask = torch.zeros((2, num_envs), dtype=torch.bool, device=sim.device)
    robot_mask[0, 0::2] = True
    robot_mask[1, 1::2] = True
    object_mask = robot_mask.clone()

    spot_spawn = copy.deepcopy(SPOT_CFG.spawn)
    anymal_spawn = copy.deepcopy(ANYMAL_C_CFG.spawn)
    spot_spawn.func(env_fmt.format(0) + "/Robot", spot_spawn, translation=SPOT_CFG.init_state.pos)
    anymal_spawn.func(env_fmt.format(1) + "/Robot", anymal_spawn, translation=ANYMAL_C_CFG.init_state.pos)

    cube_cfg = sim_utils.CuboidCfg(
        size=(0.35, 0.25, 0.25),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.2)),
    )
    sphere_cfg = sim_utils.SphereCfg(
        radius=0.18,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.7)),
    )
    cube_spawn = cube_cfg.func
    sphere_spawn = sphere_cfg.func
    assert isinstance(cube_spawn, Callable)
    assert isinstance(sphere_spawn, Callable)
    cube_spawn(env_fmt.format(0) + "/Object", cube_cfg, translation=(0.45, 0.0, 0.25))
    sphere_spawn(env_fmt.format(1) + "/Object", sphere_cfg, translation=(0.45, 0.0, 0.25))

    lab_cloner.usd_replicate(
        stage,
        [env_fmt.format(i) + f"/{asset_name}" for asset_name in ("Robot", "Object") for i in range(2)],
        [env_fmt + "/Robot", env_fmt + "/Robot", env_fmt + "/Object", env_fmt + "/Object"],
        env_ids,
        mask=torch.cat([robot_mask, object_mask], dim=0),
    )

    sim.set_clone_plan(
        ClonePlan(
            sources=(
                env_fmt.format(0) + "/Robot",
                env_fmt.format(1) + "/Robot",
                env_fmt.format(0) + "/Object",
                env_fmt.format(1) + "/Object",
            ),
            destinations=(
                env_fmt + "/Robot",
                env_fmt + "/Robot",
                env_fmt + "/Object",
                env_fmt + "/Object",
            ),
            clone_mask=torch.cat([robot_mask, object_mask], dim=0),
            env_ids=env_ids,
            positions=None,
            cfg_rows={},
        )
    )
    sim_utils.update_stage()
    return env_origins


@pytest.mark.isaacsim_ci
def test_depth_output_equal_to_usd_camera_heterogeneous_scene(setup_simulation):
    """Compare ray-caster and USD depth cameras in a heterogeneous cloned scene.

    The scene contains 16 environments with alternating Spot / ANYmal-C robot
    prototypes and alternating cube / sphere objects.  The ray-caster consumes
    the same clone plan used to build the USD scene and should match the batched
    USD camera's stable ``distance_to_image_plane`` pixels for every environment.
    """
    sim, dt, _ = setup_simulation
    num_envs = 16
    env_origins = _create_heterogeneous_clone_scene(sim, num_envs)

    height, width = 96, 128
    camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
        height=height,
        width=width,
    )
    mesh_prim_paths = [
        "/World/defaultGroundPlane",
        MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
            prim_expr="/World/envs/env_.*/Object",
            track_mesh_transforms=False,
        ),
        MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
            prim_expr="/World/envs/env_.*/Robot/.+",
            track_mesh_transforms=True,
        ),
    ]
    camera_cfg_warp = MultiMeshRayCasterCameraCfg(
        prim_path="/World/envs/env_.*/RayCasterCamera",
        mesh_prim_paths=mesh_prim_paths,
        update_period=0,
        debug_vis=False,
        pattern_cfg=camera_pattern_cfg,
        max_distance=25.0,
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        update_mesh_ids=True,
    )
    camera_warp = MultiMeshRayCasterCamera(camera_cfg_warp)

    camera_cfg_usd = CameraCfg(
        height=height,
        width=width,
        prim_path="/World/envs/env_.*/UsdCamera",
        update_period=0,
        data_types=["distance_to_image_plane"],
        spawn=PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 25.0),
        ),
    )
    camera_usd = Camera(camera_cfg_usd)

    sim.reset()
    sim.play()

    eyes = env_origins + torch.tensor((1.8, -2.5, 2.5), dtype=torch.float32, device=sim.device)
    targets = env_origins + torch.tensor((0.0, 0.0, 0.0), dtype=torch.float32, device=sim.device)
    camera_warp.set_world_poses_from_view(eyes=eyes, targets=targets)
    camera_usd.set_world_poses_from_view(eyes=eyes.cpu().numpy(), targets=targets.cpu().numpy())

    for _ in range(5):
        sim.render()

    camera_usd.update(dt)
    camera_warp.update(dt)

    ray_depth = camera_warp.data.output["distance_to_image_plane"].torch
    usd_depth = camera_usd.data.output["distance_to_image_plane"].torch
    assert ray_depth.shape == (num_envs, height, width, 1)
    assert usd_depth.shape == ray_depth.shape
    depth_diff = (ray_depth - usd_depth).abs()
    mesh_ids_proxy = getattr(camera_warp.data, "image_mesh_ids", None)
    assert mesh_ids_proxy is not None
    mesh_ids = mesh_ids_proxy.torch
    assert torch.any(mesh_ids == MESH_ID_OBJECT), "Expected object pixels in the heterogeneous scene"
    assert torch.any(mesh_ids >= MESH_ID_ROBOT_MIN), "Expected robot pixels in the heterogeneous scene"

    # The RTX and ray-cast backends can disagree by a pixel along complex robot
    # silhouettes.  Compare the stable ground pixels after dilating object/robot
    # edges and depth discontinuities.
    target_mask = mesh_ids[..., 0] != 0
    discontinuity_mask = torch.zeros_like(target_mask)
    for depth in (ray_depth, usd_depth):
        depth_image = depth[..., 0]
        discontinuity_mask[:, 1:, :] |= (depth_image[:, 1:, :] - depth_image[:, :-1, :]).abs() > 0.3
        discontinuity_mask[:, :, 1:] |= (depth_image[:, :, 1:] - depth_image[:, :, :-1]).abs() > 0.3
    edge_mask = target_mask | discontinuity_mask
    silhouette_mask = torch.nn.functional.max_pool2d(
        edge_mask[:, None, :, :].float(), kernel_size=21, stride=1, padding=10
    ).to(dtype=torch.bool)
    stable_mask = ~silhouette_mask[:, 0, :, :, None]
    assert stable_mask.float().mean() > 0.7
    stable_ray_depth = ray_depth[stable_mask]
    stable_usd_depth = usd_depth[stable_mask]
    stable_depth_diff = depth_diff[stable_mask]
    stable_close = torch.isclose(stable_ray_depth, stable_usd_depth, atol=5e-5, rtol=5e-6)
    assert stable_close.float().mean() > 0.999
    assert torch.quantile(stable_depth_diff, 0.999) < 5.0e-5

    del camera_usd, camera_warp


@pytest.mark.isaacsim_ci
def test_output_equal_to_usdcamera_offset(setup_simulation):
    """Test that ray caster camera output equals USD camera output with offset."""
    sim, dt, camera_cfg = setup_simulation
    offset_rot = (0.3617, 0.8731, -0.3020, -0.1251)

    camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
        height=240,
        width=320,
    )
    sim_utils.create_prim("/World/Camera_warp", "Xform")
    camera_cfg_warp = MultiMeshRayCasterCameraCfg(
        prim_path="/World/Camera_warp",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        update_period=0,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(2.5, 2.5, 4.0), rot=offset_rot, convention="ros"),
        debug_vis=False,
        pattern_cfg=camera_pattern_cfg,
        data_types=["distance_to_image_plane", "distance_to_camera", "normals"],
    )
    camera_warp = MultiMeshRayCasterCamera(camera_cfg_warp)

    # create usd camera
    camera_cfg_usd = CameraCfg(
        height=240,
        width=320,
        prim_path="/World/Camera_usd",
        update_period=0,
        data_types=["distance_to_image_plane", "distance_to_camera", "normals"],
        spawn=PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-6, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(2.5, 2.5, 4.0), rot=offset_rot, convention="ros"),
    )
    camera_usd = Camera(camera_cfg_usd)

    # play sim
    sim.reset()
    sim.play()

    # perform steps
    for _ in range(5):
        sim.step()

    # update camera
    camera_usd.update(dt)
    camera_warp.update(dt)

    # check image data
    torch.testing.assert_close(
        camera_usd.data.output["distance_to_image_plane"].torch,
        camera_warp.data.output["distance_to_image_plane"].torch,
        atol=5e-5,
        rtol=5e-6,
    )
    torch.testing.assert_close(
        camera_usd.data.output["distance_to_camera"].torch,
        camera_warp.data.output["distance_to_camera"].torch,
        atol=5e-5,
        rtol=5e-6,
    )

    # check normals
    # NOTE: floating point issues of ~1e-5, so using atol and rtol in this case
    torch.testing.assert_close(
        camera_usd.data.output["normals"].torch[..., :3],
        camera_warp.data.output["normals"].torch,
        rtol=1e-5,
        atol=1e-4,
    )

    del camera_usd, camera_warp


@pytest.mark.isaacsim_ci
def test_output_equal_to_usdcamera_prim_offset(setup_simulation):
    """Test that the output of the ray caster camera is equal to the output of the usd camera when both are placed
    under an XForm prim that is translated and rotated from the world origin."""
    sim, dt, camera_cfg = setup_simulation

    offset_rot = [0.3617, 0.8731, -0.3020, -0.1251]

    # gf quat (QUAT_OPENGL is xyzw, Gf.Quatd uses wxyz)
    gf_quatf = Gf.Quatd()
    gf_quatf.SetReal(QUAT_OPENGL[3])
    gf_quatf.SetImaginary(tuple(QUAT_OPENGL[:3]))

    camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
        height=240,
        width=320,
    )
    prim_raycast_cam = sim_utils.create_prim("/World/Camera_warp", "Xform")
    prim_raycast_cam.GetAttribute("xformOp:translate").Set(tuple(POSITION))
    prim_raycast_cam.GetAttribute("xformOp:orient").Set(gf_quatf)

    camera_cfg_warp = MultiMeshRayCasterCameraCfg(
        prim_path="/World/Camera_warp",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        update_period=0,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(0, 0, 2.0), rot=offset_rot, convention="ros"),
        debug_vis=False,
        pattern_cfg=camera_pattern_cfg,
        data_types=["distance_to_image_plane", "distance_to_camera", "normals"],
    )

    camera_warp = MultiMeshRayCasterCamera(camera_cfg_warp)

    # create usd camera
    camera_cfg_usd = CameraCfg(
        height=240,
        width=320,
        prim_path="/World/Camera_usd/camera",
        update_period=0,
        data_types=["distance_to_image_plane", "distance_to_camera", "normals"],
        spawn=PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-6, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0, 0, 2.0), rot=offset_rot, convention="ros"),
        update_latest_camera_pose=True,
    )
    prim_usd = sim_utils.create_prim("/World/Camera_usd", "Xform")
    prim_usd.GetAttribute("xformOp:translate").Set(tuple(POSITION))
    prim_usd.GetAttribute("xformOp:orient").Set(gf_quatf)

    camera_usd = Camera(camera_cfg_usd)

    # play sim
    sim.reset()
    sim.play()

    # perform steps
    for _ in range(5):
        sim.step()

    # update camera
    camera_usd.update(dt)
    camera_warp.update(dt)

    # check if pos and orientation are correct
    torch.testing.assert_close(camera_warp.data.pos_w.torch[0], camera_usd.data.pos_w.torch[0])
    _assert_quat_close(camera_warp.data.quat_w_ros.torch[0], camera_usd.data.quat_w_ros.torch[0])

    # check image data
    torch.testing.assert_close(
        camera_usd.data.output["distance_to_image_plane"].torch,
        camera_warp.data.output["distance_to_image_plane"].torch,
        atol=5e-5,
        rtol=5e-6,
    )
    torch.testing.assert_close(
        camera_usd.data.output["distance_to_camera"].torch,
        camera_warp.data.output["distance_to_camera"].torch,
        rtol=4e-6,
        atol=2e-5,
    )

    # check normals
    # NOTE: floating point issues of ~1e-5, so using atol and rtol in this case
    torch.testing.assert_close(
        camera_usd.data.output["normals"].torch[..., :3],
        camera_warp.data.output["normals"].torch,
        rtol=1e-5,
        atol=1e-4,
    )

    del camera_usd, camera_warp


@pytest.mark.parametrize("height,width", [(540, 960), (240, 320)])
@pytest.mark.isaacsim_ci
def test_output_equal_to_usd_camera_intrinsics(setup_simulation, height, width):
    """Test that the output of the ray caster camera and usd camera are the same when both are
    initialized with the same intrinsic matrix."""
    sim, dt, camera_cfg = setup_simulation

    # create cameras
    offset_rot = [0.3617, 0.8731, -0.3020, -0.1251]
    offset_pos = (2.5, 2.5, 4.0)
    intrinsics = [380.0831, 0.0, width / 2, 0.0, 380.0831, height / 2, 0.0, 0.0, 1.0]
    sim_utils.create_prim("/World/Camera_warp", "Xform")
    # get camera cfgs
    camera_warp_cfg = MultiMeshRayCasterCameraCfg(
        prim_path="/World/Camera_warp",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=offset_pos, rot=offset_rot, convention="ros"),
        debug_vis=False,
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsics,
            height=height,
            width=width,
            focal_length=38.0,
        ),
        max_distance=25.0,
        data_types=["distance_to_image_plane"],
    )
    camera_usd_cfg = CameraCfg(
        prim_path="/World/Camera_usd",
        offset=CameraCfg.OffsetCfg(pos=offset_pos, rot=offset_rot, convention="ros"),
        spawn=PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsics,
            height=height,
            width=width,
            clipping_range=(0.01, 25),
            focal_length=38.0,
        ),
        height=height,
        width=width,
        data_types=["distance_to_image_plane"],
    )

    # set aperture offsets to 0, as currently not supported for usd camera
    camera_warp_cfg.pattern_cfg.horizontal_aperture_offset = 0
    camera_warp_cfg.pattern_cfg.vertical_aperture_offset = 0
    camera_usd_cfg.spawn.horizontal_aperture_offset = 0
    camera_usd_cfg.spawn.vertical_aperture_offset = 0
    # init cameras
    camera_warp = MultiMeshRayCasterCamera(camera_warp_cfg)
    camera_usd = Camera(camera_usd_cfg)

    # play sim
    sim.reset()
    sim.play()

    # perform steps
    for _ in range(5):
        sim.step()

    # update camera
    camera_usd.update(dt)
    camera_warp.update(dt)

    # filter nan and inf from output
    cam_warp_output = camera_warp.data.output["distance_to_image_plane"].torch.clone()
    cam_usd_output = camera_usd.data.output["distance_to_image_plane"].torch.clone()
    cam_warp_output[torch.isnan(cam_warp_output)] = 0
    cam_warp_output[torch.isinf(cam_warp_output)] = 0
    cam_usd_output[torch.isnan(cam_usd_output)] = 0
    cam_usd_output[torch.isinf(cam_usd_output)] = 0

    # check that both have the same intrinsic matrices
    torch.testing.assert_close(
        camera_warp.data.intrinsic_matrices.torch[0], camera_usd.data.intrinsic_matrices.torch[0]
    )

    # check the apertures
    torch.testing.assert_close(
        camera_usd._sensor_prims[0].GetHorizontalApertureAttr().Get(),
        camera_warp_cfg.pattern_cfg.horizontal_aperture,
    )
    torch.testing.assert_close(
        camera_usd._sensor_prims[0].GetVerticalApertureAttr().Get(),
        camera_warp_cfg.pattern_cfg.vertical_aperture,
    )

    # check image data
    torch.testing.assert_close(
        cam_warp_output,
        cam_usd_output,
        atol=5e-5,
        rtol=1e-5,
    )

    del camera_usd, camera_warp


@pytest.mark.flaky(max_runs=3, min_passes=1)
@pytest.mark.isaacsim_ci
def test_output_equal_to_usd_camera_when_intrinsics_set(setup_simulation):
    """Test that the output of the ray caster camera is equal to the output of the usd camera when both are placed
    under an XForm prim and an intrinsic matrix is set."""
    sim, dt, camera_cfg = setup_simulation

    camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
        height=540,
        width=960,
    )
    camera_cfg_warp = MultiMeshRayCasterCameraCfg(
        prim_path="/World/Camera",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        update_period=0,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
        debug_vis=False,
        pattern_cfg=camera_pattern_cfg,
        data_types=["distance_to_camera"],
    )

    camera_warp = MultiMeshRayCasterCamera(camera_cfg_warp)

    # create usd camera
    camera_cfg_usd = CameraCfg(
        height=540,
        width=960,
        prim_path="/World/Camera_usd",
        update_period=0,
        data_types=["distance_to_camera"],
        spawn=PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-4, 1.0e5)
        ),
    )
    camera_usd = Camera(camera_cfg_usd)

    # play sim
    sim.reset()
    sim.play()

    # set intrinsic matrix
    # NOTE: extend the test to cover aperture offsets once supported by the usd camera
    intrinsic_matrix_np = np.asarray(
        [[380.0831, 0.0, camera_cfg_usd.width / 2, 0.0, 380.0831, camera_cfg_usd.height / 2, 0.0, 0.0, 1.0]]
    ).reshape(1, 3, 3)
    intrinsic_matrix = torch.tensor(intrinsic_matrix_np, device=camera_warp.device)
    camera_warp.set_intrinsic_matrices(intrinsic_matrix, focal_length=10)
    camera_usd.set_intrinsic_matrices(torch.tensor(intrinsic_matrix_np, device=camera_usd.device), focal_length=10)

    # set camera position
    camera_warp.set_world_poses_from_view(
        eyes=torch.tensor([[0.0, 0.0, 5.0]], device=camera_warp.device),
        targets=torch.tensor([[0.0, 0.0, 0.0]], device=camera_warp.device),
    )
    camera_usd.set_world_poses_from_view(
        eyes=np.asarray([[0.0, 0.0, 5.0]], dtype=np.float32),
        targets=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
    )

    # perform steps
    for _ in range(5):
        sim.step()

    # update camera
    camera_usd.update(dt)
    camera_warp.update(dt)

    # check image data
    torch.testing.assert_close(
        camera_usd.data.output["distance_to_camera"].torch,
        camera_warp.data.output["distance_to_camera"].torch,
        rtol=5e-3,
        atol=1e-4,
    )

    del camera_usd, camera_warp


@pytest.mark.isaacsim_ci
def test_image_mesh_ids_identifies_hit_mesh(setup_simulation):
    """image_mesh_ids must contain 0 for ground-plane hits (only one mesh registered)."""
    sim, dt, camera_cfg = setup_simulation

    cfg = copy.deepcopy(camera_cfg)
    cfg.update_mesh_ids = True
    cfg.data_types = ["distance_to_camera"]

    camera = MultiMeshRayCasterCamera(cfg=cfg)
    sim.reset()
    camera.update(dt)

    assert camera.data.image_mesh_ids is not None, "image_mesh_ids should not be None when update_mesh_ids=True"
    mesh_ids = camera.data.image_mesh_ids.torch  # shape (N, H, W, 1), dtype torch.int16
    assert mesh_ids.shape[-1] == 1
    assert mesh_ids.dtype == torch.int16

    # Identify actual hits via distance < inf. This relies on depth_clipping_behavior="none"
    # (the default), which leaves missed rays at the Warp-kernel fill value of inf.
    # Under "max" clipping, missed rays would be clamped to a finite max_distance, making
    # the inf comparison incorrect.
    hit_mask = camera.data.output["distance_to_camera"].torch[0, :, :, 0] < float("inf")
    assert hit_mask.any(), "Expected at least some rays to hit the ground plane"

    # All hits against the single registered mesh must carry the ground mesh id.
    hit_mesh_ids = mesh_ids[0, :, :, 0][hit_mask]
    assert torch.all(hit_mesh_ids == MESH_ID_GROUND), (
        f"All hits against the single ground mesh must have mesh_id={MESH_ID_GROUND}, got: {hit_mesh_ids.unique()}"
    )
