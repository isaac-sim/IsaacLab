# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for :class:`RayCasterCamera` and :class:`MultiMeshRayCasterCamera`.

Both camera classes share the same interface, so every test runs against each of them through the
``camera_type`` fixture. The USD :class:`Camera` is used as the rendering reference.
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import copy

import numpy as np
import pytest
import torch

import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab import cloner as lab_cloner
from isaaclab.cloner import ClonePlan
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.ray_caster import (
    MultiMeshRayCasterCamera,
    MultiMeshRayCasterCameraCfg,
    RayCasterCamera,
    RayCasterCameraCfg,
    patterns,
)
from isaaclab.sim import PinholeCameraCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.terrains.utils import create_prim_from_mesh

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG
from isaaclab_assets.robots.spot import SPOT_CFG

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.isaacsim_ci]

GROUND_PATH = "/World/defaultGroundPlane"
DT = 0.01
HEIGHT, WIDTH = 240, 320
FOCAL_LENGTH, HORIZONTAL_APERTURE = 24.0, 20.955

# sample camera pose expressed in each convention (quaternions in xyzw)
POSITION = (2.5, 2.5, 2.5)
QUAT_ROS = (0.33985114, 0.82047325, -0.42470819, -0.17591989)
QUAT_OPENGL = (0.17591988, 0.42470818, 0.82047324, 0.33985113)
QUAT_WORLD = (-0.27984815, -0.1159169, 0.88047623, -0.3647052)
OFFSET_ROT_ROS = (0.3617, 0.8731, -0.3020, -0.1251)

CAMERA_TYPES = {
    "single_mesh": (RayCasterCamera, RayCasterCameraCfg),
    "multi_mesh": (MultiMeshRayCasterCamera, MultiMeshRayCasterCameraCfg),
}


def _assert_quat_close(actual, expected, **kwargs):
    """Assert quaternions match while allowing the equivalent negated representation."""
    actual = torch.as_tensor(actual.torch if hasattr(actual, "torch") else actual)
    expected = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
    expected = torch.where((actual * expected).sum(dim=-1, keepdim=True) < 0.0, -expected, expected)
    torch.testing.assert_close(actual, expected, **kwargs)


def _pattern_cfg(height: int = HEIGHT, width: int = WIDTH) -> patterns.PinholeCameraPatternCfg:
    return patterns.PinholeCameraPatternCfg(
        focal_length=FOCAL_LENGTH, horizontal_aperture=HORIZONTAL_APERTURE, height=height, width=width
    )


def _usd_camera_cfg(prim_path: str, data_types: list[str], height: int = HEIGHT, width: int = WIDTH, **kwargs):
    return CameraCfg(
        prim_path=prim_path,
        height=height,
        width=width,
        update_period=0,
        data_types=data_types,
        spawn=PinholeCameraCfg(
            focal_length=FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=HORIZONTAL_APERTURE,
            clipping_range=(1e-4, 1.0e5),
        ),
        **kwargs,
    )


@pytest.fixture(params=list(CAMERA_TYPES))
def camera_type(request):
    return request.param


@pytest.fixture
def setup_sim(camera_type):
    """A stage with a ground plane and a base camera config for the requested ray caster camera class."""
    camera_cls, cfg_cls = CAMERA_TYPES[camera_type]
    camera_cfg = cfg_cls(
        prim_path="/World/Camera",
        mesh_prim_paths=[GROUND_PATH],
        update_period=0,
        offset=cfg_cls.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        pattern_cfg=_pattern_cfg(),
        data_types=["distance_to_image_plane"],
    )
    sim_utils.create_new_stage()
    # cameras cannot be placed directly under /World
    sim_utils.create_prim("/World/Camera", "Xform")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT))
    # the ground plane needs a visual material and a light so the reference RTX camera renders it
    mesh = make_plane(size=(100, 100), height=0.0, center_zero=True)
    create_prim_from_mesh(GROUND_PATH, mesh, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)))
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)
    sim_utils.update_stage()
    yield sim, camera_cls, camera_cfg
    rep.vp_manager.destroy_hydra_textures("Replicator")
    sim.stop()
    sim.clear_instance()


def _make_camera(camera_cls, camera_cfg, prim_path: str | None = None, create_prim: bool = True, **overrides):
    """Create a ray caster camera from a copy of ``camera_cfg``, mounted on a fresh Xform unless it exists."""
    cfg = copy.deepcopy(camera_cfg)
    if prim_path is not None:
        if create_prim:
            sim_utils.create_prim(prim_path, "Xform")
        cfg.prim_path = prim_path
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return camera_cls(cfg)


"""
Initialization and buffers
"""


def test_camera_init(setup_sim):
    """Buffers, image shapes, shared meshes, frame counters, and ``str()`` for two cameras."""
    sim, camera_cls, camera_cfg = setup_sim
    camera = camera_cls(camera_cfg)
    other = _make_camera(camera_cls, camera_cfg, "/World/Camera_2", pattern_cfg=_pattern_cfg(120, 160))
    sim.reset()

    assert camera.is_initialized and other.is_initialized
    assert camera.meshes is other.meshes
    assert camera.data.pos_w.torch.shape == (1, 3)
    for quat in (camera.data.quat_w_ros, camera.data.quat_w_world, camera.data.quat_w_opengl):
        assert quat.torch.shape == (1, 4)
    assert camera.data.intrinsic_matrices.torch.shape == (1, 3, 3)
    assert camera.data.image_shape == (HEIGHT, WIDTH)
    assert camera.data.info == {"distance_to_image_plane": None}
    print(camera)

    # reading the data above refreshed the buffers once; reset the frame counters before counting
    camera.reset()
    assert torch.all(camera.frame == 0)
    for step in range(1, 3):
        sim.step()
        camera.update(DT, force_recompute=True)
        other.update(DT)
        assert camera.data.output["distance_to_image_plane"].shape == (1, HEIGHT, WIDTH, 1)
        assert other.data.output["distance_to_image_plane"].shape == (1, 120, 160, 1)
        assert camera.frame[0].item() == step
    camera.reset(env_ids=[0])
    assert camera.frame[0].item() == 0
    sim.step()
    camera.update(DT, force_recompute=True)
    camera.reset()
    assert torch.all(camera.frame == 0)


@pytest.mark.parametrize("convention, quat", [("ros", QUAT_ROS), ("opengl", QUAT_OPENGL), ("world", QUAT_WORLD)])
def test_camera_init_offset(setup_sim, convention, quat):
    """The same offset expressed in any convention yields the same pose in every convention."""
    sim, camera_cls, camera_cfg = setup_sim
    offset = type(camera_cfg).OffsetCfg(pos=POSITION, rot=quat, convention=convention)
    camera = _make_camera(camera_cls, camera_cfg, f"/World/CameraOffset_{convention}", offset=offset)
    sim.reset()
    camera.update(DT)

    np.testing.assert_allclose(camera.data.pos_w.torch[0].cpu().numpy(), POSITION, rtol=1e-5)
    _assert_quat_close(camera.data.quat_w_ros[0], QUAT_ROS, rtol=1e-5, atol=1e-5)
    _assert_quat_close(camera.data.quat_w_opengl[0], QUAT_OPENGL, rtol=1e-5, atol=1e-5)
    _assert_quat_close(camera.data.quat_w_world[0], QUAT_WORLD, rtol=1e-5, atol=1e-5)


def test_camera_init_intrinsic_matrix(setup_sim):
    """A pattern built from the pinhole intrinsic matrix reproduces the focal-length/aperture camera."""
    sim, camera_cls, camera_cfg = setup_sim
    camera = camera_cls(camera_cfg)
    fx = WIDTH * FOCAL_LENGTH / HORIZONTAL_APERTURE
    intrinsic_matrix = [fx, 0.0, WIDTH / 2, 0.0, fx, HEIGHT / 2, 0.0, 0.0, 1.0]
    pattern_cfg = patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
        intrinsic_matrix=intrinsic_matrix, height=HEIGHT, width=WIDTH, focal_length=FOCAL_LENGTH
    )
    other = _make_camera(camera_cls, camera_cfg, "/World/Camera_2", pattern_cfg=pattern_cfg)
    sim.reset()
    camera.update(DT)
    other.update(DT)

    expected = torch.tensor(intrinsic_matrix, device=camera.device).reshape(1, 3, 3)
    torch.testing.assert_close(camera.data.intrinsic_matrices.torch, expected)
    torch.testing.assert_close(other.data.intrinsic_matrices.torch, expected)
    torch.testing.assert_close(
        camera.data.output["distance_to_image_plane"].torch, other.data.output["distance_to_image_plane"].torch
    )


def test_camera_set_world_poses(setup_sim):
    """``set_world_poses`` and ``set_world_poses_from_view`` update the reported pose."""
    sim, camera_cls, camera_cfg = setup_sim
    camera = camera_cls(camera_cfg)
    sim.reset()

    position = torch.tensor([POSITION], device=camera.device)
    orientation = torch.tensor([QUAT_WORLD], device=camera.device)
    camera.set_world_poses(position.clone(), orientation.clone(), convention="world")
    torch.testing.assert_close(camera.data.pos_w.torch, position)
    torch.testing.assert_close(camera.data.quat_w_world.torch, orientation)

    eyes = torch.tensor([POSITION], device=camera.device)
    camera.set_world_poses_from_view(eyes.clone(), torch.zeros_like(eyes))
    torch.testing.assert_close(camera.data.pos_w.torch, eyes)
    _assert_quat_close(camera.data.quat_w_ros, torch.tensor([QUAT_ROS], device=camera.device))


def test_set_intrinsic_matrices(setup_sim):
    """A runtime intrinsic matrix is retained across updates and changes the rendered depth."""
    sim, camera_cls, camera_cfg = setup_sim
    offset = type(camera_cfg).OffsetCfg(pos=(0.0, 0.0, 5.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world")
    camera = _make_camera(camera_cls, camera_cfg, offset=offset, data_types=["distance_to_camera"])
    sim.reset()
    camera.update(DT)
    output_before = camera.data.output["distance_to_camera"].torch.clone()

    # a much shorter focal length widens the field of view and changes the depth at the image edges
    new_matrix = torch.tensor([[[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]]], device=camera.device)
    camera.set_intrinsic_matrices(new_matrix.clone(), focal_length=1.0)
    for _ in range(2):
        sim.step()
        camera.update(DT)
        torch.testing.assert_close(camera.data.intrinsic_matrices.torch, new_matrix)

    output_after = camera.data.output["distance_to_camera"].torch
    assert not torch.allclose(output_before, output_after, atol=1e-3), "stale ray buffers after set_intrinsic_matrices"
    assert not torch.isnan(output_after).any()
    assert output_after[torch.isfinite(output_after)].min() > 0


def test_depth_clipping(setup_sim):
    """Depth clipping behaviors ``none``, ``zero``, and ``max``, with independent d2ip and d2c outputs."""
    sim, camera_cls, camera_cfg = setup_sim
    cfg_cls = type(camera_cfg)
    base_cfg = cfg_cls(
        prim_path="/World/Camera",
        mesh_prim_paths=[GROUND_PATH],
        offset=cfg_cls.OffsetCfg(pos=(2.5, 2.5, 6.0), rot=(0.0, 0.1305, 0.0, 0.9914449), convention="world"),
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            focal_length=38.0,
            intrinsic_matrix=[380.08, 0.0, 467.79, 0.0, 380.08, 262.05, 0.0, 0.0, 1.0],
            height=540,
            width=960,
        ),
        max_distance=10.0,
        data_types=["distance_to_image_plane", "distance_to_camera"],
    )
    cameras = {
        behavior: _make_camera(camera_cls, base_cfg, f"/World/Camera_{behavior}", depth_clipping_behavior=behavior)
        for behavior in ("none", "zero", "max")
    }
    solo = {
        data_type: _make_camera(
            camera_cls, base_cfg, f"/World/Camera_{data_type}", depth_clipping_behavior="max", data_types=[data_type]
        )
        for data_type in ("distance_to_image_plane", "distance_to_camera")
    }
    sim.reset()
    for camera in (*cameras.values(), *solo.values()):
        camera.update(DT)

    d2c_none = cameras["none"].data.output["distance_to_camera"].torch
    d2ip_none = cameras["none"].data.output["distance_to_image_plane"].torch
    missed_d2c, missed_d2ip = torch.isinf(d2c_none), torch.isnan(d2ip_none)
    assert missed_d2c.any() and missed_d2ip.any()
    assert d2c_none[~missed_d2c].max() > base_cfg.max_distance
    assert d2ip_none[~missed_d2ip].max() > base_cfg.max_distance

    for behavior, fill in (("zero", 0.0), ("max", base_cfg.max_distance)):
        d2c = cameras[behavior].data.output["distance_to_camera"].torch
        d2ip = cameras[behavior].data.output["distance_to_image_plane"].torch
        assert torch.all(d2c[missed_d2c] == fill) and torch.all(d2ip[missed_d2ip] == fill)
        assert d2c.max() <= base_cfg.max_distance and d2ip.max() <= base_cfg.max_distance

    # requesting both depth types must not corrupt either compared to requesting them alone
    for data_type, camera in solo.items():
        torch.testing.assert_close(
            cameras["max"].data.output[data_type].torch, camera.data.output[data_type].torch, atol=1e-5, rtol=1e-5
        )


"""
Equivalence with the USD camera
"""


def _assert_outputs_match_usd(camera_warp, camera_usd, data_types):
    for data_type in data_types:
        usd_output = camera_usd.data.output[data_type].torch
        if data_type == "normals":
            usd_output = usd_output[..., :3]
        torch.testing.assert_close(usd_output, camera_warp.data.output[data_type].torch, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("mount", ["look_at", "offset", "prim_offset"])
def test_output_equal_to_usd_camera(setup_sim, mount):
    """Depth and normal images match the USD camera for a look-at pose, a config offset, and a mounted prim."""
    sim, camera_cls, camera_cfg = setup_sim
    cfg_cls = type(camera_cfg)
    data_types = ["distance_to_image_plane", "distance_to_camera", "normals"]
    warp_kwargs, usd_kwargs = {}, {}
    if mount == "offset":
        warp_kwargs["offset"] = cfg_cls.OffsetCfg(pos=(2.5, 2.5, 4.0), rot=OFFSET_ROT_ROS, convention="ros")
        usd_kwargs["offset"] = CameraCfg.OffsetCfg(pos=(2.5, 2.5, 4.0), rot=OFFSET_ROT_ROS, convention="ros")
    elif mount == "prim_offset":
        warp_kwargs["offset"] = cfg_cls.OffsetCfg(pos=(0.0, 0.0, 2.0), rot=OFFSET_ROT_ROS, convention="ros")
        usd_kwargs["offset"] = CameraCfg.OffsetCfg(pos=(0.0, 0.0, 2.0), rot=OFFSET_ROT_ROS, convention="ros")
        usd_kwargs["update_latest_camera_pose"] = True
    usd_prim_path = "/World/Camera_usd"
    if mount == "prim_offset":
        # both cameras hang off a translated and rotated Xform
        for path in ("/World/Camera_warp", "/World/Camera_usd"):
            sim_utils.create_prim(path, "Xform", translation=POSITION, orientation=QUAT_OPENGL)
        usd_prim_path = "/World/Camera_usd/camera"
    camera_warp = _make_camera(
        camera_cls,
        camera_cfg,
        "/World/Camera_warp",
        create_prim=mount != "prim_offset",
        data_types=data_types,
        **warp_kwargs,
    )
    camera_usd = Camera(_usd_camera_cfg(usd_prim_path, data_types, **usd_kwargs))
    sim.reset()
    if mount == "look_at":
        eyes = np.asarray([[2.5, 2.5, 4.5]], dtype=np.float32)
        targets = np.zeros((1, 3), dtype=np.float32)
        camera_warp.set_world_poses_from_view(
            torch.tensor(eyes, device=camera_warp.device), torch.tensor(targets, device=camera_warp.device)
        )
        camera_usd.set_world_poses_from_view(eyes, targets)
    for _ in range(3):
        sim.step()
    camera_usd.update(DT)
    camera_warp.update(DT)

    torch.testing.assert_close(camera_warp.data.pos_w.torch, camera_usd.data.pos_w.torch)
    _assert_quat_close(camera_warp.data.quat_w_ros.torch, camera_usd.data.quat_w_ros.torch)
    torch.testing.assert_close(camera_usd.data.intrinsic_matrices.torch, camera_warp.data.intrinsic_matrices.torch)
    sensor_prim = camera_usd._sensor_prims[0]
    assert sensor_prim.GetHorizontalApertureAttr().Get() == pytest.approx(HORIZONTAL_APERTURE)
    assert sensor_prim.GetVerticalApertureAttr().Get() == pytest.approx(HORIZONTAL_APERTURE * HEIGHT / WIDTH)
    _assert_outputs_match_usd(camera_warp, camera_usd, data_types)


@pytest.mark.parametrize("focal_length", [1.93, 19.3])
def test_output_equal_to_usd_camera_intrinsics(setup_sim, focal_length):
    """Both cameras built from the same intrinsic matrix and focal length render the same depth."""
    sim, camera_cls, camera_cfg = setup_sim
    cfg_cls = type(camera_cfg)
    intrinsics = [380.0831, 0.0, WIDTH / 2, 0.0, 380.0831, HEIGHT / 2, 0.0, 0.0, 1.0]
    offset_pos = (2.5, 2.5, 4.0)
    sim_utils.create_prim("/World/Camera_warp", "Xform")
    camera_warp = camera_cls(
        cfg_cls(
            prim_path="/World/Camera_warp",
            mesh_prim_paths=[GROUND_PATH],
            offset=cfg_cls.OffsetCfg(pos=offset_pos, rot=OFFSET_ROT_ROS, convention="ros"),
            pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
                intrinsic_matrix=intrinsics, height=HEIGHT, width=WIDTH, focal_length=focal_length
            ),
            depth_clipping_behavior="max",
            max_distance=20.0,
            data_types=["distance_to_image_plane"],
        )
    )
    camera_usd = Camera(
        CameraCfg(
            prim_path="/World/Camera_usd",
            offset=CameraCfg.OffsetCfg(pos=offset_pos, rot=OFFSET_ROT_ROS, convention="ros"),
            spawn=PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=intrinsics,
                height=HEIGHT,
                width=WIDTH,
                clipping_range=(0.01, 20),
                focal_length=focal_length,
            ),
            height=HEIGHT,
            width=WIDTH,
            depth_clipping_behavior="max",
            data_types=["distance_to_image_plane"],
        )
    )
    sim.reset()
    for _ in range(3):
        sim.step()
    camera_usd.update(DT)
    camera_warp.update(DT)

    torch.testing.assert_close(camera_warp.data.intrinsic_matrices.torch, camera_usd.data.intrinsic_matrices.torch)
    sensor_prim = camera_usd._sensor_prims[0]
    assert sensor_prim.GetHorizontalApertureAttr().Get() == pytest.approx(
        camera_warp.cfg.pattern_cfg.horizontal_aperture
    )
    assert sensor_prim.GetVerticalApertureAttr().Get() == pytest.approx(camera_warp.cfg.pattern_cfg.vertical_aperture)
    warp_output = camera_warp.data.output["distance_to_image_plane"].torch
    usd_output = camera_usd.data.output["distance_to_image_plane"].torch
    torch.testing.assert_close(
        warp_output.nan_to_num(0.0, 0.0, 0.0), usd_output.nan_to_num(0.0, 0.0, 0.0), atol=5e-5, rtol=1e-5
    )


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_output_equal_to_usd_camera_when_intrinsics_set(setup_sim):
    """Runtime intrinsics set on both cameras keep their depth images equal."""
    sim, camera_cls, camera_cfg = setup_sim
    camera_warp = _make_camera(camera_cls, camera_cfg, data_types=["distance_to_camera"])
    camera_usd = Camera(_usd_camera_cfg("/World/Camera_usd", ["distance_to_camera"]))
    sim.reset()

    intrinsic_matrix = np.asarray([[380.0831, 0.0, WIDTH / 2], [0.0, 380.0831, HEIGHT / 2], [0.0, 0.0, 1.0]])[None]
    camera_warp.set_intrinsic_matrices(torch.tensor(intrinsic_matrix, device=camera_warp.device), focal_length=10)
    camera_usd.set_intrinsic_matrices(torch.tensor(intrinsic_matrix, device=camera_usd.device), focal_length=10)
    eyes = np.asarray([[0.0, 0.0, 5.0]], dtype=np.float32)
    targets = np.zeros((1, 3), dtype=np.float32)
    camera_warp.set_world_poses_from_view(
        torch.tensor(eyes, device=camera_warp.device), torch.tensor(targets, device=camera_warp.device)
    )
    camera_usd.set_world_poses_from_view(eyes, targets)
    for _ in range(3):
        sim.step()
    camera_usd.update(DT)
    camera_warp.update(DT)

    torch.testing.assert_close(
        camera_usd.data.output["distance_to_camera"].torch,
        camera_warp.data.output["distance_to_camera"].torch,
        rtol=5e-3,
        atol=1e-4,
    )


"""
Multi-mesh specific behavior
"""


@pytest.fixture
def multi_mesh_sim(setup_sim, camera_type):
    if camera_type != "multi_mesh":
        pytest.skip("multi-mesh camera only")
    return setup_sim


def test_image_mesh_ids_identify_hit_mesh(multi_mesh_sim):
    """``image_mesh_ids`` carries the ground mesh id for every pixel that hits the (single) ground mesh."""
    sim, camera_cls, camera_cfg = multi_mesh_sim
    camera = _make_camera(camera_cls, camera_cfg, update_mesh_ids=True, data_types=["distance_to_camera"])
    sim.reset()
    camera.update(DT)

    mesh_ids = camera.data.image_mesh_ids.torch
    assert mesh_ids.shape == (1, HEIGHT, WIDTH, 1) and mesh_ids.dtype == torch.int16
    # with the default "none" clipping, missed rays stay at inf
    hit_mask = torch.isfinite(camera.data.output["distance_to_camera"].torch[0, :, :, 0])
    assert hit_mask.any()
    assert torch.all(mesh_ids[0, :, :, 0][hit_mask] == 0)


def _create_heterogeneous_clone_scene(sim: sim_utils.SimulationContext, num_envs: int) -> torch.Tensor:
    """Alternating Spot/ANYmal robots and cube/sphere objects replicated through a clone plan."""
    stage = sim_utils.get_current_stage()
    env_fmt = "/World/envs/env_{}"
    env_ids = np.arange(num_envs, dtype=np.int64)
    env_origins, _ = lab_cloner.grid_transforms(num_envs, spacing=4.0)

    sim_utils.create_prim("/World/envs", "Xform", stage=stage)
    for env_id, origin in enumerate(env_origins):
        sim_utils.create_prim(env_fmt.format(env_id), "Xform", translation=tuple(origin), stage=stage)
        sim_utils.create_prim(env_fmt.format(env_id) + "/RayCasterCamera", "Xform", stage=stage)

    # even envs get the env_0 prototypes, odd envs the env_1 prototypes
    mask = np.zeros((2, num_envs), dtype=np.bool_)
    mask[0, 0::2] = True
    mask[1, 1::2] = True
    clone_mask = np.concatenate((mask, mask), axis=0)

    for env_id, robot_cfg in enumerate((SPOT_CFG, ANYMAL_C_CFG)):
        spawn = copy.deepcopy(robot_cfg.spawn)
        spawn.func(env_fmt.format(env_id) + "/Robot", spawn, translation=robot_cfg.init_state.pos)
    cube_cfg = sim_utils.CuboidCfg(
        size=(0.35, 0.25, 0.25), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.2))
    )
    sphere_cfg = sim_utils.SphereCfg(
        radius=0.18, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.7))
    )
    cube_cfg.func(env_fmt.format(0) + "/Object", cube_cfg, translation=(0.45, 0.0, 0.25))
    sphere_cfg.func(env_fmt.format(1) + "/Object", sphere_cfg, translation=(0.45, 0.0, 0.25))

    sources = tuple(env_fmt.format(i) + f"/{name}" for name in ("Robot", "Object") for i in range(2))
    destinations = (env_fmt + "/Robot", env_fmt + "/Robot", env_fmt + "/Object", env_fmt + "/Object")
    lab_cloner.usd_replicate(stage, list(sources), list(destinations), env_ids, mask=clone_mask)
    sim.set_clone_plan(
        ClonePlan(
            sources=sources,
            destinations=destinations,
            clone_mask=clone_mask,
            env_ids=env_ids,
            positions=None,
            cfg_rows={},
        )
    )
    sim_utils.update_stage()
    return torch.as_tensor(env_origins, device=sim.device)


def test_depth_equal_to_usd_camera_heterogeneous_scene(multi_mesh_sim):
    """The ray caster consumes the clone plan used to build a heterogeneous scene and matches the USD depth."""
    sim, camera_cls, _ = multi_mesh_sim
    num_envs = 4
    env_origins = _create_heterogeneous_clone_scene(sim, num_envs)
    height, width = 96, 128
    mesh_prim_paths = [
        GROUND_PATH,
        MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Object", track_mesh_transforms=False),
        MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/Robot/[^/]+", track_mesh_transforms=True
        ),
    ]
    camera_warp = camera_cls(
        MultiMeshRayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/RayCasterCamera",
            mesh_prim_paths=mesh_prim_paths,
            update_period=0,
            pattern_cfg=_pattern_cfg(height, width),
            max_distance=25.0,
            data_types=["distance_to_image_plane"],
            depth_clipping_behavior="max",
            update_mesh_ids=True,
        )
    )
    usd_cfg = _usd_camera_cfg("{ENV_REGEX_NS}/UsdCamera", ["distance_to_image_plane"], height, width)
    usd_cfg.spawn.clipping_range = (0.01, 25.0)
    camera_usd = Camera(usd_cfg)
    sim.reset()

    eyes = env_origins + torch.tensor((1.8, -2.5, 2.5), device=sim.device)
    camera_warp.set_world_poses_from_view(eyes=eyes, targets=env_origins)
    camera_usd.set_world_poses_from_view(eyes=eyes.cpu().numpy(), targets=env_origins.cpu().numpy())
    for _ in range(5):
        sim.render()
    camera_usd.update(DT)
    camera_warp.update(DT)

    ray_depth = camera_warp.data.output["distance_to_image_plane"].torch
    usd_depth = camera_usd.data.output["distance_to_image_plane"].torch
    assert ray_depth.shape == usd_depth.shape == (num_envs, height, width, 1)
    mesh_ids = camera_warp.data.image_mesh_ids.torch
    # mesh id 0 is the ground, 1 the object, >= 2 the robot links
    assert torch.any(mesh_ids == 1), "expected object pixels"
    assert torch.any(mesh_ids >= 2), "expected robot pixels"

    # RTX and ray casting can disagree by a pixel along silhouettes: compare stable ground pixels away from
    # object/robot edges and depth discontinuities.
    edge_mask = mesh_ids[..., 0] != 0
    for depth in (ray_depth[..., 0], usd_depth[..., 0]):
        edge_mask[:, 1:, :] |= (depth[:, 1:, :] - depth[:, :-1, :]).abs() > 0.3
        edge_mask[:, :, 1:] |= (depth[:, :, 1:] - depth[:, :, :-1]).abs() > 0.3
    dilated = torch.nn.functional.max_pool2d(edge_mask[:, None].float(), kernel_size=21, stride=1, padding=10) > 0
    stable_mask = ~dilated[:, 0, :, :, None]
    assert stable_mask.float().mean() > 0.7
    stable_close = torch.isclose(ray_depth[stable_mask], usd_depth[stable_mask], atol=5e-5, rtol=5e-6)
    assert stable_close.float().mean() > 0.999
    assert torch.quantile((ray_depth - usd_depth).abs()[stable_mask], 0.999) < 5.0e-5
