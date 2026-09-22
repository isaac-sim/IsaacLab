# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for the USD :class:`Camera` sensor rendered through Isaac Sim RTX.

Covers initialization, offsets and pose writes, runtime intrinsics, depth clipping, the output contract of
every annotator (including batched cameras spawned through regex prim paths), rendering freshness, and the
deprecated :class:`TiledCamera` alias.
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import copy
import random
import warnings

import numpy as np
import pytest
import scipy.spatial.transform as tf
import torch
import warp as wp

import omni.replicator.core as rep
from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg, TiledCamera, TiledCameraCfg

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.isaacsim_ci]

HEIGHT, WIDTH = 240, 320
DT = 0.01
FOCAL_LENGTH, HORIZONTAL_APERTURE = 24.0, 20.955

# sample camera pose expressed in each convention (quaternions in xyzw)
POSITION = (2.5, 2.5, 2.5)
QUAT_ROS = (0.33985114, 0.82047325, -0.42470819, -0.17591989)
QUAT_OPENGL = (0.17591988, 0.42470818, 0.82047324, 0.33985113)
QUAT_WORLD = (-0.27984815, -0.1159169, 0.88047623, -0.3647052)

ALL_DATA_TYPES = [
    "rgb",
    "rgba",
    "albedo",
    "depth",
    "distance_to_camera",
    "distance_to_image_plane",
    "normals",
    "motion_vectors",
    "semantic_segmentation",
    "instance_segmentation",
    "instance_id_segmentation_fast",
]
SEGMENTATION_TYPES = ["semantic_segmentation", "instance_segmentation", "instance_id_segmentation_fast"]
# expected (channels, dtype) per data type when segmentation is colorized
OUTPUT_CONTRACT = {
    "rgb": (3, wp.uint8),
    "rgba": (4, wp.uint8),
    "albedo": (4, wp.uint8),
    "depth": (1, wp.float32),
    "distance_to_camera": (1, wp.float32),
    "distance_to_image_plane": (1, wp.float32),
    "normals": (3, wp.float32),
    "motion_vectors": (2, wp.float32),
    "semantic_segmentation": (4, wp.uint8),
    "instance_segmentation": (4, wp.uint8),
    "instance_id_segmentation_fast": (4, wp.uint8),
    "simple_shading_constant_diffuse": (3, wp.uint8),
    "simple_shading_diffuse_mdl": (3, wp.uint8),
    "simple_shading_full_mdl": (3, wp.uint8),
}


def _assert_quat_close(actual, expected, **kwargs):
    """Assert quaternions match while allowing the equivalent negated representation."""
    actual = torch.as_tensor(actual.torch if hasattr(actual, "torch") else actual)
    expected = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
    expected = torch.where((actual * expected).sum(dim=-1, keepdim=True) < 0.0, -expected, expected)
    torch.testing.assert_close(actual, expected, **kwargs)


def _camera_cfg(
    prim_path: str = "/World/Camera",
    data_types=("distance_to_image_plane",),
    height: int = HEIGHT,
    width: int = WIDTH,
    **kwargs,
) -> CameraCfg:
    kwargs.setdefault(
        "spawn",
        sim_utils.PinholeCameraCfg(
            focal_length=FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=HORIZONTAL_APERTURE,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    return CameraCfg(
        prim_path=prim_path, height=height, width=width, update_period=0, data_types=list(data_types), **kwargs
    )


def _populate_scene():
    """Ground plane, lights, and ten randomly placed rigid primitives with semantic labels."""
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.SphereLightCfg()
    cfg.func("/World/Light/GreySphere", cfg, translation=(4.5, 3.5, 10.0))
    cfg.func("/World/Light/WhiteSphere", cfg, translation=(-4.5, 3.5, 10.0))
    random.seed(0)
    np.random.seed(0)
    for i in range(10):
        position = (np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])) * np.asarray([1.5, 1.5, 0.5])
        prim_type = random.choice(["Cube", "Sphere", "Cylinder"])
        prim_path = f"/World/Objects/Obj_{i:02d}"
        prim = sim_utils.create_prim(
            prim_path, prim_type, translation=position, scale=(0.25, 0.25, 0.25), semantic_label=prim_type
        )
        geom_prim = getattr(UsdGeom, prim_type)(prim)
        geom_prim.CreateDisplayColorAttr().Set([Gf.Vec3f(random.random(), random.random(), random.random())])
        sim_utils.apply_rigid_body_properties(prim_path, [sim_utils.UsdPhysicsRigidBodyCfg()], create_if_missing=True)
        sim_utils.apply_mass_properties(prim_path, [sim_utils.MassCfg(mass=5.0)], create_if_missing=True)
        sim_utils.apply_collision_properties(prim_path, [sim_utils.UsdPhysicsCollisionCfg()], create_if_missing=True)


def _set_object_colors(color: tuple[float, float, float]):
    stage = sim_utils.get_current_stage()
    for i in range(10):
        UsdGeom.Gprim(stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")).GetDisplayColorAttr().Set([Gf.Vec3f(*color)])


@pytest.fixture
def sim(request):
    """A populated stage on the device selected through the ``device`` fixture (``cuda:0`` by default)."""
    device = request.node.callspec.params.get("device", "cuda:0") if hasattr(request.node, "callspec") else "cuda:0"
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=device))
    _populate_scene()
    sim_utils.update_stage()
    yield sim
    rep.vp_manager.destroy_hydra_textures("Replicator")
    sim.stop()
    sim.clear_instance()


def _assert_output_contract(camera: Camera, num_cameras: int, colorize: bool = True):
    """Check shapes and dtypes of every output against :data:`OUTPUT_CONTRACT`."""
    for data_type, output in camera.data.output.items():
        channels, dtype = OUTPUT_CONTRACT[data_type]
        if not colorize and data_type in SEGMENTATION_TYPES:
            channels, dtype = 1, wp.int32
        assert output.shape == (num_cameras, camera.cfg.height, camera.cfg.width, channels), data_type
        assert output.dtype == dtype, data_type


"""
Initialization and pose
"""


def test_camera_init(sim):
    """Camera prims, render settings, buffers, and image shapes for two cameras with different resolutions."""
    sim.set_setting("/physics/fabricUpdateTransformations", False)
    camera = Camera(_camera_cfg())
    small = Camera(_camera_cfg("/World/Camera_2", height=120, width=160))
    assert sim.get_setting("/isaaclab/render/rtx_sensors")
    assert sim.get_setting("/physics/fabricUpdateTransformations")
    sim.reset()

    assert camera.is_initialized and small.is_initialized
    assert camera._sensor_prims[0].GetPath().pathString == camera.cfg.prim_path
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert camera.data.pos_w.torch.shape == (1, 3)
    for quat in (camera.data.quat_w_ros, camera.data.quat_w_world, camera.data.quat_w_opengl):
        assert quat.torch.shape == (1, 4)
    assert camera.data.intrinsic_matrices.torch.shape == (1, 3, 3)
    assert camera.data.image_shape == (HEIGHT, WIDTH)
    assert camera.data.info == {"distance_to_image_plane": None}
    print(camera)

    for _ in range(2):
        sim.step()
        camera.update(DT)
        small.update(DT)
        assert camera.data.output["distance_to_image_plane"].shape == (1, HEIGHT, WIDTH, 1)
        assert small.data.output["distance_to_image_plane"].shape == (1, 120, 160, 1)


def test_camera_init_offset(sim):
    """The same offset expressed in every convention lands the prim and data on the same pose."""
    cameras = {}
    for convention, quat in (("ros", QUAT_ROS), ("opengl", QUAT_OPENGL), ("world", QUAT_WORLD)):
        offset = CameraCfg.OffsetCfg(pos=POSITION, rot=quat, convention=convention)
        cfg = _camera_cfg(f"/World/CameraOffset_{convention}", offset=offset, update_latest_camera_pose=True)
        cameras[convention] = Camera(cfg)
    sim.reset()

    for camera in cameras.values():
        # USD authors the camera in the OpenGL convention; transpose to row-major
        prim_tf = np.transpose(camera._sensor_prims[0].ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        np.testing.assert_allclose(prim_tf[0:3, 3], POSITION)
        np.testing.assert_allclose(tf.Rotation.from_matrix(prim_tf[:3, :3]).as_quat(), QUAT_OPENGL, rtol=1e-5)
        np.testing.assert_allclose(camera.data.pos_w.torch[0].cpu().numpy(), POSITION, rtol=1e-5)
        _assert_quat_close(camera.data.quat_w_ros[0], QUAT_ROS, rtol=1e-5, atol=1e-5)
        _assert_quat_close(camera.data.quat_w_opengl[0], QUAT_OPENGL, rtol=1e-5, atol=1e-5)
        _assert_quat_close(camera.data.quat_w_world[0], QUAT_WORLD, rtol=1e-5, atol=1e-5)


def test_camera_init_intrinsic_matrix(sim):
    """Spawning from the pinhole intrinsic matrix reproduces the focal-length/aperture camera."""
    camera = Camera(_camera_cfg())
    fx = WIDTH * FOCAL_LENGTH / HORIZONTAL_APERTURE
    intrinsic_matrix = [fx, 0.0, WIDTH / 2, 0.0, fx, HEIGHT / 2, 0.0, 0.0, 1.0]
    cfg = _camera_cfg("/World/Camera_2")
    cfg.spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=intrinsic_matrix,
        width=WIDTH,
        height=HEIGHT,
        focal_length=FOCAL_LENGTH,
        focus_distance=400.0,
        clipping_range=(0.1, 1.0e5),
    )
    other = Camera(cfg)
    sim.reset()
    camera.update(DT)
    other.update(DT)

    expected = torch.tensor(intrinsic_matrix, device=camera.device).reshape(1, 3, 3)
    torch.testing.assert_close(camera.data.intrinsic_matrices.torch, expected, rtol=5e-3, atol=1e-4)
    torch.testing.assert_close(other.data.intrinsic_matrices.torch, expected, rtol=5e-3, atol=1e-4)
    torch.testing.assert_close(
        camera.data.output["distance_to_image_plane"].torch,
        other.data.output["distance_to_image_plane"].torch,
        rtol=5e-3,
        atol=1e-4,
    )


@pytest.mark.parametrize("update_latest_camera_pose", [False, True])
def test_camera_set_world_poses(sim, update_latest_camera_pose):
    """Explicit pose writes and look-at poses are reflected in the data buffers."""
    camera = Camera(_camera_cfg(update_latest_camera_pose=update_latest_camera_pose))
    sim.reset()

    position = np.asarray([POSITION], dtype=np.float32)
    orientation = np.asarray([QUAT_WORLD], dtype=np.float32)
    camera.set_world_poses(position, orientation, convention="world")
    np.testing.assert_allclose(camera.data.pos_w.warp.numpy(), position)
    _assert_quat_close(camera.data.quat_w_world.warp.numpy(), orientation, rtol=1e-5, atol=1e-5)

    camera.set_world_poses_from_view(position, np.zeros((1, 3), dtype=np.float32))
    np.testing.assert_allclose(camera.data.pos_w.warp.numpy(), position)
    _assert_quat_close(camera.data.quat_w_ros.torch, torch.tensor([QUAT_ROS], device=camera.device))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_camera_pose_update_reflected_in_render(sim, device):
    """Pose writes through the frame view reach the renderer: moving away increases the rendered depth."""
    camera = Camera(_camera_cfg("/World/PoseTestCam", ["distance_to_camera"], update_latest_camera_pose=True))
    sim.reset()
    target = np.zeros((1, 3), dtype=np.float32)
    max_range = camera.cfg.spawn.clipping_range[1]

    mean_depths = []
    for eye in ([[2.0, 2.0, 2.0]], [[8.0, 8.0, 8.0]]):
        camera.set_world_poses_from_view(np.asarray(eye, dtype=np.float32), target)
        sim.step()
        camera.update(DT)
        depth = camera.data.output["distance_to_camera"].torch
        valid = depth[depth < max_range]
        assert valid.numel() > 0, "no valid depth pixels"
        mean_depths.append(valid.mean().item())
    assert mean_depths[1] > 1.5 * mean_depths[0], "camera pose change did not reach the renderer"


"""
Intrinsics and depth clipping
"""


def test_intrinsic_matrix(sim):
    """Runtime calibration changes the rendered pixels without authoring USD and resets with the camera."""
    target = sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0))
    target.func("/World/CalibrationTarget", target, translation=(0.0, 0.0, 10.0))
    offset = CameraCfg.OffsetCfg(pos=(0.0, 0.0, 15.0), convention="opengl")
    camera = Camera(_camera_cfg(offset=offset, update_latest_camera_pose=True))
    sim.reset()

    def target_width():
        for _ in range(4):
            sim.step()
            camera.update(DT, force_recompute=True)
        depth = camera.data.output["distance_to_image_plane"].torch[..., 0]
        return ((depth > 4.0) & (depth < 6.0)).any(dim=1).sum(dim=1).float()

    before = target_width()
    assert (before > 20).all()
    original = camera.data.intrinsic_matrices.torch.clone()
    wider = original.clone()
    wider[:, 0, 0] *= 0.5
    wider[:, 1, 1] *= 0.5
    authored = [attr.Get() for attr in camera._sensor_prims[0].GetPrim().GetAttributes()]
    camera.set_intrinsic_matrices(wider)
    torch.testing.assert_close(target_width(), before * 0.5, atol=2.0, rtol=0.0)
    torch.testing.assert_close(camera.data.intrinsic_matrices.torch, wider)
    assert [attr.Get() for attr in camera._sensor_prims[0].GetPrim().GetAttributes()] == authored

    fabric = camera._render_data.intrinsic_stage
    row_attribute = camera._render_data.intrinsic_row_attribute
    sim.stop()
    assert not fabric.GetPrimAtPath(camera.cfg.prim_path).GetAttribute(row_attribute).IsValid()
    sim.reset()
    torch.testing.assert_close(camera.data.intrinsic_matrices.torch, original)
    torch.testing.assert_close(target_width(), before)


def test_depth_clipping(sim):
    """Depth clipping behaviors ``none``, ``zero``, and ``max`` bound both depth outputs consistently."""
    base_cfg = CameraCfg(
        prim_path="/World/Camera",
        offset=CameraCfg.OffsetCfg(pos=(2.5, 2.5, 6.0), rot=(0.362, 0.873, -0.302, -0.125), convention="ros"),
        spawn=sim_utils.PinholeCameraCfg().from_intrinsic_matrix(
            focal_length=38.0,
            intrinsic_matrix=[380.08, 0.0, 467.79, 0.0, 380.08, 262.05, 0.0, 0.0, 1.0],
            height=540,
            width=960,
            clipping_range=(0.1, 10),
        ),
        height=540,
        width=960,
        data_types=["distance_to_image_plane", "distance_to_camera"],
    )
    near, far = base_cfg.spawn.clipping_range
    cameras = {}
    for behavior in ("none", "zero", "max"):
        cfg = copy.deepcopy(base_cfg)
        cfg.prim_path = f"/World/Camera_{behavior}"
        cfg.renderer_cfg.depth_clipping_behavior = behavior
        cameras[behavior] = Camera(cfg)
    sim.reset()
    for camera in cameras.values():
        camera.update(DT)

    outputs = {behavior: camera.data.output for behavior, camera in cameras.items()}
    for data_type in base_cfg.data_types:
        missed = torch.isinf(outputs["none"][data_type].torch)
        assert missed.any()
        assert outputs["none"][data_type].torch[~missed].min() >= near
        assert outputs["none"][data_type].torch[~missed].max() <= far
        for behavior, fill in (("zero", 0.0), ("max", far)):
            output = outputs[behavior][data_type].torch
            assert torch.all(output[missed] == fill)
            assert output[output != 0.0].min() >= near
            assert output.max() <= far


"""
Output contract
"""


def test_camera_output_shapes_and_dtypes(sim):
    """Every supported data type is allocated with the documented channel count and dtype."""
    single_types = ["rgb", "rgba", "albedo", "depth"] + [
        name for name in OUTPUT_CONTRACT if name.startswith("simple_shading")
    ]
    cameras = [Camera(_camera_cfg("/World/CameraAll", ALL_DATA_TYPES))]
    raw = _camera_cfg("/World/CameraRaw", ALL_DATA_TYPES, height=512, width=512)
    raw.renderer_cfg.colorize_instance_id_segmentation = False
    raw.renderer_cfg.colorize_instance_segmentation = False
    raw.renderer_cfg.colorize_semantic_segmentation = False
    cameras.append(Camera(raw))
    cameras += [Camera(_camera_cfg(f"/World/Camera_{name}", [name])) for name in single_types]
    sim.reset()
    for camera in cameras:
        camera.update(DT)

    _assert_output_contract(cameras[0], 1)
    _assert_output_contract(cameras[1], 1, colorize=False)
    for camera, name in zip(cameras[2:], single_types):
        # "rgb" is a zero-copy view into "rgba", so requesting either allocates both
        assert name in camera.data.output
        _assert_output_contract(camera, 1)
    assert all(isinstance(cameras[1].data.info[data_type], dict) for data_type in SEGMENTATION_TYPES)


def test_camera_data_types_ordering(sim):
    """The output keys follow the renderer contract order regardless of the requested order."""
    cameras = {
        "distance": Camera(_camera_cfg("/World/CameraDistance", ["distance_to_camera"])),
        "depth": Camera(_camera_cfg("/World/CameraDepth", ["depth"])),
        "both": Camera(_camera_cfg("/World/CameraBoth", ["distance_to_camera", "depth"])),
    }
    sim.reset()
    assert list(cameras["distance"].data.output) == ["distance_to_camera"]
    assert list(cameras["depth"].data.output) == ["depth"]
    assert list(cameras["both"].data.output) == ["depth", "distance_to_camera"]


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_camera_batch_all_annotators(sim, device):
    """Regex prim paths batch nine cameras whose annotators have valid content, unit normals, and info."""
    num_cameras = 9
    for i in range(num_cameras):
        sim_utils.create_prim(f"/World/Origin_{i}", "Xform")
    offset = CameraCfg.OffsetCfg(pos=(0.0, 0.0, 4.0), rot=(0.0, 1.0, 0.0, 0.0), convention="ros")
    camera = Camera(
        _camera_cfg("/World/Origin_[^/]*/CameraSensor", ALL_DATA_TYPES, height=128, width=256, offset=offset)
    )
    sim.reset()

    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert camera.data.pos_w.torch.shape == (num_cameras, 3)
    assert camera.data.intrinsic_matrices.torch.shape == (num_cameras, 3, 3)
    assert sorted(camera.data.output) == sorted(ALL_DATA_TYPES)
    for _ in range(3):
        sim.step()
        camera.update(DT)
        _assert_output_contract(camera, num_cameras)
        for data_type, output in camera.data.output.items():
            if data_type == "motion_vectors":
                assert (output.torch.reshape(num_cameras, -1).mean(dim=1) != 0.0).all(), data_type
            elif data_type == "normals":
                norms = torch.linalg.norm(output.torch, dim=-1)
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
            else:
                assert (output.torch.reshape(num_cameras, -1).float().mean(dim=1) > 0.0).all(), data_type
    assert all(isinstance(camera.data.info[data_type], dict) for data_type in SEGMENTATION_TYPES)

    # distinct runtime intrinsics reach the matching Fabric prims regardless of bucket order
    matrices = camera.data.intrinsic_matrices.torch[[8, 0, 4]].clone()
    matrices[:, 0, 0] = matrices[:, 1, 1] = torch.tensor([100.0, 200.0, 300.0], device=device)
    camera.set_intrinsic_matrices(matrices, env_ids=[8, 0, 4])
    fabric = sim_utils.get_current_stage(fabric=True)
    for row, index in enumerate((8, 0, 4)):
        prim = fabric.GetPrimAtPath(camera._sensor_prims[index].GetPath().pathString)
        fx = camera.cfg.width * prim.GetAttribute("focalLength").Get() / prim.GetAttribute("horizontalAperture").Get()
        assert fx == pytest.approx(matrices[row, 0, 0].item())


def test_camera_batches_render_consistently(sim):
    """Independent camera batches at the same pose render the same images; batches at other poses differ."""
    offsets = [(0.0, 0.0, 4.0), (0.0, 0.0, 4.0), (0.0, 0.0, 2.0)]
    resolutions = [(128, 256), (128, 256), (23, 200)]
    cameras = []
    for i, (offset_pos, (height, width)) in enumerate(zip(offsets, resolutions)):
        for j in range(4):
            sim_utils.create_prim(f"/World/Origin_{i}_{j}", "Xform")
        offset = CameraCfg.OffsetCfg(pos=offset_pos, rot=(0.0, 1.0, 0.0, 0.0), convention="ros")
        cfg = _camera_cfg(
            f"/World/Origin_{i}[^/]*/CameraSensor",
            ["rgb", "distance_to_camera"],
            height=height,
            width=width,
            offset=offset,
        )
        cameras.append(Camera(cfg))
    sim.reset()
    for _ in range(2):
        sim.step()
        for camera in cameras:
            camera.update(DT)

    rgbs, depths = [], []
    for camera in cameras:
        assert camera._sensor_prims[1].GetPath().pathString.endswith("_1/CameraSensor")
        rgb = camera.data.output["rgb"].torch.float() / 255.0
        depth = camera.data.output["distance_to_camera"].torch.clone()
        depth[torch.isinf(depth)] = 0.0
        assert rgb.shape == (4, camera.cfg.height, camera.cfg.width, 3)
        assert (rgb.reshape(4, -1).mean(dim=1) > 0.0).all()
        assert (depth.reshape(4, -1).mean(dim=1) > 0.0).all()
        rgbs.append(rgb)
        depths.append(depth)
    # same pose and resolution: consistent images
    assert (rgbs[0] - rgbs[1]).abs().mean() < 0.05
    assert (depths[0] - depths[1]).abs().mean() < 0.01
    # a lower camera sees a closer scene
    assert depths[2].mean() < depths[0].mean()


@pytest.mark.parametrize("device", ["cuda:0"])
def test_camera_frame_offset(sim, device):
    """The rendered frame reflects a scene color change on the next update without lag."""
    offset = CameraCfg.OffsetCfg(pos=(0.0, 0.0, 4.0), rot=(0.0, 1.0, 0.0, 0.0), convention="ros")
    camera = Camera(_camera_cfg(data_types=["rgb"], height=480, width=480, offset=offset))
    _set_object_colors((1.0, 1.0, 1.0))
    sim.reset()
    for _ in range(20):
        sim.step()
        camera.update(DT)
    image_before = camera.data.output["rgb"].torch.float() / 255.0

    _set_object_colors((0.0, 0.0, 0.0))
    sim.step()
    camera.update(DT)
    image_after = camera.data.output["rgb"].torch.float() / 255.0
    assert (image_after - image_before).abs().mean() > 0.01


"""
Error handling and deprecations
"""


def test_camera_rejects_unsupported_configs(sim):
    """Renamed and renderer-unsupported data types raise; invalidating an uninitialized camera does not."""
    with pytest.raises(ValueError, match="instance_segmentation"):
        Camera(_camera_cfg(data_types=["instance_segmentation_fast"]))

    camera = Camera(_camera_cfg("/World/NeverInitialized", spawn=None))
    assert camera._view is None
    camera._invalidate_initialize_callback(None)
    # drop the camera so it does not try to initialize (and fail) on the reset below
    del camera

    from isaaclab.renderers.base_renderer import BaseRenderer
    from isaaclab.sensors.camera.camera_data import RenderBufferKind, RenderBufferSpec

    class _PartialRenderer(BaseRenderer):
        """Publishes only ``rgba`` in its supported-output contract."""

        def __init__(self, cfg=None):
            self.cfg = cfg

        def supported_output_types(self):
            return {RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8)}

        def prepare_stage(self, stage, num_envs):
            pass

        def create_render_data(self, sensor):
            return object()

        def set_outputs(self, render_data, output_data):
            pass

        def update_transforms(self):
            pass

        def update_geometries(self):
            pass

        def update_camera(self, render_data, positions, orientations, intrinsics):
            pass

        def render(self, render_data):
            pass

        def read_output(self, render_data, camera_data):
            pass

        def cleanup(self, render_data):
            pass

    cfg = _camera_cfg("/World/PartialCamera", ["rgba", "depth", "normals"])
    cfg.renderer_cfg.class_type = _PartialRenderer
    partial_camera = Camera(cfg)
    with pytest.raises(ValueError, match="_PartialRenderer") as exc_info:
        sim.reset()
    assert "Hint:" not in str(exc_info.value)
    del partial_camera


def test_tiled_camera_alias(sim):
    """``TiledCamera``/``TiledCameraCfg`` warn about their deprecation and still behave like ``Camera``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = TiledCameraCfg(
            prim_path="/World/Camera",
            height=64,
            width=128,
            data_types=["rgb"],
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 4.0), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
            spawn=sim_utils.PinholeCameraCfg(),
        )
        camera = TiledCamera(cfg)
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("TiledCameraCfg is deprecated" in message for message in messages)
    assert any("TiledCamera is deprecated" in message for message in messages)
    assert isinstance(camera, Camera)

    sim.reset()
    sim.step()
    camera.update(DT)
    rgb = camera.data.output["rgb"]
    assert rgb.shape == (1, 64, 128, 3)
    assert (rgb.torch.float() / 255.0).mean() > 0.0
