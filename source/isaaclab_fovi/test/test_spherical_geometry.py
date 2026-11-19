# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering, gaze and software-sampling contracts, without launching a simulator."""

import math
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from fovi.sensing.projection import CameraModel, angular_directions
from fovi.sensing.retina import RetinalTransform
from isaaclab_fovi.sensors.calibrated_geometry import retinal_directions, sampling_geometry
from isaaclab_fovi.sensors.foveated_camera import FoveatedCamera
from isaaclab_fovi.sensors.foveated_rtx_camera import FoveatedRTXCamera
from isaaclab_fovi.sensors.pooling import build_lowres_coords

from isaaclab.sensors.camera.camera import Camera
from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    create_rotation_matrix_from_view,
    matrix_from_quat,
    quat_apply,
    quat_from_matrix,
    quat_mul,
)
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.kitless

CPU_AND_CUDA = test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA)


@pytest.fixture
def tf32_enabled():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def camera_config() -> SimpleNamespace:
    camera = CameraModel("fisheye", (80, 120), (55, 55, 59.5, 39.5), (0.02, -0.003, 0, 0))
    return SimpleNamespace(
        camera_model=asdict(camera),
        cmf_a_deg=2.5,
        spawn=None,
        retinal_window_reference_side="long",
        retinal_window_fraction=0.75,
        foveated_res=12,
        foveated_style="isotropic",
        fov_type="circular",
        radius_norm=2.0,
        isotropic_plotting_type="v1like",
        image_rotation_deg=0.0,
    )


def test_rendering_and_software_sampling_share_window_and_nodes() -> None:
    cfg = camera_config()
    fov, a = sampling_geometry(cfg)
    camera = CameraModel(**cfg.camera_model)
    assert fov == pytest.approx(camera.field_of_view("long", 0.75))
    assert a == 2.5
    coords, _ = build_lowres_coords(cfg, "cpu")
    retina = RetinalTransform(
        fov=fov,
        cmf_a=a,
        resolution=12,
        device="cpu",
        field_geometry="spherical",
        camera_model=camera,
        gaze_convention="pan_tilt",
    )
    torch.testing.assert_close(coords.cartesian, retina.sampler.coords.cartesian)


class GazeHarness(FoveatedCamera):
    """The foveated camera's gaze state and kernels on stand-in pose buffers, without a simulator."""

    def __new__(cls, *args, **overrides):
        return object.__new__(cls)

    def __init__(self, count: int = 2, device: str = "cpu", **overrides) -> None:
        self.cfg = camera_config()
        for name, value in overrides.items():
            setattr(self.cfg, name, value)
        self._view = SimpleNamespace(count=count)
        self._device = device
        self._ALL_INDICES = wp.array(list(range(count)), dtype=wp.int32, device=device)
        self._sampling_coords, _ = build_lowres_coords(self.cfg, device)
        self._sampling_coords_highres = None
        # The camera's own pose buffers, at the identity orientation, with their world-to-ROS conversion.
        self._data = CameraData()
        self._data.create_buffers(count, device)
        self._data.quat_w_world.torch[:, 3] = 1.0
        # Isaac RTX's behaviour; Newton and OVRTX turn the pose by the gaze in their pose update.
        self._gaze_pose_reads_angles = False
        self._gaze_written = False
        self._initialize_gaze()

    @property
    def orientation(self) -> torch.Tensor:
        """The camera's world orientation (x, y, z, w), world convention, shape (count, 4)."""
        return self._data.quat_w_world.torch

    def __del__(self) -> None:
        """The harness registers no simulator callbacks to release."""


class PrimView:
    """Camera prims under an unturned parent: their world poses are their local poses (x, y, z, w), OpenGL."""

    def __init__(self, orientations: torch.Tensor) -> None:
        self.count = len(orientations)
        self.local = orientations.clone()

    def get_world_poses(self, indices=None):
        ids = slice(None) if indices is None else wp.to_torch(indices).long()
        orientations = self.local[ids].contiguous()
        positions = torch.zeros(len(orientations), 3)
        return ProxyArray(wp.from_torch(positions, dtype=wp.vec3f)), ProxyArray(
            wp.from_torch(orientations, dtype=wp.quatf)
        )

    def get_local_poses(self):
        return None, ProxyArray(wp.from_torch(self.local.clone(), dtype=wp.quatf))

    def set_local_poses(self, *, orientations) -> None:
        self.local = wp.to_torch(orientations).clone()


class RtxGazeHarness(GazeHarness, FoveatedRTXCamera):
    """The Isaac RTX camera's prim-writing gaze on stand-in camera prims."""

    def __init__(self, mount_opengl: torch.Tensor) -> None:
        super().__init__(count=len(mount_opengl))
        self._view = PrimView(mount_opengl)
        self._base_local_orientation = matrix_from_quat(mount_opengl)
        self.mount_turns = 0


MOUNT = torch.nn.functional.normalize(torch.tensor([[0.2, -0.4, 0.1, 0.9], [0.0, 0.3, -0.5, 0.6]]), dim=-1)
"""Two cameras' mount orientations (x, y, z, w), OpenGL convention, for the prim-writing harness."""


def write_view(camera, eyes, targets, env_ids=None):
    """Stand-in for the stock look-at write on :class:`RtxGazeHarness`: each row whose eye differs from its target
    looks at it; the others keep their poses."""
    rotation = create_rotation_matrix_from_view(eyes, targets, "Z")
    looks = ~torch.isnan(rotation).any(dim=(-2, -1))
    ids = camera._env_indices(env_ids)[looks]
    camera._view.local[ids] = quat_from_matrix(rotation[looks])


def write_prims(camera, positions=None, orientations=None, env_ids=None, convention="ros"):
    """Stand-in for the stock pose write on :class:`RtxGazeHarness`: the orientations become the prims' poses."""
    if orientations is not None:
        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids)
        camera._view.local[ids] = convert_camera_frame_orientation_convention(orientations, convention, "opengl")


def gaze_direction(angles_deg: torch.Tensor) -> torch.Tensor:
    """OpenCV camera-frame unit direction of (tilt, pan) gaze angles [deg]: up and left are positive."""
    tilt, pan = angles_deg.deg2rad().unbind(-1)
    return torch.stack((-tilt.cos() * pan.sin(), -tilt.sin(), tilt.cos() * pan.cos()), -1)


def angles_toward(camera: CameraModel, pixel: tuple[float, float]) -> torch.Tensor:
    """(tilt, pan) gaze angles [deg] that look along the ray through a source pixel (x, y)."""
    direction, _ = camera.unproject(torch.tensor([pixel]))
    x, y, z = direction.unbind(-1)
    return torch.stack((-torch.atan2(y, torch.sqrt(x * x + z * z)), -torch.atan2(x, z)), -1).rad2deg()


@pytest.mark.parametrize("pixel", [(119.5, 39.5), (59.5, 79.5), (120.5, 39.5), (59.5, -1.3)])
def test_edge_gaze_preserves_visible_rays(pixel: tuple[float, float]) -> None:
    """Looking at, or just past, an image edge keeps the part of the retina the camera still sees."""
    camera = GazeHarness(count=1)
    camera.set_gaze_angles(angles_toward(camera._camera_model, pixel))

    # An edge-centered receptive field straddles the physical sensor boundary.
    # Its visible part must survive even when the center lies outside the image.
    valid = camera._gaze_validity()
    assert valid.any()
    assert not valid.all()


def test_gaze_away_from_sensor_masks_all_rays() -> None:
    camera = GazeHarness(count=1)
    camera.set_gaze_angles(torch.tensor([[0.0, 180.0]]))
    assert not camera._gaze_validity().any()


@pytest.mark.parametrize("device", CPU_AND_CUDA)
@pytest.mark.parametrize(
    "mode, autocast",
    [
        ("spherical", False),
        ("spherical", True),
        ("spherical_ids", False),
        ("spherical_warp_ids", False),
        ("spherical_slice", False),
        ("spherical_mask", False),
    ],
)
def test_gaze_preserves_camera_rotation_with_tf32(device: str, autocast: bool, mode: str, tf32_enabled) -> None:
    batch_size = 48
    source_gl = torch.tensor([[0.4625, 0.4940, -0.5341, 0.5067]], device=device).repeat(batch_size, 1)
    conversion = source_gl.new_tensor([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).double()
    source = quat_from_matrix(matrix_from_quat(source_gl.double()) @ conversion.T).float()
    original = source.clone()
    angles = torch.tensor([[7.5, -12.0]], device=device).repeat(batch_size, 1)
    angles[1::2] = 0
    expected = matrix_from_quat(source.double()) @ conversion
    camera = GazeHarness(count=batch_size, device=device)
    camera.orientation.copy_(source)
    # The gaze turns the camera's own pose buffer in place.
    source = camera.orientation
    camera._gaze_basis = camera._gaze_basis.to(device)
    camera.set_gaze_angles(angles)
    basis = camera._gaze_basis.double()
    expected = expected @ basis @ camera._gaze_rotation.torch.double() @ basis.T @ conversion.T
    selected = torch.arange(batch_size, device=device) % 2 == 0
    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        if mode == "spherical_ids":
            camera._apply_gaze_pose(torch.nonzero(selected).flatten())
        elif mode == "spherical_warp_ids":
            # An int64 Warp selector, which the kernel's int32 indices must not reinterpret.
            camera._apply_gaze_pose(wp.from_torch(torch.nonzero(selected).flatten(), dtype=wp.int64))
        elif mode == "spherical_slice":
            camera._apply_gaze_pose(slice(0, None, 2))
        elif mode == "spherical_mask":
            camera._apply_gaze_pose(env_mask=wp.from_torch(selected))
        else:
            camera._apply_gaze_pose()
    if mode != "spherical":
        torch.testing.assert_close(source[~selected], original[~selected], atol=0, rtol=0)
        source = source[selected]
        expected = expected[selected]
    assert source.dtype == torch.float32
    assert torch.isfinite(source).all()
    torch.testing.assert_close(source.norm(dim=-1), torch.ones(len(source), device=device), atol=1e-6, rtol=0)
    torch.testing.assert_close(matrix_from_quat(source.double()), expected, atol=1e-6, rtol=0)
    assert torch.get_float32_matmul_precision() == "high"


@pytest.mark.parametrize("rotation_deg", [0.0, 30.0, -75.0])
def test_sensor_mask_checks_the_rendered_rays_of_a_rolled_mount(rotation_deg: float) -> None:
    """The mask sees exactly the rays the turned camera renders, found here through the camera pose instead."""
    # An off-center, wide image makes a roll change which samples the camera sees.
    model = CameraModel("pinhole", (80, 120), (60.0, 60.0, 45.0, 30.0))
    camera = GazeHarness(camera_model=asdict(model), image_rotation_deg=rotation_deg)
    camera.set_gaze_angles(torch.tensor([[6.0, -11.0], [-9.0, 17.0]]))
    mount = camera.orientation.clone()
    camera._apply_gaze_pose()

    # Rendered rays leave the turned camera along its OpenGL frame; express them in the mount's OpenGL frame.
    def opengl(orientation: torch.Tensor) -> torch.Tensor:
        return matrix_from_quat(convert_camera_frame_orientation_convention(orientation, "world", "opengl"))

    turn = opengl(mount).transpose(-1, -2) @ opengl(camera.orientation)
    coords = camera._sampling_coords
    rays = retinal_directions(coords.cartesian, coords.fov, rotation_deg)
    seen = (turn[:, None] @ rays[None, :, :, None]).squeeze(-1) * torch.tensor([1.0, -1.0, -1.0])
    expected = model.project(seen)[1]
    assert expected.any() and (~expected).any()
    assert (camera._gaze_validity() == expected).float().mean() > 0.999


def test_only_a_turned_mount_is_recorded(monkeypatch) -> None:
    """The fixed-mount teacher reads the live camera position, so only writes that turn a mount count: not a position
    move, and not writing back the reported mount (of either quaternion sign), even with a gaze held in the prims."""
    monkeypatch.setattr(Camera, "set_world_poses", write_prims)
    monkeypatch.setattr(Camera, "set_world_poses_from_view", write_view)
    eyes = torch.zeros(2, 3)
    targets = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def view_twice(camera):
        # Looking again from the same eye at the same target leaves the mount where the first view put it.
        camera.set_world_poses_from_view(eyes, targets)
        camera.mount_turns = 0
        camera.set_world_poses_from_view(eyes, targets)

    def reported(camera, env_ids=None):
        return camera.get_world_poses(env_ids, convention="ros")[1]

    def turn_second(camera):
        # One degree about the second camera's optical axis.
        half = torch.tensor(math.radians(1.0) / 2)
        roll = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, float(half.sin()), float(half.cos())]])
        camera.set_world_poses(orientations=quat_mul(reported(camera), roll))

    for move, turned in (
        (lambda camera: camera.set_world_poses(positions=torch.zeros(2, 3)), False),
        (lambda camera: camera.set_world_poses(orientations=reported(camera)), False),
        (lambda camera: camera.set_world_poses(orientations=-reported(camera, [1]), env_ids=[1]), False),
        (turn_second, True),
        (lambda camera: camera.set_world_poses_from_view(eyes, eyes), False),
        (lambda camera: camera.set_world_poses_from_view(eyes, targets), True),
        (view_twice, False),
    ):
        camera = RtxGazeHarness(MOUNT)
        camera.set_gaze_angles(torch.tensor([[6.0, -11.0], [-9.0, 17.0]]))
        move(camera)
        assert camera.mount_turns == int(turned)


def test_a_rejected_mount_move_is_not_recorded(monkeypatch) -> None:
    def reject(*args, **kwargs):
        raise ValueError("rejected")

    monkeypatch.setattr(Camera, "set_world_poses", reject)
    # The identity differs from both mounts, so only the rejection keeps the count.
    camera = RtxGazeHarness(MOUNT)
    with pytest.raises(ValueError, match="rejected"):
        camera.set_world_poses(orientations=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2))
    assert camera.mount_turns == 0


@pytest.mark.parametrize("shape", [(1, 2), (2, 3), (3, 2)])
def test_gaze_angles_need_one_row_per_camera(shape) -> None:
    camera = GazeHarness()
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        camera.set_gaze_angles(torch.zeros(shape))
    assert camera._gaze_angles is None


def test_mount_record_starts_at_the_current_pose() -> None:
    """Allocating the gaze before any pose update records the camera's current, ungazed mount, not zeros."""
    camera = GazeHarness()
    camera._gaze_pose_reads_angles = True
    mount = torch.nn.functional.normalize(torch.tensor([[0.2, -0.4, 0.1, 0.9], [0.0, 0.3, -0.5, 0.6]]), dim=-1)
    camera.orientation.copy_(mount)
    camera.gaze_angles
    expected = convert_camera_frame_orientation_convention(mount, "world", "ros")
    torch.testing.assert_close(
        matrix_from_quat(camera.ungazed_quat_w_ros()), matrix_from_quat(expected), atol=1e-6, rtol=0
    )


def test_prim_writing_camera_reads_its_mount_before_and_after_a_gaze() -> None:
    """Isaac RTX writes the gaze into its prims; its mount reads the same before and after a gaze is set, so a
    fixed-mount teacher created after a gaze aims from the real mount."""
    camera = RtxGazeHarness(MOUNT)
    # The last pose update read the bare mount.
    camera.orientation.copy_(convert_camera_frame_orientation_convention(MOUNT, "opengl", "world"))
    expected = matrix_from_quat(convert_camera_frame_orientation_convention(MOUNT, "opengl", "ros"))
    torch.testing.assert_close(matrix_from_quat(camera.ungazed_quat_w_ros()), expected, atol=1e-6, rtol=0)
    camera.set_gaze_angles(torch.tensor([[6.0, -11.0], [-9.0, 17.0]]))
    assert not torch.allclose(matrix_from_quat(camera._view.local), matrix_from_quat(MOUNT), atol=1e-3)
    torch.testing.assert_close(matrix_from_quat(camera.ungazed_quat_w_ros()), expected, atol=1e-6, rtol=0)


def test_sensor_mask_reuses_its_buffer_and_keeps_rows_that_were_not_updated() -> None:
    camera = GazeHarness()
    camera.cfg.mask_to_sensor_fov = True
    camera.set_gaze_angles(torch.tensor([[0.0, 0.0], [0.0, 0.0]]))
    first = camera._gaze_validity()
    kept = first[0].clone()
    output = {"distance_to_camera": torch.ones(2, len(camera._gaze_rays), 1, 1)}
    camera.set_gaze_angles(torch.tensor([[0.0, 180.0], [0.0, 180.0]]))
    camera._apply_sensor_fov_mask(output, wp.array([False, True], dtype=wp.bool, device="cpu"))
    assert camera._visibility["valid"].torch is first
    # The environment that was not updated keeps its samples and its last visibility.
    torch.testing.assert_close(first[0], kept)
    assert bool(output["distance_to_camera"][0].all())
    assert not output["distance_to_camera"][1].any()


def test_pose_update_records_the_mount_before_the_gaze() -> None:
    camera = GazeHarness()
    mount = torch.nn.functional.normalize(torch.tensor([[0.2, -0.4, 0.1, 0.9], [0.0, 0.3, -0.5, 0.6]]), dim=-1)
    camera.orientation.copy_(mount)
    camera.set_gaze_angles(torch.tensor([[6.0, -11.0], [-9.0, 17.0]]))
    camera._apply_gaze_pose()
    expected = convert_camera_frame_orientation_convention(mount, "world", "ros")
    torch.testing.assert_close(
        matrix_from_quat(camera._mount_quat_ros.torch), matrix_from_quat(expected), atol=1e-6, rtol=0
    )
    assert not torch.allclose(camera.orientation, mount)


def test_virtual_pose_aims_rays_without_translation_and_preserves_unselected_pose() -> None:
    camera = GazeHarness()
    angles = torch.tensor([[6.0, -11.0], [-9.0, 17.0]])
    camera.set_gaze_angles(angles)
    camera._apply_gaze_pose()
    forward = quat_apply(camera.orientation, torch.tensor([[1.0, 0.0, 0.0]]).expand(2, -1))
    source = CameraModel(**camera.cfg.camera_model)
    # The world camera frame looks along +X with +Z up; OpenCV looks along +Z with +Y down.
    directions = gaze_direction(angles)
    expected = torch.stack((directions[:, 2], -directions[:, 0], -directions[:, 1]), -1)
    torch.testing.assert_close(forward, expected, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(camera._data.pos_w.torch) == 0
    untouched = camera.orientation[1].clone()
    camera.orientation[0] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    camera._apply_gaze_pose([0])
    torch.testing.assert_close(camera.orientation[1], untouched)
    torch.testing.assert_close(
        camera._gaze_validity(),
        source.project(
            angular_directions(camera._sampling_coords.cartesian, camera._sampling_coords.fov)
            @ camera._gaze_rotation.torch.transpose(-1, -2)
        )[1],
    )


def test_renderer_ray_frame_matches_cv_frame() -> None:
    coords = torch.tensor([[0.0, 0.0], [0.3, -0.4], [-1.0, 0.0]])
    rendered = retinal_directions(coords, 120)
    cv = angular_directions(coords, 120)
    torch.testing.assert_close(rendered, cv * torch.tensor([1, -1, -1]))


@pytest.mark.parametrize("device", CPU_AND_CUDA)
def test_gaze_pose_is_stable_inside_model_autocast(device: str) -> None:
    cameras = [GazeHarness(device=device), GazeHarness(device=device)]
    for camera in cameras:
        camera.orientation.copy_(torch.tensor([[-0.4625, -0.4940, 0.5341, -0.5067]], device=device).repeat(2, 1))
        camera._gaze_basis = camera._gaze_basis.to(device)
        camera._gaze_rays = camera._gaze_rays.to(device)
        camera._gaze_identity = camera._gaze_identity.to(device)
    target = torch.tensor([[0.002, 0.003], [-0.1, 0.2]], device=device)
    for camera, enabled in zip(cameras, [False, True], strict=True):
        with torch.autocast(device, dtype=torch.bfloat16, enabled=enabled):
            camera.set_gaze_angles(target)
            camera._apply_gaze_pose()
            camera._test_validity = camera._gaze_validity().clone()
    assert torch.isfinite(cameras[1].orientation).all()
    torch.testing.assert_close(cameras[1].orientation, cameras[0].orientation)
    torch.testing.assert_close(cameras[1]._gaze_rotation.torch, cameras[0]._gaze_rotation.torch)

    torch.testing.assert_close(cameras[1].orientation.norm(dim=-1), torch.ones(2, device=device))
    torch.testing.assert_close(cameras[1]._test_validity, cameras[0]._test_validity)


@pytest.mark.parametrize("device", CPU_AND_CUDA)
def test_rtx_gaze_pose_writer_is_stable_inside_model_autocast(device: str, tf32_enabled) -> None:
    camera = GazeHarness(count=48, device=device)
    camera._gaze_basis = camera._gaze_basis.to(device)
    camera._base_local_orientation = matrix_from_quat(
        torch.tensor([[-0.4625, -0.4940, 0.5341, -0.5067]], device=device).repeat(48, 1)
    )
    written = []
    camera._view = SimpleNamespace(
        count=48,
        set_local_poses=lambda *, orientations: written.append(torch.utils.dlpack.from_dlpack(orientations).clone()),
    )
    # Policy outputs may themselves be BF16; both paths use the same input values.
    angles = torch.tensor([[-0.006677, -0.010717]], device=device, dtype=torch.bfloat16).repeat(48, 1)
    for enabled in [False, True]:
        with torch.autocast(device, dtype=torch.bfloat16, enabled=enabled):
            camera.set_gaze_angles(angles)
            FoveatedRTXCamera._write_gaze_local_pose(camera)
    assert torch.isfinite(written[1]).all()
    assert written[1].dtype == torch.float32
    torch.testing.assert_close(written[1].norm(dim=-1), torch.ones(48, device=device))
    torch.testing.assert_close(written[1], written[0])


def test_rtx_world_poses_are_the_mount_so_a_round_trip_keeps_mount_and_gaze(monkeypatch) -> None:
    """Isaac RTX holds the gaze in its prims; its world poses still report the bare mount, as the other renderers'
    do, so writing them back (as a viewer navigating the camera does) neither moves the mount nor doubles the gaze."""
    monkeypatch.setattr(Camera, "set_world_poses", write_prims)
    camera = RtxGazeHarness(MOUNT)
    mount = MOUNT
    camera.set_gaze_angles(torch.tensor([[6.0, -11.0], [-9.0, 17.0]]))
    gazed = camera._view.local.clone()
    assert not torch.allclose(matrix_from_quat(gazed), matrix_from_quat(mount), atol=1e-3)

    _, orientations = camera.get_world_poses(env_ids=[1], convention="ros")
    expected = convert_camera_frame_orientation_convention(mount[1:], "opengl", "ros")
    torch.testing.assert_close(matrix_from_quat(orientations), matrix_from_quat(expected), atol=1e-6, rtol=0)

    _, orientations = camera.get_world_poses(convention="world")
    camera.set_world_poses(orientations=orientations, convention="world")
    torch.testing.assert_close(camera._base_local_orientation, matrix_from_quat(mount), atol=1e-6, rtol=0)
    torch.testing.assert_close(matrix_from_quat(camera._view.local), matrix_from_quat(gazed), atol=1e-6, rtol=0)


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_gaze_kernels_preserve_shared_stream_for_warp_capture(device: str) -> None:
    wp.init()
    stream = wp.Stream(device)
    with wp.ScopedStream(stream), torch.cuda.stream(wp.stream_to_torch(stream)):
        camera = GazeHarness(device=device)
        camera.set_gaze_angles(torch.tensor([[5.0, -8.0], [0.0, 0.0]], device=device))
        valid = camera._gaze_validity()
        target = wp.zeros(valid.shape, dtype=wp.bool, device=device)
        # Physics must still be able to capture after the camera's temporary stream scopes exit.
        with wp.ScopedCapture(stream=stream) as capture:
            wp.copy(target, wp.from_torch(valid))
        wp.capture_launch(capture.graph)
        torch.testing.assert_close(wp.to_torch(target), valid)
        assert valid[1].any()
