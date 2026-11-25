# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The camera's gaze, mount and sensor mask, and the Warp kernels behind them, without launching a simulator."""

import math
from dataclasses import asdict
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import CameraModel, gaze_rotation
from fovi.sensing.retina import RetinalTransform
from isaaclab_fovi.sensors.calibrated_geometry import retinal_directions
from isaaclab_fovi.sensors.foveated_camera import FoveatedCamera
from isaaclab_fovi.sensors.foveated_rtx_camera import FoveatedRTXCamera
from isaaclab_fovi.sensors.gaze_controls import ViewerGazeControls, add_viewer_gaze_controls
from isaaclab_fovi.sensors.gaze_kernels import (
    camera_params,
    project_directions,
    publish_grid_kernel,
    set_gaze_kernel,
    torch_stream,
)
from isaaclab_fovi.sensors.native_grid import GRID_STYLE, grid_sample_indices, native_grid_directions
from isaaclab_fovi.sensors.pooling import build_retinal_coords

from isaaclab.sensors.camera.camera import Camera
from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    create_rotation_matrix_from_view,
    matrix_from_quat,
    quat_from_matrix,
    quat_mul,
)
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.kitless

CUDA = test_devices(DeviceScope.DEFAULT_CUDA)
"""The kernels and their Python paths are the same on every device; the default GPU is what simulations use."""


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
        pool_enabled=False,
    )


def test_rendering_and_software_sampling_share_window_and_nodes() -> None:
    cfg = camera_config()
    camera = CameraModel(**cfg.camera_model)
    fov = camera.field_of_view("long", 0.75)
    coords, _ = build_retinal_coords(cfg, camera, "cpu")
    assert coords.fov == pytest.approx(fov)
    assert coords.cmf_a == 2.5
    retina = RetinalTransform(
        fov=fov,
        cmf_a=2.5,
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
        self._prepare_sampling()
        # The camera's own pose buffers, at the identity orientation, with their world-to-ROS conversion.
        self._data = CameraData()
        self._data.create_buffers(count, device)
        self._data.quat_w_world.torch[:, 3] = 1.0
        # Isaac RTX's behaviour; Newton and OVRTX turn the pose by the gaze in their pose update.
        self._gaze_pose_reads_angles = False
        self._gaze_written = False
        self._initialize_sampling()

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


@pytest.mark.parametrize("device", CUDA)
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


def test_sensor_mask_checks_the_rendered_rays_of_a_rolled_mount() -> None:
    """The mask sees exactly the rays the turned camera renders, found here through the camera pose instead."""
    rotation_deg = 30.0
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


def test_gaze_angles_need_one_row_per_camera() -> None:
    camera = GazeHarness()
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        camera.set_gaze_angles(torch.zeros(3, 2))
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
    # An indexed pose update turns only the selected cameras.
    untouched = camera.orientation[1].clone()
    camera.orientation[0] = mount[0]
    camera._apply_gaze_pose([0])
    torch.testing.assert_close(camera.orientation[1], untouched)


@pytest.mark.parametrize("device", CUDA)
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


def test_rtx_world_poses_are_the_mount_and_only_a_turned_mount_is_recorded(monkeypatch) -> None:
    """Isaac RTX holds the gaze in its prims; its world poses still report the bare mount, as the other renderers'
    do, so writing them back (as a viewer navigating the camera does) neither moves the mount nor doubles the gaze.

    The fixed-mount teacher reads the live camera position, so only writes that turn a mount count: not a position
    move, and not writing back the reported mount (of either quaternion sign), even with a gaze held in the prims."""
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

    # Only a write that turns a mount counts as a turn.
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


# -- The gaze kernels against fovi's camera geometry ------------------------------------------------------------------

KERNEL_DEVICE = next(iter(CUDA), "cpu")
"""The run's default GPU, or the CPU when the run selects none."""

KERNEL_CAMERAS = {
    "distorted pinhole": CameraModel(
        "pinhole",
        (480, 640),
        (520.0, 515.0, 322.0, 241.0),
        distortion=(-0.28, 0.09, 0.001, -0.002, -0.01, 0.02, -0.01, 0.005),
    ),
    "fisheye with image circle": CameraModel(
        "fisheye",
        (800, 800),
        (230.0, 230.0, 399.5, 399.5),
        distortion=(0.02, -0.01, 0.003, -0.001),
        image_circle=(399.5, 399.5, 380.0),
        max_angle_deg=95.0,
    ),
}


def set_gaze(angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The stored angles and gaze rotations :func:`set_gaze_kernel` writes."""
    stored = torch.empty_like(angles)
    rotation = torch.empty(len(angles), 3, 3, device=angles.device)
    with torch_stream(angles.device):
        wp.launch(
            set_gaze_kernel,
            dim=len(angles),
            inputs=[wp.from_torch(angles.contiguous(), dtype=wp.vec2f)],
            outputs=[wp.from_torch(stored, dtype=wp.vec2f), wp.from_torch(rotation, dtype=wp.mat33f)],
        )
    return stored, rotation


@pytest.mark.parametrize("name", KERNEL_CAMERAS)
def test_projection_kernel_matches_fovi(name: str) -> None:
    """The sensor mask projects with fovi's camera model, as the software sampler does on a real camera."""
    camera = KERNEL_CAMERAS[name]
    generator = torch.Generator().manual_seed(0)
    directions = torch.randn(20000, 3, generator=generator)
    directions[:, 2] = directions[:, 2].abs() * 1.5  # mostly forward, some sideways and behind
    directions[:2000, 2] *= -1
    directions = torch.nn.functional.normalize(directions, dim=-1).to(KERNEL_DEVICE)
    expected_pixels, expected_valid = camera.project(directions)
    pixels, valid = project_directions(directions, camera_params(camera))
    # Float32 rounding may flip a sample that lies on the image edge, nothing more.
    assert (valid == expected_valid).float().mean() > 0.9995
    assert expected_valid.any() and (~expected_valid).any()
    both = valid & expected_valid
    torch.testing.assert_close(pixels[both], expected_pixels[both], rtol=1e-4, atol=1e-2)


def test_gaze_rotation_follows_fovis_pan_tilt_convention() -> None:
    """The gaze turns the rays as fovi's software sampler turns them, so a policy sees the same retina on hardware."""
    angles = ((torch.rand(64, 2, generator=torch.Generator().manual_seed(1)) * 2 - 1) * 40).to(KERNEL_DEVICE)
    stored, rotation = set_gaze(angles)
    tilt, pan = angles.deg2rad().unbind(-1)
    direction = torch.stack((-tilt.cos() * pan.sin(), -tilt.sin(), tilt.cos() * pan.cos()), -1)
    torch.testing.assert_close(stored, angles, atol=0, rtol=0)
    torch.testing.assert_close(rotation, gaze_rotation(direction, "pan_tilt"), atol=1e-5, rtol=0)


def test_publish_kernel_matches_masking_then_copying_to_the_grid() -> None:
    camera = KERNEL_CAMERAS["distorted pinhole"]
    coords = SamplingCoords(
        camera.field_of_view("short"), 0.591, 24, style=GRID_STYLE, field_geometry="spherical", device=KERNEL_DEVICE
    )
    rays = (native_grid_directions(coords) * coords.cartesian.new_tensor((1, -1, -1))).float().contiguous()
    generator = torch.Generator().manual_seed(2)
    angles = ((torch.rand(6, 2, generator=generator) * 2 - 1) * 25).to(KERNEL_DEVICE)
    _, rotation = set_gaze(angles)
    count = len(coords)
    outputs = {
        "rgba": torch.randint(1, 255, (6, count, 1, 4), generator=generator, dtype=torch.uint8).to(KERNEL_DEVICE),
        "depth": (torch.rand(6, count, 1, 1, generator=generator) + 0.5).to(KERNEL_DEVICE),
    }
    valid = camera.project(rays @ rotation.transpose(-1, -2))[1]
    # Environments 1 and 4 were not rendered this frame and keep their previous image.
    updated = torch.tensor([True, False, True, True, False, True], device=KERNEL_DEVICE)

    # The sequence the kernel replaces: footprint mask, sensor mask, then the upright layout. The layout comes from
    # fovi's own grid reshaping rather than the index permutation the kernel uses.
    keep = valid & coords.valid_mask[None]
    expected_grid = {
        name: coords.as_grid(value.squeeze(2) * keep[..., None], sample_dim=1) for name, value in outputs.items()
    }
    indices = grid_sample_indices(coords).to(torch.int32).contiguous()
    visible = torch.zeros(6, count, dtype=torch.bool, device=KERNEL_DEVICE)
    identity = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # Only the first output receives the real gaze. Reprojecting depth instead of reusing the first launch's
    # visibility would point the depth rays the wrong way and hide visible samples.
    _, looking_away = set_gaze(torch.tensor([[0.0, 180.0]], device=KERNEL_DEVICE).repeat(6, 1))
    for index, (name, value) in enumerate(outputs.items()):
        previous = torch.full_like(expected_grid[name], 7)
        target = previous.clone()
        with torch_stream(KERNEL_DEVICE):
            wp.launch(
                publish_grid_kernel,
                dim=(6, count),
                inputs=[
                    wp.from_torch(value.squeeze(2).view(torch.uint8)),
                    wp.from_torch(indices),
                    wp.from_torch(coords.valid_mask.contiguous()),
                    wp.from_torch(updated),
                    wp.from_torch(rays, dtype=wp.vec3f),
                    wp.from_torch(rotation if index == 0 else looking_away, dtype=wp.mat33f),
                    identity,
                    camera_params(camera),
                    1,
                    int(index == 0),
                    wp.from_torch(visible),
                ],
                outputs=[wp.from_torch(target.view(6, count, -1).view(torch.uint8))],
            )
        # Float32 rounding may flip a sample on the image edge, nothing more.
        assert (target[updated] == expected_grid[name][updated]).all(-1).float().mean() > 0.999
        assert torch.equal(target[~updated], previous[~updated])
    assert (visible[updated] == valid[updated]).float().mean() > 0.999
    assert not visible[~updated].any()
    assert valid.any() and (~valid).any()


class _SliderCamera:
    """The parts of a foveated camera that the viewer's gaze and foveation controls use."""

    def __init__(self, num_envs: int, switchable: bool = True):
        self.cfg = SimpleNamespace(
            prim_path="/World/envs/env_.*/Eye",
            cmf_a_deg=0.591,
            foveated_style="warped_cartesian_as_grid",
        )
        self.is_initialized = True
        self.supports_runtime_cmf_a = switchable
        self._renders_switchable_rays = switchable
        self.gaze_angles = SimpleNamespace(torch=torch.tensor([[5.0, -10.0]]).repeat(num_envs, 1))
        self.written = []

    def set_gaze_angles(self, angles: torch.Tensor) -> None:
        self.written.append(angles.clone())
        self.gaze_angles.torch.copy_(angles)

    def set_cmf_a(self, cmf_a_deg: float) -> None:
        self.cfg.cmf_a_deg = cmf_a_deg


class _Imgui:
    """The ImGui calls the controls make; each float slider returns a scripted (changed, value), and the foveation
    slider a scripted (changed, index)."""

    def __init__(self, sliders=(), button=False, foveation=None):
        self.sliders = list(sliders)
        self.button_pressed = button
        self.foveation = foveation
        self.shown = []
        self.ids = []
        self.scopes = []

    def push_id(self, name):
        self.ids.append(name)
        self.scopes.append(name)

    def pop_id(self):
        self.ids.pop()

    def separator_text(self, label):
        self.shown.append(label)

    def slider_float(self, label, value, *_args):
        # Widgets with the same label in two cameras' sections need distinct ID scopes.
        assert self.ids, "slider drawn outside its camera's ID scope"
        self.shown.append((label, value))
        return self.sliders.pop(0) if self.sliders else (False, value)

    def slider_int(self, label, index, low, high, display):
        self.shown.append((label, index, low, high, display))
        return (True, self.foveation) if self.foveation is not None else (False, index)

    def text(self, text):
        self.shown.append(text)

    def is_item_hovered(self):
        return False

    def button(self, label):
        return self.button_pressed


def _visualizer(kind: str) -> SimpleNamespace:
    """A visualizer of ``kind`` that records its sensor UI callbacks and counts redraws of its camera view."""
    view = SimpleNamespace(invalidated=0)
    view.invalidate = lambda: setattr(view, "invalidated", view.invalidated + 1)
    callbacks = []
    return SimpleNamespace(
        cfg=SimpleNamespace(backend=SimpleNamespace(visualizer_type=kind)),
        register_sensor_ui_callback=callbacks.append,
        current_image_view=view,
        callbacks=callbacks,
    )


def test_viewer_gaze_sliders_follow_and_set_the_gaze_in_newton_windows(monkeypatch) -> None:
    monkeypatch.setattr(ViewerGazeControls, "cmf_a_values", (0.1, 1.0, 10.0))
    camera = _SliderCamera(num_envs=3)
    newton, kit = _visualizer("newton"), _visualizer("kit")
    controls = ViewerGazeControls(camera)
    # Each simulation reset attaches again; a window gets the sliders once, and only Newton windows get them.
    controls.attach(SimpleNamespace(visualizers=[newton, kit]))
    controls.attach(SimpleNamespace(visualizers=[newton, kit]))
    assert len(newton.callbacks) == 1 and not kit.callbacks
    # The window draws the sensor callbacks with its camera source; here, the camera.
    render = partial(newton.callbacks[0], sensor=camera)

    # The sliders show the first environment's gaze, and leave it alone until moved.
    imgui = _Imgui()
    render(imgui)
    assert ("Gaze tilt", 5.0) in imgui.shown and ("Gaze pan", -10.0) in imgui.shown and not camera.written
    assert not imgui.ids  # Each draw closes the ID scope it opens.
    # A gaze set elsewhere, as by a gaze action, shows on the next draw.
    camera.gaze_angles.torch[:] = torch.tensor([-30.0, 40.0])
    imgui = _Imgui()
    render(imgui)
    assert ("Gaze tilt", -30.0) in imgui.shown and ("Gaze pan", 40.0) in imgui.shown and not camera.written
    # Two cameras' sections open different ID scopes, so their same-named widgets stay apart.
    other, other_camera = _Imgui(), _SliderCamera(num_envs=1)
    ViewerGazeControls(other_camera)._render(other, other_camera)
    assert imgui.scopes and other.scopes and imgui.scopes != other.scopes
    # A window draws the controls only while the camera is its source.
    second = _visualizer("newton")
    controls.attach(SimpleNamespace(visualizers=[newton, second]))
    imgui = _Imgui()
    second.callbacks[0](imgui, other_camera)
    assert not imgui.shown
    second.callbacks[0](imgui, camera)
    assert "Foveated Camera" in imgui.shown and ("Gaze tilt", -30.0) in imgui.shown
    # Moving one sets that gaze in every environment and redraws the camera view of every window, even another one.
    render(_Imgui(sliders=[(True, 20.0), (False, -10.0)]))
    assert torch.equal(camera.written[-1], torch.tensor([[20.0, -10.0]]).repeat(3, 1))
    assert newton.current_image_view.invalidated == 1 and second.current_image_view.invalidated == 1
    render(_Imgui(button=True))
    assert not camera.written[-1].any()

    # The foveation slider steps through ViewerGazeControls.cmf_a_values with the camera's own added, in order.
    imgui = _Imgui()
    render(imgui)
    assert ("Foveation", 1, 0, 3, "cmf_a 0.591 deg") in imgui.shown  # 0.1, 0.591, 1, 10
    render(_Imgui(foveation=3))
    assert camera.cfg.cmf_a_deg == 10.0 and newton.current_image_view.invalidated == 3
    # The starting constant stays an option after the switch, so the panel can switch back to it.
    imgui = _Imgui()
    render(imgui)
    assert ("Foveation", 3, 0, 3, "cmf_a 10 deg") in imgui.shown
    # A camera that cannot switch shows its constant as fixed instead.
    fixed = _SliderCamera(num_envs=1, switchable=False)
    fixed_controls = ViewerGazeControls(fixed)
    fixed_window = _visualizer("newton")
    fixed_controls.attach(SimpleNamespace(visualizers=[fixed_window]))
    imgui = _Imgui(foveation=0)
    fixed_window.callbacks[0](imgui, fixed)
    assert "Foveation: cmf_a 0.591 deg (fixed)" in imgui.shown and fixed.cfg.cmf_a_deg == 0.591

    # Once the camera is gone, the window's callback draws nothing, whatever the window shows.
    callback = newton.callbacks[0]
    del camera, render
    imgui = _Imgui()
    callback(imgui, other_camera)
    assert not imgui.shown and controls._camera() is None


def test_every_foveated_camera_adds_its_sliders_on_reset_until_it_is_gone() -> None:
    camera = _SliderCamera(num_envs=1)
    before = set(SimulationContext._reset_callbacks)
    add_viewer_gaze_controls(camera)
    (name,) = set(SimulationContext._reset_callbacks) - before
    newton = _visualizer("newton")
    SimulationContext._reset_callbacks[name](SimpleNamespace(visualizers=[newton]))
    assert len(newton.callbacks) == 1
    del camera
    assert name not in SimulationContext._reset_callbacks


class _UnbuiltRTXCamera(FoveatedRTXCamera):
    """An RTX camera made without a simulation, which has nothing to release when collected."""

    def __del__(self):
        pass


def test_rtx_cameras_refuse_runtime_cmf_a() -> None:
    # OVRTX and Isaac RTX share FoveatedRTXCamera, whose renderers read the retina's lens textures once.
    camera = object.__new__(_UnbuiltRTXCamera)
    camera._is_initialized = True
    camera.cfg = SimpleNamespace(foveated_style="warped_cartesian_as_grid", cmf_a_deg=1.0)
    assert not camera.supports_runtime_cmf_a
    with pytest.raises(NotImplementedError, match="lens textures"):
        camera.set_cmf_a(10.0)
