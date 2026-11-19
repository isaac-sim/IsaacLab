# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The camera's gaze, mount and sensor mask, and the Warp kernels behind them, without launching a simulator."""

import math
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import CameraModel, angular_directions, gaze_rotation
from fovi.sensing.retina import RetinalTransform
from isaaclab_fovi.sensors import FoveatedCameraCfg
from isaaclab_fovi.sensors.foveated_camera import FoveatedCamera
from isaaclab_fovi.sensors.foveated_rtx_camera import FoveatedRTXCamera
from isaaclab_fovi.sensors.gaze_kernels import (
    CameraParams,
    camera_params,
    project_direction,
    set_gaze_kernel,
    torch_stream,
)
from isaaclab_fovi.sensors.native_grid import native_grid_directions

from isaaclab.sensors.camera.camera import Camera
from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    create_rotation_matrix_from_view,
    matrix_from_quat,
    quat_apply,
    quat_from_matrix,
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
        mask_to_sensor_fov=True,
        image_rotation_deg=0.0,
    )


def test_camera_sampling_matches_software_sampling(monkeypatch) -> None:
    data = camera_config()
    model = CameraModel(**data.camera_model)
    cfg = FoveatedCameraCfg(prim_path="/World/Camera", **vars(data))
    monkeypatch.setattr(Camera, "__init__", lambda self, cfg: setattr(self, "cfg", cfg))
    monkeypatch.setattr(Camera, "__del__", lambda self: None)
    camera = FoveatedCamera.__new__(FoveatedCamera, cfg)
    FoveatedCamera.__init__(camera, cfg)
    software = RetinalTransform(
        fov=model.field_of_view("long", 0.75),
        cmf_a=2.5,
        resolution=12,
        device="cpu",
        style="warped_cartesian_as_grid",
        field_geometry="spherical",
        camera_model=model,
        gaze_convention="pan_tilt",
    )
    torch.testing.assert_close(camera.sampling_coords.cartesian, software.sampler.coords.cartesian)
    assert (camera.cfg.width, camera.cfg.height) == (12, 12)
    assert (cfg.width, cfg.height) == (1, 1)


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
        self._camera_model = CameraModel(**self.cfg.camera_model)
        fov = self._camera_model.field_of_view(self.cfg.retinal_window_reference_side, self.cfg.retinal_window_fraction)
        self._sampling_coords = SamplingCoords(
            fov, self.cfg.cmf_a_deg, self.cfg.foveated_res, style="warped_cartesian_as_grid", field_geometry="spherical"
        )
        self._retinal_rays = native_grid_directions(
            self._sampling_coords, image_rotation_deg=self.cfg.image_rotation_deg
        )
        # The camera's own pose buffers, at the identity orientation, with their world-to-ROS conversion.
        self._data = CameraData()
        self._data.create_buffers(count, device)
        self._data.quat_w_world.torch[:, 3] = 1.0
        # Isaac RTX's behaviour; Newton and OVRTX turn the pose by the gaze in their pose update.
        self._gaze_pose_reads_angles = False
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

    def __init__(self, mount_opengl: torch.Tensor, device="cpu") -> None:
        super().__init__(count=len(mount_opengl), device=device)
        self._view = PrimView(mount_opengl)
        self._base_local_orientation = mount_opengl.clone()
        self._gaze_basis_quat = quat_from_matrix(self._gaze_basis)


MOUNT = torch.nn.functional.normalize(torch.tensor([[0.2, -0.4, 0.1, 0.9], [0.0, 0.3, -0.5, 0.6]]), dim=-1)
"""Two cameras' mount orientations (x, y, z, w), OpenGL convention, for the prim-writing harness."""


def write_view(camera, eyes, targets, env_ids=None):
    """Stand-in for the stock look-at write on :class:`RtxGazeHarness`: each row whose eye differs from its target
    looks at it; the others keep their poses."""
    rotation = create_rotation_matrix_from_view(eyes, targets, "Z")
    looks = ~torch.isnan(rotation).any(dim=(-2, -1))
    ids = (torch.arange(camera._view.count) if env_ids is None else torch.as_tensor(env_ids))[looks]
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


@pytest.mark.parametrize("model_name", ["distorted pinhole", "fisheye with image circle"])
def test_camera_publishes_only_visible_samples_in_updated_rows(model_name):
    model = KERNEL_CAMERAS[model_name]
    camera = GazeHarness(
        device=KERNEL_DEVICE,
        camera_model=asdict(model),
        image_rotation_deg=37.0,
        retinal_window_reference_side="short",
        retinal_window_fraction=0.7,
    )
    coords = camera.sampling_coords
    n = coords.resolution
    rgba = torch.arange(2 * n * n * 4, device=KERNEL_DEVICE).reshape(2, n, n, 4).to(torch.uint8)
    depth = torch.arange(1, 2 * n * n + 1, device=KERNEL_DEVICE, dtype=torch.float32).reshape(2, n, n, 1)
    camera._prepare_grid_publish({"rgba": rgba, "rgb": rgba[..., :3], "distance_to_camera": depth})
    outputs = camera._data.output
    angles = torch.tensor([[6.0, -11.0], [-9.0, 17.0]], device=KERNEL_DEVICE)
    camera.set_gaze_angles(angles)
    mount = camera.orientation.clone()
    camera._apply_gaze_pose()

    def opengl(orientation):
        return matrix_from_quat(convert_camera_frame_orientation_convention(orientation, "world", "opengl"))

    rotation = opengl(mount).transpose(-1, -2) @ opengl(camera.orientation)
    # fovi's direction generation in the physical frame, independently of the publication kernel.
    ray = angular_directions(coords.cartesian, coords.fov)
    x, y, z = ray.unbind(-1)
    c, s = math.cos(math.radians(37)), math.sin(math.radians(37))
    gl = torch.stack((c * x + s * y, s * x - c * y, -z), -1)
    turned_ray = (gl @ rotation.transpose(-1, -2)) * gl.new_tensor((1.0, -1.0, -1.0))
    keep = model.project(turned_ray)[1] & coords.valid_mask[None]
    assert keep.any() and (~keep).any()

    all_rows = wp.ones(2, dtype=wp.bool, device=KERNEL_DEVICE)
    camera._publish_grid(all_rows)
    for name, values in (("rgba", rgba), ("distance_to_camera", depth)):
        canonical = values.reshape(2, n * n, -1)
        expected = coords.as_grid(canonical * keep[..., None], sample_dim=1)
        torch.testing.assert_close(outputs[name].torch, expected)
    assert outputs["rgb"].warp.ptr == outputs["rgba"].warp.ptr

    last = outputs["distance_to_camera"].torch.clone()
    depth.add_(1)
    camera._publish_grid(wp.array([False, True], dtype=wp.bool, device=KERNEL_DEVICE))
    assert camera._data.output["distance_to_camera"] is outputs["distance_to_camera"]
    assert torch.equal(last[0], outputs["distance_to_camera"].torch[0])
    assert not torch.equal(last[1], outputs["distance_to_camera"].torch[1])


def test_gaze_angles_need_one_row_per_camera() -> None:
    camera = GazeHarness()
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        camera.set_gaze_angles(torch.zeros(3, 2))
    assert camera._gaze_angles is None


def test_virtual_pose_aims_rays_without_translation_and_preserves_unselected_pose() -> None:
    camera = GazeHarness()
    angles = torch.tensor([[6.0, -11.0], [-9.0, 17.0]])
    camera.set_gaze_angles(angles)
    camera._apply_gaze_pose()
    forward = quat_apply(camera.orientation, torch.tensor([[1.0, 0.0, 0.0]]).expand(2, -1))
    # The world camera frame looks along +X with +Z up; OpenCV looks along +Z with +Y down.
    directions = gaze_direction(angles)
    expected = torch.stack((directions[:, 2], -directions[:, 0], -directions[:, 1]), -1)
    torch.testing.assert_close(forward, expected, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(camera._data.pos_w.torch) == 0
    untouched = camera.orientation[1].clone()
    camera.orientation[0] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    camera._apply_gaze_pose([0])
    torch.testing.assert_close(camera.orientation[1], untouched)


@pytest.mark.parametrize("device", CUDA)
def test_rtx_gaze_pose_writer_is_stable_inside_model_autocast(device: str, tf32_enabled) -> None:
    camera = RtxGazeHarness(
        torch.tensor([[-0.4625, -0.4940, 0.5341, -0.5067]], device=device).repeat(48, 1), device=device
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
    assert torch.isfinite(written[1]).all()
    assert written[1].dtype == torch.float32
    torch.testing.assert_close(written[1].norm(dim=-1), torch.ones(48, device=device))
    torch.testing.assert_close(written[1], written[0])


def test_rtx_world_poses_are_the_mount_so_a_round_trip_keeps_mount_and_gaze(monkeypatch) -> None:
    """Isaac RTX holds the gaze in its prims; its world poses still report the bare mount, as the other renderers'
    do, so writing them back (as a viewer navigating the camera does) neither moves the mount nor doubles the gaze."""
    monkeypatch.setattr(Camera, "set_world_poses", write_prims)
    monkeypatch.setattr(Camera, "set_world_poses_from_view", write_view)
    camera = RtxGazeHarness(MOUNT)
    mount = MOUNT
    camera.set_gaze_angles(torch.tensor([[6.0, -11.0], [-9.0, 17.0]]))
    gazed = camera._view.local.clone()
    assert not torch.allclose(matrix_from_quat(gazed), matrix_from_quat(mount), atol=1e-3)

    _, orientations = camera.get_world_poses(env_ids=[1], convention="ros")
    expected = convert_camera_frame_orientation_convention(mount[1:], "opengl", "ros")
    torch.testing.assert_close(matrix_from_quat(orientations), matrix_from_quat(expected), atol=1e-6, rtol=0)

    camera.set_world_poses(positions=torch.zeros(2, 3))
    torch.testing.assert_close(matrix_from_quat(camera._view.local), matrix_from_quat(gazed), atol=1e-6, rtol=0)
    _, orientations = camera.get_world_poses(convention="world")
    camera.set_world_poses(orientations=orientations, convention="world")
    torch.testing.assert_close(
        matrix_from_quat(camera._base_local_orientation), matrix_from_quat(mount), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(matrix_from_quat(camera._view.local), matrix_from_quat(gazed), atol=1e-6, rtol=0)

    eyes = torch.zeros(2, 3)
    targets = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    camera.set_world_poses_from_view(eyes, targets)
    torch.testing.assert_close(matrix_from_quat(camera._view.local[:1]), matrix_from_quat(gazed[:1]), atol=1e-6, rtol=0)
    _, pose = camera.get_world_poses(convention="opengl")
    expected = create_rotation_matrix_from_view(eyes[1:], targets[1:], "Z")
    torch.testing.assert_close(matrix_from_quat(pose[1:]), expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_gaze_kernels_preserve_shared_stream_for_warp_capture(device: str) -> None:
    wp.init()
    stream = wp.Stream(device)
    with wp.ScopedStream(stream), torch.cuda.stream(wp.stream_to_torch(stream)):
        camera = GazeHarness(device=device)
        camera.set_gaze_angles(torch.tensor([[5.0, -8.0], [0.0, 0.0]], device=device))
        n = camera.sampling_coords.resolution
        camera._prepare_grid_publish({"rgb": torch.ones(2, n, n, 3, dtype=torch.uint8, device=device)})
        camera._publish_grid(wp.ones(2, dtype=wp.bool, device=device))
        output = camera._data.output["rgb"].warp
        target = wp.zeros_like(output)
        # Physics must still be able to capture after the camera's temporary stream scopes exit.
        with wp.ScopedCapture(stream=stream) as capture:
            wp.copy(target, output)
        wp.capture_launch(capture.graph)
        torch.testing.assert_close(wp.to_torch(target), wp.to_torch(output))
        assert wp.to_torch(output)[1].any()


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


@wp.kernel
def _probe_projection(
    directions: wp.array(dtype=wp.vec3f),
    camera: CameraParams,
    pixels: wp.array(dtype=wp.vec2f),
    valid: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    result = project_direction(directions[i], camera)
    pixels[i] = wp.vec2f(result[0], result[1])
    valid[i] = result[2] > 0.5


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
    pixels = torch.empty_like(expected_pixels)
    valid = torch.empty_like(expected_valid)
    with torch_stream(KERNEL_DEVICE):
        wp.launch(
            _probe_projection,
            len(directions),
            [
                wp.from_torch(directions, dtype=wp.vec3f),
                camera_params(camera),
                wp.from_torch(pixels, dtype=wp.vec2f),
                wp.from_torch(valid),
            ],
        )
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
