# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering, gaze and software-sampling contracts, without launching a simulator."""

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
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    matrix_from_quat,
    quat_apply,
    quat_from_matrix,
)

pytestmark = pytest.mark.kitless


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
        retinal_window_short_side_fraction=0.75,
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
        self._sampling_coords, _ = build_lowres_coords(self.cfg, device)
        self._sampling_coords_highres = None
        self._data = SimpleNamespace(
            quat_w_world=torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(count, 1),
            pos_w=torch.zeros(count, 3, device=device),
        )
        self._initialize_gaze()


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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "mode, autocast",
    [("spherical", False), ("spherical", True), ("spherical_ids", False), ("spherical_mask", False)],
)
def test_gaze_preserves_camera_rotation_with_tf32(device: str, autocast: bool, mode: str, tf32_enabled) -> None:
    batch_size = 48
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source_gl = torch.tensor([[0.4625, 0.4940, -0.5341, 0.5067]], device=device).repeat(batch_size, 1)
    conversion = source_gl.new_tensor([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).double()
    source = quat_from_matrix(matrix_from_quat(source_gl.double()) @ conversion.T).float()
    original = source.clone()
    angles = torch.tensor([[7.5, -12.0]], device=device).repeat(batch_size, 1)
    angles[1::2] = 0
    expected = matrix_from_quat(source.double()) @ conversion
    camera = GazeHarness(count=batch_size, device=device)
    camera._data.quat_w_world = source
    camera._gaze_basis = camera._gaze_basis.to(device)
    camera.set_gaze_angles(angles)
    basis = camera._gaze_basis.double()
    expected = expected @ basis @ camera._gaze_rotation.torch.double() @ basis.T @ conversion.T
    selected = torch.arange(batch_size, device=device) % 2 == 0
    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        if mode == "spherical_ids":
            camera._apply_gaze_pose(torch.nonzero(selected).flatten())
        elif mode == "spherical_mask":
            camera._apply_gaze_pose(env_mask=wp.from_torch(selected))
        else:
            camera._apply_gaze_pose()
    if mode in ("spherical_ids", "spherical_mask"):
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
    mount = camera._data.quat_w_world.clone()
    camera._apply_gaze_pose()

    # Rendered rays leave the turned camera along its OpenGL frame; express them in the mount's OpenGL frame.
    def opengl(orientation: torch.Tensor) -> torch.Tensor:
        return matrix_from_quat(convert_camera_frame_orientation_convention(orientation, "world", "opengl"))

    turn = opengl(mount).transpose(-1, -2) @ opengl(camera._data.quat_w_world)
    coords = camera._sampling_coords
    rays = retinal_directions(coords.cartesian, coords.fov, rotation_deg)
    seen = (turn[:, None] @ rays[None, :, :, None]).squeeze(-1) * torch.tensor([1.0, -1.0, -1.0])
    expected = model.project(seen)[1]
    assert expected.any() and (~expected).any()
    assert (camera._gaze_validity() == expected).float().mean() > 0.999


def test_only_a_turned_mount_is_recorded(monkeypatch) -> None:
    """The fixed-mount teacher reads the live camera position, so only successful turns of the mount count."""
    monkeypatch.setattr(Camera, "set_world_poses", lambda *args, **kwargs: None)
    monkeypatch.setattr(Camera, "set_world_poses_from_view", lambda *args, **kwargs: None)
    eyes = torch.zeros(2, 3)
    for move, turned in (
        (lambda camera: camera.set_world_poses(positions=torch.zeros(2, 3)), False),
        (lambda camera: camera.set_world_poses(orientations=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2)), True),
        (lambda camera: camera.set_world_poses_from_view(eyes, eyes), False),
        (lambda camera: camera.set_world_poses_from_view(eyes, torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])), True),
    ):
        camera = GazeHarness()
        camera.mount_moved = False
        move(camera)
        assert camera.mount_moved is turned


def test_a_rejected_mount_move_is_not_recorded(monkeypatch) -> None:
    def reject(*args, **kwargs):
        raise ValueError("rejected")

    monkeypatch.setattr(Camera, "set_world_poses", reject)
    camera = GazeHarness()
    camera.mount_moved = False
    with pytest.raises(ValueError, match="rejected"):
        camera.set_world_poses(orientations=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2))
    assert not camera.mount_moved


@pytest.mark.parametrize("shape", [(1, 2), (2, 3), (3, 2)])
def test_gaze_angles_need_one_row_per_camera(shape) -> None:
    camera = GazeHarness()
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        camera.set_gaze_angles(torch.zeros(shape))
    assert camera._gaze_angles is None


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
    camera._data.quat_w_world = mount.clone()
    camera.set_gaze_angles(torch.tensor([[6.0, -11.0], [-9.0, 17.0]]))
    camera._apply_gaze_pose()
    expected = convert_camera_frame_orientation_convention(mount, "world", "ros")
    torch.testing.assert_close(
        matrix_from_quat(camera._mount_quat_ros.torch), matrix_from_quat(expected), atol=1e-6, rtol=0
    )
    assert not torch.allclose(camera._data.quat_w_world, mount)


def test_virtual_pose_aims_rays_without_translation_and_preserves_unselected_pose() -> None:
    camera = GazeHarness()
    angles = torch.tensor([[6.0, -11.0], [-9.0, 17.0]])
    camera.set_gaze_angles(angles)
    camera._apply_gaze_pose()
    forward = quat_apply(camera._data.quat_w_world, torch.tensor([[1.0, 0.0, 0.0]]).expand(2, -1))
    source = CameraModel(**camera.cfg.camera_model)
    # The world camera frame looks along +X with +Z up; OpenCV looks along +Z with +Y down.
    directions = gaze_direction(angles)
    expected = torch.stack((directions[:, 2], -directions[:, 0], -directions[:, 1]), -1)
    torch.testing.assert_close(forward, expected, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(camera._data.pos_w) == 0
    untouched = camera._data.quat_w_world[1].clone()
    camera._data.quat_w_world[0] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    camera._apply_gaze_pose([0])
    torch.testing.assert_close(camera._data.quat_w_world[1], untouched)
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gaze_pose_is_stable_inside_model_autocast(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cameras = [GazeHarness(device=device), GazeHarness(device=device)]
    for camera in cameras:
        camera._data.quat_w_world = torch.tensor([[-0.4625, -0.4940, 0.5341, -0.5067]], device=device).repeat(2, 1)
        camera._gaze_basis = camera._gaze_basis.to(device)
        camera._gaze_rays = camera._gaze_rays.to(device)
        camera._gaze_identity = camera._gaze_identity.to(device)
    target = torch.tensor([[0.002, 0.003], [-0.1, 0.2]], device=device)
    for camera, enabled in zip(cameras, [False, True], strict=True):
        with torch.autocast(device, dtype=torch.bfloat16, enabled=enabled):
            camera.set_gaze_angles(target)
            camera._apply_gaze_pose()
            camera._test_validity = camera._gaze_validity().clone()
    assert torch.isfinite(cameras[1]._data.quat_w_world).all()
    torch.testing.assert_close(cameras[1]._data.quat_w_world, cameras[0]._data.quat_w_world)
    torch.testing.assert_close(cameras[1]._gaze_rotation.torch, cameras[0]._gaze_rotation.torch)

    torch.testing.assert_close(cameras[1]._data.quat_w_world.norm(dim=-1), torch.ones(2, device=device))
    torch.testing.assert_close(cameras[1]._test_validity, cameras[0]._test_validity)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rtx_gaze_pose_writer_is_stable_inside_model_autocast(device: str, tf32_enabled) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gaze_kernels_preserve_shared_stream_for_warp_capture() -> None:
    wp.init()
    stream = wp.Stream("cuda:0")
    with wp.ScopedStream(stream), torch.cuda.stream(wp.stream_to_torch(stream)):
        camera = GazeHarness(device="cuda:0")
        camera.set_gaze_angles(torch.tensor([[5.0, -8.0], [0.0, 0.0]], device="cuda:0"))
        valid = camera._gaze_validity()
        target = wp.zeros(valid.shape, dtype=wp.bool, device="cuda:0")
        # Physics must still be able to capture after the camera's temporary stream scopes exit.
        with wp.ScopedCapture(stream=stream) as capture:
            wp.copy(target, wp.from_torch(valid))
        wp.capture_launch(capture.graph)
        torch.testing.assert_close(wp.to_torch(target), valid)
        assert valid[1].any()
