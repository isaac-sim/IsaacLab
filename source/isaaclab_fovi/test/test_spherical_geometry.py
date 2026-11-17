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
from isaaclab_fovi.sensors.calibrated_geometry import (
    retinal_directions,
    rotate_camera_rays,
    sampling_geometry,
)
from isaaclab_fovi.sensors.foveated_rtx_camera import FoveatedRTXCamera
from isaaclab_fovi.sensors.pooling import build_lowres_coords
from isaaclab_fovi.sensors.virtual_gaze import VirtualGaze

from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_from_matrix


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
    actual = retina(torch.ones(2, 3, 80, 120), torch.tensor([[0.5, 0.5], [0.3, 0.8]]))
    assert actual.shape == (2, 3, len(coords))
    assert torch.isfinite(actual).all()


class GazeHarness(VirtualGaze):
    def __init__(self) -> None:
        self.cfg = camera_config()
        self._sampling_coords, _ = build_lowres_coords(self.cfg, "cpu")
        self._sampling_coords_highres = None
        self._gaze_rotation = None
        self._data = SimpleNamespace(
            quat_w_world=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1), pos_w=torch.zeros(2, 3)
        )
        self._initialize_gaze()


def gaze_direction(angles: torch.Tensor) -> torch.Tensor:
    """OpenCV camera-frame unit direction of (tilt, pan) gaze angles [rad]: up and left are positive."""
    tilt, pan = angles.unbind(-1)
    return torch.stack((-tilt.cos() * pan.sin(), -tilt.sin(), tilt.cos() * pan.cos()), -1)


def angles_toward(camera: CameraModel, pixel: tuple[float, float]) -> torch.Tensor:
    """(tilt, pan) gaze angles [rad] that look along the ray through a source pixel (x, y)."""
    direction, _ = camera.unproject(torch.tensor([pixel]))
    x, y, z = direction.unbind(-1)
    return torch.stack((-torch.atan2(y, torch.sqrt(x * x + z * z)), -torch.atan2(x, z)), -1)


@pytest.mark.parametrize("pixel", [(119.5, 39.5), (59.5, 79.5), (120.5, 39.5), (59.5, -1.3)])
def test_edge_gaze_preserves_visible_rays(pixel: tuple[float, float]) -> None:
    """Looking at, or just past, an image edge keeps the part of the retina the camera still sees."""
    camera = GazeHarness()
    camera.set_gaze_angles(angles_toward(camera._camera_model, pixel))

    # An edge-centered receptive field straddles the physical sensor boundary.
    # Its visible part must survive even when the center lies outside the image.
    valid = camera._gaze_validity(1)
    assert valid.any()
    assert not valid.all()


def test_gaze_away_from_sensor_masks_all_rays() -> None:
    camera = GazeHarness()
    camera.set_gaze_angles(torch.tensor([[0.0, torch.pi]]))
    assert not camera._gaze_validity(1).any()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("batch_size", [48, 256])
@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("mode", ["spherical", "spherical_ids", "spherical_mask"])
def test_gaze_preserves_camera_rotation_with_tf32(
    device: str, batch_size: int, autocast: bool, mode: str, tf32_enabled
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source_gl = torch.tensor([[0.4625, 0.4940, -0.5341, 0.5067]], device=device).repeat(batch_size, 1)
    conversion = source_gl.new_tensor([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).double()
    source = quat_from_matrix(matrix_from_quat(source_gl.double()) @ conversion.T).float()
    original = source.clone()
    angles = torch.tensor([[0.13, -0.21]], device=device).repeat(batch_size, 1)
    angles[1::2] = 0
    expected = matrix_from_quat(source.double()) @ conversion
    camera = GazeHarness()
    camera._data.quat_w_world = source
    camera._gaze_basis = camera._gaze_basis.to(device)
    camera.set_gaze_angles(angles)
    basis = camera._gaze_basis.double()
    expected = expected @ basis @ camera._gaze_rotation.double() @ basis.T @ conversion.T
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


def test_virtual_pose_aims_rays_without_translation_and_preserves_unselected_pose() -> None:
    camera = GazeHarness()
    angles = torch.tensor([[0.1, -0.2], [-0.15, 0.3]])
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
        camera._gaze_validity(2),
        source.project(
            angular_directions(camera._sampling_coords.cartesian, camera._sampling_coords.fov)
            @ camera._gaze_rotation.transpose(-1, -2)
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
    cameras = [GazeHarness(), GazeHarness()]
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
            camera._test_validity = camera._gaze_validity(2)
    assert torch.isfinite(cameras[1]._data.quat_w_world).all()
    torch.testing.assert_close(cameras[1]._data.quat_w_world, cameras[0]._data.quat_w_world)
    torch.testing.assert_close(cameras[1]._gaze_rotation, cameras[0]._gaze_rotation)

    torch.testing.assert_close(cameras[1]._data.quat_w_world.norm(dim=-1), torch.ones(2, device=device))
    torch.testing.assert_close(cameras[1]._test_validity, cameras[0]._test_validity)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rtx_gaze_pose_writer_is_stable_inside_model_autocast(device: str, tf32_enabled) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    camera = GazeHarness()
    camera._gaze_basis = camera._gaze_basis.to(device)
    camera._base_local_orientation = matrix_from_quat(
        torch.tensor([[-0.4625, -0.4940, 0.5341, -0.5067]], device=device).repeat(48, 1)
    )
    written = []
    camera._view = SimpleNamespace(
        set_local_poses=lambda *, orientations: written.append(torch.utils.dlpack.from_dlpack(orientations).clone())
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
def test_visibility_ray_rotation_preserves_precision_and_cuda_capture(tf32_enabled) -> None:
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(42)
    rays = torch.randn(16384, 3, device=device, generator=generator)
    rays = rays / rays.norm(dim=-1, keepdim=True)
    rotations = matrix_from_quat(torch.randn(48, 4, device=device, generator=generator))
    expected = (rays.double() @ rotations.double().transpose(-1, -2)).float()
    for _ in range(3):
        actual = rotate_camera_rays(rays, rotations)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = rotate_camera_rays(rays, rotations)
    graph.replay()
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-6)
    assert torch.get_float32_matmul_precision() == "high"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_visibility_ray_rotation_preserves_shared_stream_for_warp_capture() -> None:
    wp.init()
    stream = wp.Stream("cuda:0")
    with wp.ScopedStream(stream), torch.cuda.stream(wp.stream_to_torch(stream)):
        rays = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, 1.0]], device="cuda:0")
        rotation = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], device="cuda:0")
        target = wp.zeros((1, 2), dtype=wp.vec3, device="cuda:0")
        actual = rotate_camera_rays(rays, rotation)
        source = wp.from_torch(actual, dtype=wp.vec3)
        # Physics must still be able to capture after the camera's temporary scope exits.
        with wp.ScopedCapture(stream=stream) as capture:
            wp.copy(target, source)
        wp.capture_launch(capture.graph)
        torch.testing.assert_close(
            wp.to_torch(target),
            torch.tensor([[[-2.0, 1.0, 3.0], [-4.0, -2.0, 1.0]]], device="cuda:0"),
        )
