# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical-lens raster rays checked against independent OpenCV projections."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# OpenCV provides the independent projection reference for these checks.
cv2 = pytest.importorskip("cv2")
CameraModel = pytest.importorskip("fovi.sensing.projection").CameraModel

from isaaclab_fovi.sensors.dense_calibrated_geometry import (  # noqa: E402
    apply_lens_schema,
    dense_camera_rays,
    lens_schema_values,
)
from isaaclab_fovi.sensors.generalized_projection import (  # noqa: E402
    build_dense_calibrated_textures,
)


@pytest.mark.parametrize(
    "model,distortion",
    [
        ("pinhole", (-0.1, 0.02, 0.001, -0.002, 0.003)),
        ("fisheye", (0.01, -0.003, 0.0002, -0.00001)),
    ],
)
def test_dense_rays_reproject_to_original_pixel_centers(model: str, distortion: tuple[float, ...]) -> None:
    camera = CameraModel(model, (120, 192), (72.5, 74, 93.2, 61.7), distortion)
    rays, valid = dense_camera_rays(camera)
    cv_rays = rays.double() * torch.tensor((1, -1, -1))
    points = cv_rays[valid].numpy().reshape(-1, 1, 3)
    fx, fy, cx, cy = camera.intrinsics
    matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    project = cv2.fisheye.projectPoints if model == "fisheye" else cv2.projectPoints
    pixels, _ = project(points, np.zeros(3), np.zeros(3), matrix, np.array(distortion))
    expected = valid.nonzero()[:, [1, 0]].numpy()
    np.testing.assert_allclose(pixels[:, 0], expected, atol=5e-5)
    assert valid.float().mean() > 0.9


def test_image_circle_and_angular_domain_mask_dense_pixels() -> None:
    camera = CameraModel("fisheye", (80, 120), (40, 40, 59.5, 39.5), image_circle=(59.5, 39.5, 35), max_angle_deg=40)
    rays, valid = dense_camera_rays(camera)
    assert valid[40, 60] and not valid[0, 0]
    assert not valid[40, 95]  # Inside raster but beyond angular domain.
    assert torch.isfinite(rays).all()
    torch.testing.assert_close(rays[~valid], torch.tensor([0.0, 0.0, -1.0]).expand((~valid).sum(), 3))


def test_native_schema_keeps_brown_and_fisheye_coefficient_order() -> None:
    camera = CameraModel("pinhole", (80, 120), (55, 58, 57, 42), (1, 2, 3, 4, 5))
    schema, token, attrs = lens_schema_values(camera)
    assert schema == "OmniLensDistortionOpenCvPinholeAPI" and token == "opencvPinhole"
    assert (attrs["p1"], attrs["p2"], attrs["k3"], attrs["k4"]) == (3, 4, 5, 0)
    schema, token, attrs = lens_schema_values(CameraModel("fisheye", (80, 120), (55, 58, 57, 42)))
    assert schema == "OmniLensDistortionOpenCvFisheyeAPI" and token == "opencvFisheye"
    assert [attrs[f"k{i}"] for i in range(1, 5)] == [0, 0, 0, 0]


def test_native_schema_converts_integer_centers_to_pixel_edge_coordinates() -> None:
    for model in ("pinhole", "fisheye"):
        camera = CameraModel(model, (80, 120), (55, 58, 57, 42))
        _, _, attrs = lens_schema_values(camera)
        assert (attrs["cx"], attrs["cy"]) == (57.5, 42.5)


@pytest.mark.parametrize("model", ["pinhole", "fisheye"])
def test_native_usd_calibration_survives_stage_roundtrip(model: str) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    from pxr import UsdGeom

    camera = CameraModel(model, (80, 120), (55, 58, 57, 42), (0.01, -0.002, 0.0001, 0.0))
    schema, token, attrs = lens_schema_values(camera)
    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Camera.Define(stage, "/Camera").GetPrim()
    apply_lens_schema(prim, camera)
    text = stage.GetRootLayer().ExportToString()
    restored = Usd.Stage.CreateInMemory()
    restored.GetRootLayer().ImportFromString(text)
    prim = restored.GetPrimAtPath("/Camera")
    assert schema in prim.GetMetadata("apiSchemas").GetAppliedItems()
    assert prim.GetAttribute("omni:lensdistortion:model").Get() == token
    assert tuple(prim.GetAttribute(f"omni:lensdistortion:{token}:imageSize").Get()) == (120, 80)
    for key, value in attrs.items():
        assert prim.GetAttribute(f"omni:lensdistortion:{token}:{key}").Get() == pytest.approx(value)


def test_dense_fisheye_rtx_uses_lut_for_valid_rectangular_corners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from isaaclab.sensors import Camera
    from pxr import Usd, UsdGeom

    from isaaclab_fovi.sensors.calibrated_camera import CalibratedTiledCamera

    class InitializationOnlyCamera(CalibratedTiledCamera):
        def __init__(self) -> None:
            self.camera_model = CameraModel("fisheye", (24, 40), (18, 19, 18, 12))
            self.cfg = SimpleNamespace(prim_path="/Camera", renderer_cfg=SimpleNamespace(renderer_type="isaac_rtx"))
            self.stage = Usd.Stage.CreateInMemory()
            self._device = "cpu"

        def __del__(self) -> None:
            pass

    camera = InitializationOnlyCamera()
    prim = UsdGeom.Camera.Define(camera.stage, "/Camera").GetPrim()
    monkeypatch.setattr(Camera, "_initialize_impl", lambda self: None)
    camera._initialize_impl()

    assert not camera._invalid_pixels.any()
    assert prim.GetAttribute("omni:lensdistortion:model").Get() == "lut"
    assert "OmniLensDistortionLutAPI" in prim.GetMetadata("apiSchemas").GetAppliedItems()
    paths = [
        Path(prim.GetAttribute(f"omni:lensdistortion:lut:{name}").Get().path)
        for name in ("rayEnterDirectionTexture", "rayExitPositionTexture")
    ]
    assert all(path.is_file() for path in paths)
    rays_bgr = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
    expected, _ = dense_camera_rays(camera.camera_model)
    np.testing.assert_array_equal(rays_bgr[..., ::-1], expected.numpy())


@pytest.mark.parametrize("renderer", ["isaac_rtx", "ovrtx", "newton_warp"])
def test_calibrated_metadata_initialization_preserves_measured_intrinsics(
    renderer: str, caplog: pytest.LogCaptureFixture
) -> None:
    from isaaclab.sensors.camera.camera_data import CameraData
    from pxr import Sdf, Usd, UsdGeom

    from isaaclab_fovi.sensors.calibrated_camera import CalibratedTiledCamera

    class MetadataOnlyCamera(CalibratedTiledCamera):
        def __init__(self) -> None:
            self.camera_model = CameraModel("fisheye", (80, 120), (55, 58, 57, 42)).resized((40, 60))
            self.cfg = SimpleNamespace(height=40, width=60)
            self._device = "cpu"
            self._view = SimpleNamespace(count=2)
            self._data_generation = self._data_generation_last_update = 0
            self._data_dirty = False
            self.stage = Usd.Stage.CreateInMemory()
            self._sensor_prims = [UsdGeom.Camera.Define(self.stage, f"/Camera{i}") for i in range(2)]
            for camera in self._sensor_prims:
                camera.GetFocalLengthAttr().Set(20)
                camera.GetHorizontalApertureAttr().Set(10)
                if renderer != "newton_warp":
                    camera.GetPrim().CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("lut")
            self._data = CameraData()
            self._data.create_buffers(2, "cpu")

        def __del__(self) -> None:
            pass

    camera = MetadataOnlyCamera()
    camera._initialize_intrinsics()
    # Half-pixel-aware resize of the original integer-center calibration.
    expected = np.array(((27.5, 0, 28.25), (0, 29, 20.75), (0, 0, 1)))
    np.testing.assert_array_equal(camera.data.intrinsic_matrices.torch.numpy(), np.repeat(expected[None], 2, axis=0))
    assert "falling back" not in caplog.text
    assert camera._sensor_prims[0].GetFocalLengthAttr().Get() == 20


@pytest.mark.parametrize("distortion", [(), (0.01, -0.003, 0.0002, -0.00001)])
def test_dense_lut_inverse_preserves_pixel_centers_and_depth_directions(
    distortion: tuple[float, ...],
) -> None:
    height, width = 24, 40
    camera = CameraModel("fisheye", (height, width), (18, 19, 18, 12), distortion)
    directions, inverse = build_dense_calibrated_textures(camera, exit_size=4 * width + 1)
    # These physically valid corners lie outside the native RTX fisheye circle.
    assert all(
        torch.isfinite(directions[row, col]).all()
        for row, col in ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))
    )
    cv_directions = directions.numpy() * (1, -1, -1)
    matrix = np.array(((18, 0, 18), (0, 19, 12), (0, 0, 1)), dtype=np.float64)
    pixels, _ = cv2.fisheye.projectPoints(
        cv_directions.reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        matrix,
        np.asarray(distortion or (0, 0, 0, 0), dtype=np.float64),
    )
    row, col = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    expected_pixels = torch.stack((col, row), -1).numpy()
    np.testing.assert_allclose(pixels.reshape(height, width, 2), expected_pixels, atol=5e-5)
    torch.testing.assert_close(directions.norm(dim=-1), torch.ones(height, width))

    # For front-hemisphere rays, octahedral texture coordinates are the signed
    # XY components divided by the L1 norm, then mapped from [-1, 1] to [0, 1].
    octahedral_xy = directions[..., :2] / directions.abs().sum(-1, keepdim=True)
    recovered_uv = torch.nn.functional.grid_sample(
        inverse.permute(2, 0, 1)[None],
        octahedral_xy[None],
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)
    recovered_pixels = recovered_uv * torch.tensor((width, height)) - 0.5
    # Bilinear interpolation across octahedral axis seams has finite-table error;
    # keep it below a tenth pixel, well below a half-pixel convention mismatch.
    # Exclude raster boundaries where interpolation includes the invalid sentinel.
    torch.testing.assert_close(
        recovered_pixels[1:-1, 1:-1],
        torch.stack((col, row), -1).float()[1:-1, 1:-1],
        rtol=0,
        atol=0.1,
    )
    torch.testing.assert_close(recovered_pixels[12, 18], torch.tensor((18.0, 12.0)), rtol=0, atol=2e-6)


def test_dense_lut_inverse_keeps_declared_image_circle_and_angle_mask() -> None:
    camera = CameraModel("fisheye", (24, 40), (18, 19, 18, 12), image_circle=(18, 12, 7), max_angle_deg=15)
    _, inverse = build_dense_calibrated_textures(camera, exit_size=128)
    valid = (inverse >= 0).all(-1)
    assert valid.any() and not valid.all()
    pixels = inverse[valid].double() * torch.tensor((40, 24)) - 0.5
    rays, pixel_valid = camera.unproject(pixels)
    assert pixel_valid.all()
    assert torch.linalg.vector_norm(pixels - pixels.new_tensor((18, 12)), dim=-1).max() <= 7
    assert torch.acos(rays[:, 2]).rad2deg().max() <= 15
    assert torch.all(inverse[~valid] == -1)
