# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration defaults of the foveated camera and the physical camera it looks through."""

import math

import pytest
import torch
from isaaclab_fovi.sensors.calibrated_geometry import resolve_camera_model, sampling_geometry
from isaaclab_fovi.sensors.foveated_camera_cfg import FoveatedCameraCfg

from isaaclab.sim import (
    CuboidCfg,
    FisheyeCameraCfg,
    OpenCvFisheyeDistortionCfg,
    OpenCvPinholeDistortionCfg,
    PinholeCameraCfg,
)

pytestmark = pytest.mark.kitless

CAMERA_MODEL = {"model": "pinhole", "image_size": (480, 640), "intrinsics": (550.0, 550.0, 319.5, 239.5)}


def make_cfg(**kwargs) -> FoveatedCameraCfg:
    return FoveatedCameraCfg(prim_path="/World/Camera", **kwargs)


def ray_angle_deg(camera, pixel: tuple[float, float]) -> float:
    """Angle [deg] between the optical axis and the ray through a source pixel (x, y)."""
    direction, valid = camera.unproject(torch.tensor([pixel], dtype=torch.float64))
    assert valid.all()
    return math.degrees(math.acos(float(direction[0, 2])))


def test_defaults_are_grid_sampling_at_the_human_cmf_estimate() -> None:
    cfg = FoveatedCameraCfg()
    assert cfg.foveated_style == "warped_cartesian_as_grid"
    assert cfg.cmf_a_deg == 0.591
    assert cfg.camera_model is None


def test_default_pinhole_spawn_describes_an_ideal_pinhole() -> None:
    spawn = PinholeCameraCfg()
    fov, cmf_a = sampling_geometry(make_cfg(spawn=spawn))
    expected = math.degrees(2 * math.atan(spawn.horizontal_aperture / (2 * spawn.focal_length)))
    assert fov == pytest.approx(expected, rel=1e-6)
    assert cmf_a == 0.591
    camera = resolve_camera_model(make_cfg(spawn=spawn))
    assert camera.model == "pinhole"
    assert camera.distortion == ()
    # Without a vertical aperture the image is square.
    assert camera.image_size[0] == camera.image_size[1]


def test_vertical_aperture_sets_the_short_side() -> None:
    spawn = PinholeCameraCfg(focal_length=10.0, horizontal_aperture=20.0, vertical_aperture=10.0)
    fov, _ = sampling_geometry(make_cfg(spawn=spawn))
    # The short side is vertical: 2 * atan(5 / 10).
    assert fov == pytest.approx(math.degrees(2 * math.atan(0.5)), rel=1e-6)


def test_opencv_pinhole_spawn_gives_that_calibrated_camera() -> None:
    distortion = OpenCvPinholeDistortionCfg(
        fx=500.0, fy=510.0, cx=320.0, cy=240.0, image_size=(640, 480), k1=-0.1, k2=0.01, p1=0.001, k4=0.002
    )
    camera = resolve_camera_model(make_cfg(spawn=PinholeCameraCfg(distortion=distortion)))
    assert camera.model == "pinhole"
    # Isaac Lab sizes are (width, height); fovi's are (height, width).
    assert camera.image_size == (480, 640)
    assert camera.intrinsics == (500.0, 510.0, 320.0, 240.0)
    assert camera.distortion == (-0.1, 0.01, 0.001, 0.0, 0.0, 0.002, 0.0, 0.0)

    distortion.apply_lens_distortion = False
    assert resolve_camera_model(make_cfg(spawn=PinholeCameraCfg(distortion=distortion))).distortion == ()


def test_zero_coefficient_opencv_fisheye_is_an_ideal_equidistant_fisheye() -> None:
    # The image edge is 320 / 250 rad (73 deg) off axis, inside the default 180 deg lens.
    focal = 250.0
    distortion = OpenCvFisheyeDistortionCfg(fx=focal, fy=focal, cx=319.5, cy=319.5, image_size=(640, 640))
    cfg = make_cfg(spawn=PinholeCameraCfg(distortion=distortion))
    camera = resolve_camera_model(cfg)
    assert camera.model == "fisheye"
    assert camera.distortion == ()
    assert camera.max_angle_deg == pytest.approx(90.0)
    # Equidistant: the angle off the optical axis grows linearly with image radius.
    for radius in (50.0, 150.0, 310.0):
        assert ray_angle_deg(camera, (319.5 + radius, 319.5)) == pytest.approx(math.degrees(radius / focal), rel=1e-6)
    fov, _ = sampling_geometry(cfg)
    assert fov == pytest.approx(math.degrees(640 / focal), rel=1e-6)


def test_explicit_camera_model_wins_over_the_spawn() -> None:
    cfg = make_cfg(spawn=PinholeCameraCfg(), camera_model=CAMERA_MODEL)
    assert resolve_camera_model(cfg).intrinsics == CAMERA_MODEL["intrinsics"]
    # A prim without spawn configuration needs a camera model.
    assert resolve_camera_model(make_cfg(spawn=None, camera_model=CAMERA_MODEL)).image_size == (480, 640)


@pytest.mark.parametrize(
    ("spawn", "message"),
    [
        (None, "camera_model"),
        (CuboidCfg(size=(1.0, 1.0, 1.0)), "does not spawn a camera"),
        (FisheyeCameraCfg(), "OpenCvFisheyeDistortionCfg"),
        (PinholeCameraCfg(horizontal_aperture_offset=1.0), "Aperture offsets"),
        (
            PinholeCameraCfg(
                distortion=OpenCvPinholeDistortionCfg(fx=1.0, fy=1.0, cx=1.0, cy=1.0, image_size=(4, 4), s1=0.1)
            ),
            "Thin-prism",
        ),
    ],
)
def test_cameras_fovi_cannot_model_are_rejected(spawn, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_camera_model(make_cfg(spawn=spawn))


@pytest.mark.parametrize("mode", ["avg", "gaussian", "max", "sum"])
def test_pool_mode_accepts_fovi_modes(mode: str) -> None:
    assert make_cfg(pool_mode=mode).pool_mode == mode


@pytest.mark.parametrize("mode", ["modal", "median", "average"])
def test_pool_mode_rejects_other_modes(mode: str) -> None:
    with pytest.raises(ValueError, match="pool_mode must be one of"):
        make_cfg(pool_mode=mode)


@pytest.mark.parametrize("mode", ["avg", "gaussian", "max", "sum", "mode"])
def test_segmentation_pools_only_by_modal_label(mode: str) -> None:
    """Pooling modes that would mix labels into ones no object has are rejected for segmentation."""
    assert make_cfg().pool_int_mode == "modal"
    with pytest.raises(ValueError, match="pool_int_mode must be one of"):
        make_cfg(pool_int_mode=mode)
