# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The foveated camera's configuration: Isaac Lab camera configs it reads, its validation, renderer dispatch, and its
import boundary."""

import math
import subprocess
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from fovi.sensing.projection import CameraModel
from isaaclab_fovi.sensors import FoveatedCamera, FoveatedCameraCfg
from isaaclab_fovi.sensors.calibrated_geometry import camera_model_from_spawn
from isaaclab_fovi.sensors.foveated_newton_camera import FoveatedNewtonCamera
from isaaclab_fovi.sensors.foveated_rtx_camera import FoveatedRTXCamera
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera.camera import Camera
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


def test_pinhole_spawn_describes_an_ideal_pinhole_across_its_short_side() -> None:
    spawn = PinholeCameraCfg()
    expected = math.degrees(2 * math.atan(spawn.horizontal_aperture / (2 * spawn.focal_length)))
    camera = CameraModel(**camera_model_from_spawn(spawn))
    assert camera.field_of_view("short") == pytest.approx(expected, rel=1e-6)
    assert camera.model == "pinhole"
    assert camera.distortion == ()
    # Without a vertical aperture the image is square; with one, the retina spans the short (vertical) side.
    assert camera.image_size[0] == camera.image_size[1]
    spawn = PinholeCameraCfg(focal_length=10.0, horizontal_aperture=20.0, vertical_aperture=10.0)
    camera = CameraModel(**camera_model_from_spawn(spawn))
    assert camera.field_of_view("short") == pytest.approx(math.degrees(2 * math.atan(0.5)), rel=1e-6)


def test_opencv_pinhole_spawn_gives_that_calibrated_camera() -> None:
    distortion = OpenCvPinholeDistortionCfg(
        fx=500.0, fy=510.0, cx=320.0, cy=240.0, image_size=(640, 480), k1=-0.1, k2=0.01, p1=0.001, k4=0.002
    )
    camera = CameraModel(**camera_model_from_spawn(PinholeCameraCfg(distortion=distortion)))
    assert camera.model == "pinhole"
    # Isaac Lab sizes are (width, height); fovi's are (height, width).
    assert camera.image_size == (480, 640)
    assert camera.intrinsics == (500.0, 510.0, 320.0, 240.0)
    assert camera.distortion == (-0.1, 0.01, 0.001, 0.0, 0.0, 0.002, 0.0, 0.0)

    distortion.apply_lens_distortion = False
    assert CameraModel(**camera_model_from_spawn(PinholeCameraCfg(distortion=distortion))).distortion == ()


def test_zero_coefficient_opencv_fisheye_is_an_ideal_equidistant_fisheye() -> None:
    # The image edge is 320 / 250 rad (73 deg) off axis, inside the default 180 deg lens.
    focal = 250.0
    distortion = OpenCvFisheyeDistortionCfg(fx=focal, fy=focal, cx=319.5, cy=319.5, image_size=(640, 640))
    camera = CameraModel(**camera_model_from_spawn(PinholeCameraCfg(distortion=distortion)))
    assert camera.model == "fisheye"
    assert camera.distortion == ()
    assert camera.max_angle_deg == pytest.approx(90.0)
    # Equidistant: the angle off the optical axis grows linearly with image radius.
    for radius in (50.0, 150.0, 310.0):
        assert ray_angle_deg(camera, (319.5 + radius, 319.5)) == pytest.approx(math.degrees(radius / focal), rel=1e-6)
    assert camera.field_of_view("short") == pytest.approx(math.degrees(640 / focal), rel=1e-6)


def test_explicit_camera_model_wins_over_the_spawn() -> None:
    for spawn in (PinholeCameraCfg(), None):
        camera = SimpleNamespace(cfg=make_cfg(spawn=spawn, camera_model=CAMERA_MODEL, foveated_res=8))
        FoveatedCamera._prepare_sampling(camera)
        assert camera._camera_model == CameraModel(**CAMERA_MODEL)


def test_cameras_fovi_cannot_model_are_rejected() -> None:
    thin_prism = OpenCvPinholeDistortionCfg(fx=1.0, fy=1.0, cx=1.0, cy=1.0, image_size=(4, 4), s1=0.1)
    for spawn, message in (
        (None, "existing prim needs camera_model"),
        (CuboidCfg(size=(1.0, 1.0, 1.0)), "does not spawn a camera"),
        (FisheyeCameraCfg(), "OpenCvFisheyeDistortionCfg"),
        (PinholeCameraCfg(horizontal_aperture_offset=1.0), "Aperture offsets"),
        (PinholeCameraCfg(distortion=thin_prism), "Thin-prism"),
    ):
        with pytest.raises(ValueError, match=message):
            FoveatedCamera._prepare_sampling(SimpleNamespace(cfg=make_cfg(spawn=spawn)))


def test_config_validates_pooling_and_shell_settings() -> None:
    assert make_cfg(pool_mode="gaussian").pool_mode == "gaussian"
    with pytest.raises(ValueError, match="pool_mode must be one of"):
        make_cfg(pool_mode="median")
    # Segmentation pools only by modal label: other modes would mix labels into ones no object has.
    assert make_cfg().pool_int_mode == "modal"
    with pytest.raises(ValueError, match="pool_int_mode must be one of"):
        make_cfg(pool_int_mode="avg")
    # Square shells need a square or circular footprint and a grid style.
    grid = dict(foveated_style="warped_cartesian_as_grid", radius_norm=math.inf)
    assert make_cfg(**grid, fov_type="square").fov_type == "square"
    with pytest.raises(ValueError, match="wang"):
        make_cfg(**grid, fov_type="wang")
    for kwargs in (
        dict(radius_norm=1.0, foveated_style="warped_cartesian_as_grid"),
        dict(radius_norm=math.inf, foveated_style="isotropic"),
    ):
        with pytest.raises(ValueError, match="radius_norm"):
            make_cfg(**kwargs)


@dataclass
class RendererSelection:
    renderer_type: str


@dataclass
class CameraSelection:
    renderer_cfg: RendererSelection


def test_renderer_selects_the_implementation() -> None:
    def dispatch(cfg) -> type:
        return type(FoveatedCamera.__new__(FoveatedCamera, cfg))

    # The default renderer is Isaac RTX.
    assert dispatch(make_cfg()) is FoveatedRTXCamera
    for renderer_type, expected in (
        ("ovrtx", FoveatedRTXCamera),
        ("newton_warp", FoveatedNewtonCamera),
        ("newton_optix", FoveatedNewtonCamera),
    ):
        assert dispatch(CameraSelection(RendererSelection(renderer_type))) is expected
    with pytest.raises(ValueError, match="Unsupported foveated renderer"):
        FoveatedCamera(CameraSelection(RendererSelection("unknown")))


def test_cameras_that_turn_their_pose_by_the_gaze_need_per_frame_pose_updates(monkeypatch) -> None:
    """Newton and OVRTX turn the rendered pose by the gaze in each pose update, so without pose updates the renderer
    would never see the gaze; Isaac RTX writes the gaze into its camera prims instead."""
    monkeypatch.setattr(Camera, "__init__", lambda self, cfg: setattr(self, "stage", None))
    monkeypatch.setattr(Camera, "__del__", lambda self: None)
    monkeypatch.setattr(FoveatedRTXCamera, "_prepare_projection", lambda self: None)
    monkeypatch.setattr(sim_utils, "find_matching_prims", lambda *args: [])

    def cfg(renderer_cfg) -> FoveatedCameraCfg:
        return make_cfg(spawn=PinholeCameraCfg(), renderer_cfg=renderer_cfg, update_latest_camera_pose=False)

    for renderer in (NewtonWarpRendererCfg(), OVRTXRendererCfg()):
        with pytest.raises(ValueError, match="update_latest_camera_pose=True"):
            FoveatedCamera(cfg(renderer))
    assert not FoveatedCamera(cfg(IsaacRtxRendererCfg())).turns_pose_by_gaze


IMPORT_BOUNDARY = """
import importlib, importlib.abc, sys

# Kit, fovi's research modules and training frameworks must not load with the sensor package.
BLOCKED = ("isaacsim", "omni", "fovi.models", "fovi.training", "ffcv", "wandb", "lightning", "timm")


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in BLOCKED):
            raise AssertionError(f"imported {fullname}")


sys.meta_path.insert(0, Block())
for module in ("isaaclab_fovi", "isaaclab_fovi.sensors", "isaaclab_fovi.sensors.pooling", "isaaclab_fovi.mdp"):
    importlib.import_module(module)
print("import boundary passed")
"""


def test_sensor_package_imports_without_kit_or_fovi_research_modules() -> None:
    result = subprocess.run([sys.executable, "-c", IMPORT_BOUNDARY], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "import boundary passed" in result.stdout
