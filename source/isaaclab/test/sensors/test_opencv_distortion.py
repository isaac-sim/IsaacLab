# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less tests for the renderer-agnostic OpenCV lens-distortion camera model.

Two renderer-independent code paths are covered without a running simulation app, renderer or GPU:

* Authoring: :func:`isaaclab.sim.spawners.sensors.spawn_camera` with a ``distortion`` cfg authors the
  ``omni:lensdistortion:*`` USD API on the camera prim (which the RTX/OVRTX renderer round-trips
  through the USD export it loads).
* Readback: :meth:`isaaclab.sensors.camera.Camera._update_intrinsic_matrices` reconstructs
  ``camera.data.intrinsic_matrices`` from the authored ``fx/fy/cx/cy`` (which may be non-square or
  off-center) instead of assuming ``fx == fy`` and a centered principal point.

The renderer actually rendering *through* the distortion is a backend-specific concern covered in
``test_camera_opencv_distortion_ovrtx.py``.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

_REQUIRED_MODULES = ("isaaclab", "pxr", "warp")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = pytest.mark.skipif(
    bool(_MISSING_MODULES),
    reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
)

if not _MISSING_MODULES:
    from pxr import Gf, Sdf, Usd, UsdGeom

    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera.camera import Camera
    from isaaclab.sim.spawners.sensors.sensors import spawn_camera
    from isaaclab.sim.spawners.sensors.sensors_cfg import (
        FisheyeCameraCfg,
        OpenCvFisheyeDistortionCfg,
        OpenCvPinholeDistortionCfg,
        PinholeCameraCfg,
    )


# Real SO-101 wrist-camera calibration: exercises fx != fy and an off-center principal point, which
# stock Isaac Lab camera cfgs cannot express.
_PINHOLE_CALIB = dict(
    fx=339.26592887,
    fy=338.82010626,
    cx=323.55809091,
    cy=250.27360914,
    image_size=(640, 480),
    k1=0.07702322,
    k2=-0.13605453,
    k3=0.05163219,
    p1=-0.00024938,
    p2=-0.00175006,
)


"""
Authoring: spawn_camera with a distortion cfg.
"""


def _spawn_camera_on_new_stage(cfg, prim_path="/World/envs/env_0/Camera"):
    """Create a fresh in-memory stage and spawn a camera prim carrying ``cfg`` on it."""
    stage = sim_utils.create_new_stage()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    # spawn_camera is @clone-decorated; call the undecorated function to spawn a single prim.
    prim = spawn_camera.__wrapped__(prim_path, cfg)
    return stage, prim


def test_pinhole_distortion_authors_opencv_api():
    """Spawning a pinhole camera with a distortion cfg authors the OpenCV pinhole USD API."""
    cfg = PinholeCameraCfg(distortion=OpenCvPinholeDistortionCfg(apply_lens_distortion=True, **_PINHOLE_CALIB))
    _stage, prim = _spawn_camera_on_new_stage(cfg)

    prefix = "omni:lensdistortion:opencvPinhole"
    assert prim.GetAttribute("omni:lensdistortion:model").Get() == "opencvPinhole"
    assert "OmniLensDistortionOpenCvPinholeAPI" in prim.GetMetadata("apiSchemas").GetAppliedItems()
    # intrinsics
    assert prim.GetAttribute(f"{prefix}:fx").Get() == pytest.approx(_PINHOLE_CALIB["fx"], abs=1e-3)
    assert prim.GetAttribute(f"{prefix}:fy").Get() == pytest.approx(_PINHOLE_CALIB["fy"], abs=1e-3)
    assert prim.GetAttribute(f"{prefix}:cx").Get() == pytest.approx(_PINHOLE_CALIB["cx"], abs=1e-3)
    assert prim.GetAttribute(f"{prefix}:cy").Get() == pytest.approx(_PINHOLE_CALIB["cy"], abs=1e-3)
    assert tuple(prim.GetAttribute(f"{prefix}:imageSize").Get()) == (640, 480)
    # distortion coefficients (applied)
    assert prim.GetAttribute(f"{prefix}:k1").Get() == pytest.approx(_PINHOLE_CALIB["k1"], abs=1e-6)
    assert prim.GetAttribute(f"{prefix}:p2").Get() == pytest.approx(_PINHOLE_CALIB["p2"], abs=1e-6)
    # unused higher-order coefficients default to zero
    assert prim.GetAttribute(f"{prefix}:k6").Get() == pytest.approx(0.0)
    assert prim.GetAttribute(f"{prefix}:s4").Get() == pytest.approx(0.0)


def test_apply_lens_distortion_false_zeros_coefficients():
    """With ``apply_lens_distortion=False`` the intrinsics are kept but coefficients are muted to zero."""
    cfg = PinholeCameraCfg(distortion=OpenCvPinholeDistortionCfg(apply_lens_distortion=False, **_PINHOLE_CALIB))
    _stage, prim = _spawn_camera_on_new_stage(cfg)

    prefix = "omni:lensdistortion:opencvPinhole"
    # intrinsics are still authored
    assert prim.GetAttribute(f"{prefix}:fx").Get() == pytest.approx(_PINHOLE_CALIB["fx"], abs=1e-3)
    # coefficients are zeroed despite non-zero cfg values
    assert prim.GetAttribute(f"{prefix}:k1").Get() == pytest.approx(0.0)
    assert prim.GetAttribute(f"{prefix}:k2").Get() == pytest.approx(0.0)
    assert prim.GetAttribute(f"{prefix}:p1").Get() == pytest.approx(0.0)


def test_fisheye_distortion_authors_opencv_fisheye_api():
    """A distortion cfg with the fisheye model authors the OpenCV fisheye USD API."""
    cfg = FisheyeCameraCfg(
        distortion=OpenCvFisheyeDistortionCfg(
            fx=300.0, fy=300.0, cx=320.0, cy=240.0, image_size=(640, 480), k1=0.1, k2=0.02, k3=0.0, k4=0.0
        )
    )
    _stage, prim = _spawn_camera_on_new_stage(cfg)

    prefix = "omni:lensdistortion:opencvFisheye"
    assert prim.GetAttribute("omni:lensdistortion:model").Get() == "opencvFisheye"
    assert "OmniLensDistortionOpenCvFisheyeAPI" in prim.GetMetadata("apiSchemas").GetAppliedItems()
    assert prim.GetAttribute(f"{prefix}:fx").Get() == pytest.approx(300.0)
    assert prim.GetAttribute(f"{prefix}:k1").Get() == pytest.approx(0.1)
    assert prim.GetAttribute(f"{prefix}:k4").Get() == pytest.approx(0.0)


def test_distortion_attributes_survive_usd_export():
    """The authored OpenCV distortion API survives serialization, which the RTX/OVRTX path consumes."""
    cfg = PinholeCameraCfg(distortion=OpenCvPinholeDistortionCfg(apply_lens_distortion=True, **_PINHOLE_CALIB))
    stage, _prim = _spawn_camera_on_new_stage(cfg)

    exported = stage.ExportToString()
    assert "OmniLensDistortionOpenCvPinholeAPI" in exported
    assert "omni:lensdistortion:opencvPinhole:fx" in exported
    assert "omni:lensdistortion:model" in exported


def test_camera_without_distortion_authors_no_opencv_api():
    """A camera cfg without a distortion model does not author any ``omni:lensdistortion`` attributes."""
    cfg = PinholeCameraCfg()
    _stage, prim = _spawn_camera_on_new_stage(cfg)

    assert not prim.GetAttribute("omni:lensdistortion:model").IsValid()
    applied = prim.GetMetadata("apiSchemas")
    applied_items = list(applied.GetAppliedItems()) if applied else []
    assert not any("LensDistortion" in item for item in applied_items)


"""
Readback: Camera._update_intrinsic_matrices with an authored distortion model.
"""


def _camera_prim_with_pinhole_distortion(fx, fy, cx, cy, width=640, height=480):
    """Build an in-memory ``UsdGeom.Camera`` carrying an OpenCV pinhole distortion model."""
    stage = Usd.Stage.CreateInMemory()
    cam = UsdGeom.Camera.Define(stage, "/Camera")
    prim = cam.GetPrim()
    prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token).Set("opencvPinhole")
    prefix = "omni:lensdistortion:opencvPinhole"
    for name, value in (("fx", fx), ("fy", fy), ("cx", cx), ("cy", cy)):
        prim.CreateAttribute(f"{prefix}:{name}", Sdf.ValueTypeNames.Float).Set(float(value))
    prim.CreateAttribute(f"{prefix}:imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(width, height))
    return stage, cam


def _read_back_intrinsics(cam, width, height):
    """Invoke :meth:`Camera._update_intrinsic_matrices` on a minimal fake camera and return K."""
    captured = {}

    def _capture_state(env_ids=None, intrinsics_src=None, update_intrinsics=False):
        captured["K"] = intrinsics_src.numpy()

    fake = Camera.__new__(Camera)
    fake._sensor_prims = [cam]
    fake._device = "cpu"
    fake._warned_distortion_image_size = False
    fake.cfg = SimpleNamespace(height=height, width=width)
    fake._resolve_env_ids_np = lambda env_ids: np.array([0])
    fake._update_camera_state = _capture_state
    # attributes touched by ``__del__``/``_clear_callbacks`` when the fake object is garbage collected
    fake._initialize_handle = None
    fake._invalidate_initialize_handle = None
    fake._prim_deletion_handle = None
    fake._debug_vis_handle = None
    fake._renderer = None
    fake._render_data = None

    Camera._update_intrinsic_matrices(fake)
    return captured["K"][0]


def test_readback_uses_authored_fx_fy_cx_cy():
    """The reconstructed intrinsic matrix reflects the authored, non-square, off-center calibration."""
    fx, fy, cx, cy = 339.26592887, 338.82010626, 323.55809091, 250.27360914
    width, height = 640, 480
    _stage, cam = _camera_prim_with_pinhole_distortion(fx, fy, cx, cy, width, height)

    k = _read_back_intrinsics(cam, width, height)

    assert k[0, 0] == pytest.approx(fx, abs=1e-2)
    assert k[1, 1] == pytest.approx(fy, abs=1e-2)
    assert k[0, 2] == pytest.approx(cx, abs=1e-2)
    assert k[1, 2] == pytest.approx(cy, abs=1e-2)
    assert k[2, 2] == pytest.approx(1.0)


def test_readback_preserves_non_square_and_off_center():
    """The readback must not collapse to the stock ``fx == fy`` / centered principal point."""
    fx, fy, cx, cy = 339.26592887, 338.82010626, 323.55809091, 250.27360914
    width, height = 640, 480
    _stage, cam = _camera_prim_with_pinhole_distortion(fx, fy, cx, cy, width, height)

    k = _read_back_intrinsics(cam, width, height)

    # non-square pixels are preserved (the old readback forced fx == fy)
    assert k[0, 0] != k[1, 1]
    # principal point is off-center (the old readback forced width/2, height/2)
    assert k[0, 2] != pytest.approx(width / 2)
    assert k[1, 2] != pytest.approx(height / 2)
