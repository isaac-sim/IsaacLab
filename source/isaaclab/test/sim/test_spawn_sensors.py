# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.sensors.sensors import CUSTOM_FISHEYE_CAMERA_ATTRIBUTES, CUSTOM_PINHOLE_CAMERA_ATTRIBUTES
from isaaclab.utils.string import to_camel_case

pytestmark = [pytest.mark.unit, pytest.mark.isaacsim_ci]

_NON_USD_PARAMS = {"func", "copy_from_source", "lock_camera", "visible", "semantic_tags", "spawn_path", "distortion"}


@pytest.mark.parametrize(
    ("cfg", "custom_attributes"),
    [
        (
            sim_utils.PinholeCameraCfg(
                focal_length=5.0, f_stop=10.0, clipping_range=(0.1, 1000.0), horizontal_aperture=10.0
            ),
            CUSTOM_PINHOLE_CAMERA_ATTRIBUTES,
        ),
        (
            sim_utils.FisheyeCameraCfg(
                projection_type="fisheyePolynomial",
                focal_length=5.0,
                f_stop=10.0,
                clipping_range=(0.1, 1000.0),
                horizontal_aperture=10.0,
            ),
            CUSTOM_FISHEYE_CAMERA_ATTRIBUTES,
        ),
    ],
    ids=["pinhole", "fisheye"],
)
def test_spawn_camera(cfg, custom_attributes):
    sim_utils.create_new_stage()
    prim = cfg.func("/World/camera", cfg)

    assert prim.GetPath() == "/World/camera"
    assert prim.GetTypeName() == "Camera"
    assert prim.GetAttribute("omni:kit:cameraLock").Get() is True
    # camera-model attributes outside the USD camera schema are authored under their custom names
    for attr_name, attr_value in cfg.__dict__.items():
        if attr_name in _NON_USD_PARAMS or attr_value is None:
            continue
        prim_prop_name = custom_attributes.get(attr_name, (to_camel_case(attr_name, to="cC"),))[0]
        assert prim.GetAttribute(prim_prop_name).Get() == pytest.approx(attr_value, rel=1e-5), attr_name

    with pytest.raises(ValueError, match="already exists"):
        cfg.func("/World/camera", cfg)


def test_spawn_camera_with_opencv_distortion():
    sim_utils.create_new_stage()
    distortion = sim_utils.OpenCvPinholeDistortionCfg(
        fx=500.0, fy=510.0, cx=320.0, cy=240.0, image_size=(640, 480), k1=0.1, p1=0.01, apply_lens_distortion=True
    )
    cfg = sim_utils.PinholeCameraCfg(distortion=distortion)
    prim = cfg.func("/World/camera", cfg)

    prefix = "omni:lensdistortion:opencvPinhole"
    # the schema is authored as metadata, so it needs no registration
    assert "OmniLensDistortionOpenCvPinholeAPI" in prim.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert prim.GetAttribute("omni:lensdistortion:model").Get() == "opencvPinhole"
    assert tuple(prim.GetAttribute(f"{prefix}:imageSize").Get()) == (640, 480)
    assert prim.GetAttribute(f"{prefix}:fx").Get() == pytest.approx(500.0)
    assert prim.GetAttribute(f"{prefix}:k1").Get() == pytest.approx(0.1)
    assert prim.GetAttribute(f"{prefix}:p1").Get() == pytest.approx(0.01)

    # coefficients are muted when the distortion is not applied
    cfg = sim_utils.PinholeCameraCfg(distortion=distortion.replace(apply_lens_distortion=False))
    prim = cfg.func("/World/camera_muted", cfg)
    assert prim.GetAttribute(f"{prefix}:k1").Get() == 0.0
    assert prim.GetAttribute(f"{prefix}:fx").Get() == pytest.approx(500.0)


def test_spawn_sensor_frame():
    stage = sim_utils.create_new_stage()
    sim_utils.create_prim("/World/Parent", "Xform", translation=(1.0, 0.0, 0.0))
    cfg = sim_utils.SensorFrameCfg()
    prim = cfg.func("/World/Parent/frame", cfg, translation=(0.0, 2.0, 0.0))

    assert prim.GetTypeName() == "Xform"
    assert stage.GetPrimAtPath("/World/Parent/frame") == prim
    assert sim_utils.resolve_prim_pose(prim)[0] == pytest.approx((1.0, 2.0, 0.0))
