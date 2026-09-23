# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PPISP USD parsing helpers."""

from __future__ import annotations

import pytest
from isaaclab_ppisp import (
    PpispCfg,
    auto_any_ppisp_cfg,
    auto_camera_ppisp_cfg,
    default_ppisp_inputs,
    has_ppisp_camera_attrs,
    normalize_ppisp_cfg,
    ppisp_cfg_from_usd_camera,
)
from isaaclab_ppisp.cfg import PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN, resolve_and_normalize

from pxr import Gf, Sdf, Usd, Vt

_PPISP_FLOAT2_ATTRS = {
    "vignettingCenterR",
    "vignettingCenterG",
    "vignettingCenterB",
    "colorLatentBlue",
    "colorLatentRed",
    "colorLatentGreen",
    "colorLatentNeutral",
}


def _author_ppisp_attr(camera_prim: Usd.Prim, name: str, value):
    value_type = Sdf.ValueTypeNames.Float2 if name in _PPISP_FLOAT2_ATTRS else Sdf.ValueTypeNames.Float
    attr = camera_prim.CreateAttribute(f"ppisp:{name}", value_type)
    attr.Set(Gf.Vec2f(*value) if name in _PPISP_FLOAT2_ATTRS else value)
    return attr


def _author_camera(stage: Usd.Stage, camera_path: str = "/World/Camera") -> Usd.Prim:
    return stage.DefinePrim(camera_path, "Camera")


def _author_ppisp_camera(
    stage: Usd.Stage,
    camera_path: str = "/World/Camera_ppisp",
    *,
    inherits: str | None = "/World/Camera",
    attrs: dict | None = None,
    controller_weights: list[float] | None = None,
) -> Usd.Prim:
    camera_prim = _author_camera(stage, camera_path)
    if inherits is not None:
        _author_camera(stage, inherits)
        camera_prim.GetInherits().AddInherit(Sdf.Path(inherits))

    for name, value in (attrs or {}).items():
        _author_ppisp_attr(camera_prim, name, value)

    if controller_weights is not None:
        camera_prim.CreateAttribute("ppisp:controllerWeights", Sdf.ValueTypeNames.FloatArray).Set(
            Vt.FloatArray(controller_weights)
        )
    return camera_prim


def _controller_weights() -> list[float]:
    return [0.0] * PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN


def test_ppisp_camera_attr_import_uses_first_time_sample():
    stage = Usd.Stage.CreateInMemory()
    ppisp_camera = _author_ppisp_camera(stage, attrs={})

    exposure = ppisp_camera.CreateAttribute("ppisp:exposureOffset", Sdf.ValueTypeNames.Float)
    exposure.Set(1.0)
    exposure.Set(2.0, 10.0)
    exposure.Set(3.0, 20.0)

    color = ppisp_camera.CreateAttribute("ppisp:colorLatentBlue", Sdf.ValueTypeNames.Float2)
    color.Set(Gf.Vec2f(0.0, 0.0))
    color.Set(Gf.Vec2f(0.1, 0.2), 5.0)

    cfg = ppisp_cfg_from_usd_camera(ppisp_camera)

    assert cfg.camera_prim_path is None
    assert cfg.inputs["exposureOffset"] == 2.0
    assert cfg.inputs["colorLatentBlue"] == pytest.approx((0.1, 0.2))


def test_normalize_ppisp_cfg_imports_camera_attrs_from_stage():
    stage = Usd.Stage.CreateInMemory()
    _author_ppisp_camera(
        stage,
        attrs={
            "exposureOffset": 1.5,
            "colorLatentRed": (0.25, -0.5),
        },
    )

    cfg = normalize_ppisp_cfg(PpispCfg(camera_prim_path="/World/Camera_ppisp"), stage=stage)

    assert cfg.camera_prim_path is None
    assert cfg.inputs["exposureOffset"] == 1.5
    assert cfg.inputs["colorLatentRed"] == pytest.approx((0.25, -0.5))


def test_normalize_ppisp_cfg_requires_stage_for_camera_prim_path():
    with pytest.raises(ValueError, match="requires a USD stage"):
        normalize_ppisp_cfg(PpispCfg(camera_prim_path="/World/Camera_ppisp"))


def test_normalize_ppisp_cfg_applies_explicit_overrides_after_camera_attr_import():
    stage = Usd.Stage.CreateInMemory()
    _author_ppisp_camera(
        stage,
        attrs={
            "exposureOffset": 1.5,
            "colorLatentRed": (0.25, -0.5),
        },
    )

    cfg = normalize_ppisp_cfg(
        PpispCfg(
            camera_prim_path="/World/Camera_ppisp",
            inputs={"exposureOffset": 2.0},
        ),
        stage=stage,
    )

    assert cfg.inputs["exposureOffset"] == 2.0
    assert cfg.inputs["colorLatentRed"] == pytest.approx((0.25, -0.5))


def test_normalize_ppisp_cfg_default_list_values_do_not_override_camera_attrs():
    stage = Usd.Stage.CreateInMemory()
    _author_ppisp_camera(stage, attrs={"exposureOffset": 1.5})
    inputs = default_ppisp_inputs()
    inputs["vignettingCenterR"] = [0.0, 0.0]

    cfg = normalize_ppisp_cfg(PpispCfg(camera_prim_path="/World/Camera_ppisp", inputs=inputs), stage=stage)

    assert cfg.inputs["exposureOffset"] == pytest.approx(1.5)


def test_normalize_ppisp_cfg_controller_responsivity_follows_responsivity_override():
    stage = Usd.Stage.CreateInMemory()
    _author_ppisp_camera(
        stage,
        attrs={"responsivity": 2.5},
        controller_weights=_controller_weights(),
    )

    cfg = normalize_ppisp_cfg(
        PpispCfg(
            camera_prim_path="/World/Camera_ppisp",
            inputs={"responsivity": 3.0},
        ),
        stage=stage,
    )

    assert cfg.inputs["responsivity"] == pytest.approx(3.0)
    assert cfg.controller_responsivity == pytest.approx(3.0)


def test_ppisp_cfg_from_usd_camera_requires_camera_attrs():
    stage = Usd.Stage.CreateInMemory()
    camera = _author_camera(stage)

    with pytest.raises(ValueError, match="expected ppisp:\\* attributes"):
        ppisp_cfg_from_usd_camera(camera)


def test_has_ppisp_camera_attrs_ignores_unknown_ppisp_attrs():
    stage = Usd.Stage.CreateInMemory()
    camera = _author_camera(stage)
    camera.CreateAttribute("ppisp:version", Sdf.ValueTypeNames.String).Set("1")

    assert not has_ppisp_camera_attrs(camera)


def test_auto_camera_ppisp_cfg_reads_direct_camera_attrs():
    stage = Usd.Stage.CreateInMemory()
    _author_ppisp_camera(
        stage,
        "/World/Camera",
        inherits=None,
        attrs={
            "responsivity": 0.75,
            "exposureOffset": 1.25,
            "colorLatentBlue": (0.1, 0.2),
        },
    )

    cfg = auto_camera_ppisp_cfg(stage, "/World/Camera")

    assert cfg is not None
    assert cfg.camera_prim_path is None
    assert cfg.inputs["responsivity"] == pytest.approx(0.75)
    assert cfg.inputs["exposureOffset"] == pytest.approx(1.25)
    assert cfg.inputs["colorLatentBlue"] == pytest.approx((0.1, 0.2))


def test_auto_camera_ppisp_cfg_does_not_scan_unmatched_cameras():
    stage = Usd.Stage.CreateInMemory()
    _author_camera(stage, "/World/Camera")
    _author_ppisp_camera(
        stage,
        "/World/Camera_ppisp",
        inherits=None,
        attrs={"exposureOffset": 1.5},
    )

    cfg = auto_camera_ppisp_cfg(stage, "/World/Camera")

    assert cfg is None


def test_auto_any_ppisp_cfg_reads_first_camera_with_ppisp_attrs():
    stage = Usd.Stage.CreateInMemory()
    _author_camera(stage, "/World/CameraWithoutPpisp")
    _author_ppisp_camera(
        stage,
        "/World/CameraB_ppisp",
        inherits=None,
        attrs={"exposureOffset": 2.0},
    )
    _author_ppisp_camera(
        stage,
        "/World/CameraC_ppisp",
        inherits=None,
        attrs={"exposureOffset": 3.0},
    )

    cfg = auto_any_ppisp_cfg(stage)

    assert cfg is not None
    assert cfg.camera_prim_path is None
    assert cfg.inputs["exposureOffset"] == pytest.approx(2.0)


def test_resolve_and_normalize_without_camera_uses_first_ppisp_camera():
    from isaaclab.sensors.camera.camera_isp import CameraISPMode

    stage = Usd.Stage.CreateInMemory()
    _author_camera(stage, "/World/CameraWithoutPpisp")
    _author_ppisp_camera(stage, "/World/Camera_ppisp", inherits=None, attrs={"exposureOffset": 2.0})

    cfg = resolve_and_normalize(CameraISPMode.AUTO_CAMERA, stage)

    assert cfg is not None
    assert cfg.camera_prim_path is None
    assert cfg.inputs["exposureOffset"] == pytest.approx(2.0)


def test_ppisp_cfg_from_usd_camera_reads_controller_weights_from_camera_attrs():
    stage = Usd.Stage.CreateInMemory()
    camera = _author_ppisp_camera(
        stage,
        attrs={
            "responsivity": 2.5,
            "vignettingAlpha1R": 0.25,
            "crfToeB": 0.125,
        },
        controller_weights=_controller_weights(),
    )

    cfg = ppisp_cfg_from_usd_camera(camera)

    assert cfg.inputs["responsivity"] == pytest.approx(2.5)
    assert cfg.inputs["vignettingAlpha1R"] == pytest.approx(0.25)
    assert cfg.inputs["crfToeB"] == pytest.approx(0.125)
    assert cfg.controller_prior_exposure == pytest.approx(0.0)
    assert cfg.controller_responsivity == pytest.approx(2.5)
    assert cfg.controller_weights is not None
    assert len(cfg.controller_weights) == PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN


def test_ppisp_cfg_from_usd_camera_validates_controller_weights_len():
    stage = Usd.Stage.CreateInMemory()
    camera = _author_ppisp_camera(
        stage,
        attrs={"responsivity": 2.5},
        controller_weights=[0.0],
    )

    with pytest.raises(ValueError, match=f"Expected {PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN}"):
        ppisp_cfg_from_usd_camera(camera)
