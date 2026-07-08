# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Isaac RTX global render settings."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import (
    IsaacRtxRendererGlobalSettingsCfg,
)

_QUALITY_DEFAULT_SETTINGS = {
    "rtx.translucency.enabled": True,
    "rtx.reflections.enabled": True,
    "rtx.indirectDiffuse.enabled": True,
    "rtx.rtpt.maxBounces": 3,
    "rtx.rtpt.cached.enabled": False,
    "rtx.rtpt.lightcache.cached.enabled": False,
    "rtx.rtpt.translucency.virtualMotion.enabled": False,
    "rtx.rtpt.splitRoughReflection": True,
    "rtx.rtpt.adaptiveSampling.disocclusion.enabled": True,
    "rtx.rtpt.adaptiveSampling.disocclusion.spp": 4,
    "rtx.sceneDb.ambientLightIntensity": 1.0,
    "rtx.shadows.enabled": True,
    "rtx.ambientOcclusion.enabled": True,
    "rtx.ambientOcclusion.denoiserMode": 0,
    "rtx.raytracing.subpixel.mode": 1,
    "rtx.raytracing.cached.enabled": True,
    "rtx-transient.dlssg.enabled": False,
    "rtx.post.dlss.execMode": 2,
    "rtx.pathtracing.maxSamplesPerLaunch": 1000000,
    "rtx.viewTile.limit": 1000000,
}

pytestmark = [pytest.mark.integration, pytest.mark.rendering]


class _FakeSettings:
    """Small settings manager double used by the Isaac RTX settings helper."""

    def __init__(self):
        self.values = {}

    def get(self, name: str):
        return self.values.get(name)

    def set(self, name: str, value):
        self.values[name] = value


class _FakeReplicatorSettings:
    """Replicator settings double for anti-aliasing calls."""

    def __init__(self):
        self.antialiasing = None

    def set_render_rtx_realtime(self, antialiasing):
        self.antialiasing = antialiasing


def _install_omni_stubs(monkeypatch) -> _FakeReplicatorSettings:
    """Install the Omniverse modules imported by the Isaac RTX utility."""
    omni_mod = types.ModuleType("omni")
    omni_mod.__path__ = []
    usd_mod = types.ModuleType("omni.usd")
    replicator_mod = types.ModuleType("omni.replicator")
    core_mod = types.ModuleType("omni.replicator.core")

    rep_settings = _FakeReplicatorSettings()
    setattr(core_mod, "settings", rep_settings)
    setattr(omni_mod, "usd", usd_mod)
    setattr(omni_mod, "replicator", replicator_mod)
    setattr(replicator_mod, "core", core_mod)

    monkeypatch.setitem(sys.modules, "omni", omni_mod)
    monkeypatch.setitem(sys.modules, "omni.usd", usd_mod)
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator_mod)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", core_mod)
    return rep_settings


def _import_isaac_rtx_utils(monkeypatch):
    """Import Isaac RTX utilities after installing Kit/Replicator stubs."""
    _install_omni_stubs(monkeypatch)
    return importlib.import_module("isaaclab_physx.renderers.isaac_rtx_renderer_utils")


def _parse_kit_setting_values(path: Path) -> dict[str, object]:
    """Parse scalar Kit setting assignments used by this test."""
    values = {}
    expected_names = set(_QUALITY_DEFAULT_SETTINGS)
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, raw_value = (item.strip() for item in line.split("=", 1))
        if key not in expected_names:
            continue
        if raw_value in {"true", "false"}:
            values[key] = raw_value == "true"
        elif "." in raw_value:
            values[key] = float(raw_value)
        else:
            values[key] = int(raw_value)
    return values


def test_isaac_rtx_global_settings(monkeypatch):
    """Test that Isaac RTX global settings are applied by the helper."""
    rep_settings = _install_omni_stubs(monkeypatch)
    utils = importlib.import_module("isaaclab_physx.renderers.isaac_rtx_renderer_utils")
    settings = _FakeSettings()
    monkeypatch.setattr(utils, "get_settings_manager", lambda: settings)
    global_settings = IsaacRtxRendererGlobalSettingsCfg(
        enable_translucency=True,
        enable_reflections=True,
        enable_global_illumination=True,
        antialiasing_mode="DLAA",
        enable_dlssg=True,
        enable_dl_denoiser=True,
        dlss_mode=0,
        enable_direct_lighting=True,
        samples_per_pixel=4,
        enable_shadows=True,
        enable_ambient_occlusion=True,
        max_bounces=4,
        split_glass=True,
        split_clearcoat=True,
        split_rough_reflection=True,
        ambient_light_intensity=0.5,
        ambient_occlusion_denoiser_mode=0,
        subpixel_mode=1,
        enable_cached_raytracing=True,
        max_samples_per_launch=500000,
        view_tile_limit=500000,
        carb_settings={
            "rtx.raytracing.subpixel.mode": 2,
            "/rtx/custom/setting": True,
        },
    )

    utils.apply_isaac_rtx_global_settings(global_settings)

    assert settings.get("/rtx/translucency/enabled") is True
    assert settings.get("/rtx/reflections/enabled") is True
    assert settings.get("/rtx/indirectDiffuse/enabled") is True
    assert settings.get("/rtx-transient/dlssg/enabled") is True
    assert settings.get("/rtx-transient/dldenoiser/enabled") is True
    assert settings.get("/rtx/post/dlss/execMode") == 0
    assert settings.get("/rtx/directLighting/enabled") is True
    assert settings.get("/rtx/directLighting/sampledLighting/samplesPerPixel") == 4
    assert settings.get("/rtx/shadows/enabled") is True
    assert settings.get("/rtx/ambientOcclusion/enabled") is True
    assert settings.get("/rtx/rtpt/maxBounces") == 4
    assert settings.get("/rtx/rtpt/splitGlass") is True
    assert settings.get("/rtx/rtpt/splitClearcoat") is True
    assert settings.get("/rtx/rtpt/splitRoughReflection") is True
    assert settings.get("/rtx/sceneDb/ambientLightIntensity") == 0.5
    assert settings.get("/rtx/ambientOcclusion/denoiserMode") == 0
    assert settings.get("/rtx/raytracing/subpixel/mode") == 2
    assert settings.get("/rtx/raytracing/cached/enabled") is True
    assert settings.get("/rtx/pathtracing/maxSamplesPerLaunch") == 500000
    assert settings.get("/rtx/viewTile/limit") == 500000
    assert settings.get("/rtx/custom/setting") is True
    assert rep_settings.antialiasing == "DLAA"


def test_isaac_rtx_camera_experience_defaults():
    """Test that camera app files carry the high-fidelity RTX defaults."""
    isaaclab_app_exp_path = Path(__file__).resolve().parents[4] / "apps"

    for app_filename in (
        "isaaclab.python.rendering.kit",
        "isaaclab.python.headless.rendering.kit",
    ):
        app_settings = _parse_kit_setting_values(isaaclab_app_exp_path / app_filename)

        for setting_name, expected_value in _QUALITY_DEFAULT_SETTINGS.items():
            assert setting_name in app_settings, f"'{setting_name}' is not defined in '{app_filename}'"
            assert app_settings[setting_name] == expected_value, (
                f"Mismatch for '{setting_name}' in '{app_filename}': "
                f"expected {expected_value!r}, "
                f"got {app_settings[setting_name]!r}"
            )
