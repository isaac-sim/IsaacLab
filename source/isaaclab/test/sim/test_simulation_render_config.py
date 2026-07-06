# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Isaac RTX global render settings."""

from __future__ import annotations

import importlib
import os
import sys
import types

import tomllib
from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import (
    IsaacRtxRendererGlobalSettingsCfg,
)
from packaging.version import Version


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


def _flatten_preset(data: dict, prefix: str = "") -> dict[str, object]:
    """Flatten nested preset dictionaries using dot-separated keys."""
    flattened = {}
    for key, value in data.items():
        key_path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_preset(value, key_path))
        else:
            flattened[key_path] = value
    return flattened


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
    assert settings.get("/rtx/raytracing/subpixel/mode") == 1
    assert settings.get("/rtx/raytracing/cached/enabled") is True
    assert settings.get("/rtx/pathtracing/maxSamplesPerLaunch") == 500000
    assert settings.get("/rtx/viewTile/limit") == 500000
    assert rep_settings.antialiasing == "DLAA"


def test_isaac_rtx_global_settings_presets(monkeypatch):
    """Test that Isaac RTX rendering-mode presets apply before overrides."""
    utils = _import_isaac_rtx_utils(monkeypatch)
    import isaaclab.utils.version as version_utils

    monkeypatch.setattr(version_utils, "get_isaac_sim_version", lambda: Version("6.0.0"))

    carb_settings = {
        "/rtx/raytracing/subpixel/mode": 3,
        "/rtx/pathtracing/maxSamplesPerLaunch": 999999,
    }
    dlss_mode = ("/rtx/post/dlss/execMode", 5)

    rendering_modes = ["performance", "balanced", "quality"]

    for rendering_mode in rendering_modes:
        settings = _FakeSettings()
        monkeypatch.setattr(utils, "get_settings_manager", lambda: settings)
        isaaclab_app_exp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4, "apps")
        preset_filename = os.path.join(isaaclab_app_exp_path, f"rendering_modes/{rendering_mode}.kit")
        with open(preset_filename, "rb") as file:
            preset_dict = tomllib.load(file)
        preset_dict = _flatten_preset(preset_dict)

        global_settings = IsaacRtxRendererGlobalSettingsCfg(
            rendering_mode=rendering_mode,
            dlss_mode=dlss_mode[1],
            carb_settings=carb_settings,
        )
        utils.apply_isaac_rtx_global_settings(global_settings)

        for key, val in preset_dict.items():
            setting_name = "/" + key.replace(".", "/")
            if setting_name in carb_settings:
                setting_gt = carb_settings[setting_name]
            elif setting_name == dlss_mode[0]:
                setting_gt = dlss_mode[1]
            else:
                setting_gt = val

            setting_val = settings.get(setting_name)

            assert setting_gt == setting_val, (
                f"Mismatch for '{setting_name}' in mode '{rendering_mode}': "
                f"expected {setting_gt!r}, got {setting_val!r}"
            )
