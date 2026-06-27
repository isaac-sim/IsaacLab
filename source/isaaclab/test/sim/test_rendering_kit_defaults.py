# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that the high-fidelity RTX defaults live only in the rendering experience files."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_APPS_DIR = Path(__file__).resolve().parents[4] / "apps"

# Experience files loaded when camera (RGB) rendering is enabled.
_RENDERING_KITS = (
    "isaaclab.python.rendering.kit",
    "isaaclab.python.headless.rendering.kit",
)

# Experience file loaded when cameras are disabled.
_BASE_KIT = "isaaclab.python.kit"

# Representative high-fidelity RTX defaults, mirroring the runtime checks in test_simulation_render_config.py.
_HIGH_FIDELITY_DEFAULTS = (
    "rtx.rtpt.maxBounces",
    "rtx.shadows.enabled",
    "rtx.ambientOcclusion.enabled",
    "rtx.raytracing.cached.enabled",
    "rtx.viewTile.limit",
)

# Full set of high-fidelity RTX defaults that must stay byte-for-byte identical across both rendering
# kits. They are maintained in parallel in each experience file, so this guards against silent drift.
_SHARED_RTX_DEFAULTS = (
    "rtx.translucency.enabled",
    "rtx.reflections.enabled",
    "rtx.indirectDiffuse.enabled",
    "rtx-transient.dlssg.enabled",
    "rtx.directLighting.sampledLighting.enabled",
    "rtx.directLighting.sampledLighting.samplesPerPixel",
    "rtx.sceneDb.ambientLightIntensity",
    "rtx.shadows.enabled",
    "rtx.rtpt.maxBounces",
    "rtx.rtpt.cached.enabled",
    "rtx.rtpt.lightcache.cached.enabled",
    "rtx.rtpt.translucency.virtualMotion.enabled",
    "rtx.rtpt.splitRoughReflection",
    "rtx.rtpt.adaptiveSampling.disocclusion.enabled",
    "rtx.rtpt.adaptiveSampling.disocclusion.spp",
    "rtx.pathtracing.maxSamplesPerLaunch",
    "rtx.viewTile.limit",
    "rtx.raytracing.cached.enabled",
    "rtx.raytracing.subpixel.mode",
    "rtx.ambientOcclusion.enabled",
    "rtx.ambientOcclusion.denoiserMode",
    "rtx.post.dlss.execMode",
)


def _read_kit(name: str) -> str:
    """Return the contents of the experience file ``name`` under ``apps/``."""
    return (_APPS_DIR / name).read_text()


def _kit_defines(content: str, key: str) -> bool:
    """Return whether the experience-file ``content`` assigns the dotted carb ``key``."""
    return re.search(rf"^\s*{re.escape(key)}\s*=", content, re.MULTILINE) is not None


def _kit_value(content: str, key: str) -> str | None:
    """Return the right-hand-side value assigned to ``key`` in ``content``, or ``None`` if unset."""
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*$", content, re.MULTILINE)
    return match.group(1) if match else None


@pytest.mark.parametrize("kit_name", _RENDERING_KITS)
def test_rendering_kits_define_high_fidelity_defaults(kit_name):
    """Rendering experience files carry the high-fidelity RTX defaults."""
    content = _read_kit(kit_name)
    missing = [key for key in _HIGH_FIDELITY_DEFAULTS if not _kit_defines(content, key)]
    assert not missing, f"{kit_name} is missing high-fidelity RTX defaults: {missing}"


def test_base_kit_omits_high_fidelity_defaults():
    """Base (cameras-disabled) experience file omits the high-fidelity RTX defaults."""
    content = _read_kit(_BASE_KIT)
    present = [key for key in _HIGH_FIDELITY_DEFAULTS if _kit_defines(content, key)]
    assert not present, f"{_BASE_KIT} unexpectedly defines high-fidelity RTX defaults: {present}"


def test_rendering_kits_agree_on_rtx_defaults():
    """Both rendering experience files define identical values for every shared RTX default (drift guard)."""
    primary, secondary = _RENDERING_KITS
    primary_content, secondary_content = _read_kit(primary), _read_kit(secondary)
    mismatches = {}
    for key in _SHARED_RTX_DEFAULTS:
        primary_value = _kit_value(primary_content, key)
        secondary_value = _kit_value(secondary_content, key)
        if primary_value != secondary_value:
            mismatches[key] = {primary: primary_value, secondary: secondary_value}
    assert not mismatches, f"RTX defaults drifted between rendering kits: {mismatches}"
