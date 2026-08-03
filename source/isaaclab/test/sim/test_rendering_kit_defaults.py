# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""App-free checks that the high-fidelity RTX defaults live only in the *.rendering.kit and stay in sync."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_APPS_DIR = Path(__file__).resolve().parents[4] / "apps"

# Kit loaded when cameras are enabled.
_RENDERING_KIT = (
    "isaaclab.python.rendering.kit",
    "isaaclab.python.headless.rendering.kit",
)

# Kit loaded when cameras are disabled.
_BASE_KIT = "isaaclab.python.kit"

# Defaults exclusive to the *.rendering.kit; keys also present in the *.kit are excluded so the
# negative test stays meaningful.
_HIGH_FIDELITY_DEFAULTS = (
    "rtx.rtpt.maxBounces",
    "rtx.shadows.enabled",
    "rtx.ambientOcclusion.enabled",
    "rtx.raytracing.cached.enabled",
    "rtx.viewTile.limit",
)

# Defaults that must stay identical across both rendering kits (drift guard).
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
    """Return the text of kit ``name`` under ``apps/``."""
    return (_APPS_DIR / name).read_text()


def _kit_defines(content: str, key: str) -> bool:
    """Return whether ``content`` assigns ``key``."""
    return re.search(rf"^\s*{re.escape(key)}\s*=", content, re.MULTILINE) is not None


def _kit_value(content: str, key: str) -> str | None:
    """Return the value assigned to ``key`` (comment stripped), or ``None`` if unset."""
    match = re.search(rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*$", content, re.MULTILINE)
    if match is None:
        return None
    return match.group(1).split("#", 1)[0].strip()


@pytest.mark.parametrize("kit_name", _RENDERING_KIT)
def test_rendering_kits_define_high_fidelity_defaults(kit_name):
    """Rendering kits carry the high-fidelity RTX defaults."""
    content = _read_kit(kit_name)
    missing = [key for key in _HIGH_FIDELITY_DEFAULTS if not _kit_defines(content, key)]
    assert not missing, f"{kit_name} is missing high-fidelity RTX defaults: {missing}"


def test_base_kit_omits_high_fidelity_defaults():
    """Base kit (cameras disabled) omits the high-fidelity RTX defaults."""
    content = _read_kit(_BASE_KIT)
    present = [key for key in _HIGH_FIDELITY_DEFAULTS if _kit_defines(content, key)]
    assert not present, f"{_BASE_KIT} unexpectedly defines high-fidelity RTX defaults: {present}"


def test_rendering_kits_agree_on_rtx_defaults():
    """Both rendering kits define identical values for every shared RTX default."""
    primary, secondary = _RENDERING_KIT
    primary_content, secondary_content = _read_kit(primary), _read_kit(secondary)
    mismatches = {}
    for key in _SHARED_RTX_DEFAULTS:
        primary_value = _kit_value(primary_content, key)
        secondary_value = _kit_value(secondary_content, key)
        if primary_value != secondary_value:
            mismatches[key] = {primary: primary_value, secondary: secondary_value}
    assert not mismatches, f"RTX defaults drifted between rendering kits: {mismatches}"
