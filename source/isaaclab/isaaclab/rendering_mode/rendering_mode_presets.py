# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Built-in rendering mode presets for RTX/Kit rendering.

Presets are sourced from the latest Isaac Lab app rendering profiles (apps/rendering_modes).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# Latest preset values sourced from apps/rendering_modes/*.kit.
_KIT_PRESETS: dict[str, dict[str, Any]] = {
    "performance": {
        "/rtx/rtpt/cached/enabled": False,
        "/rtx/rtpt/lightcache/cached/enabled": False,
        "/rtx/rtpt/translucency/virtualMotion/enabled": False,
        "/rtx/rtpt/maxBounces": 2,
        "/rtx/rtpt/splitGlass": False,
        "/rtx/rtpt/splitClearcoat": False,
        "/rtx/rtpt/splitRoughReflection": True,
        "/rtx/rtpt/useAmbientOcclusionForAmbientLight": False,
        "/rtx/sceneDb/ambientLightIntensity": 1.0,
        "/rtx/shadows/enabled": True,
        "/rtx/domeLight/upperLowerStrategy": 3,
        "/rtx/ambientOcclusion/enabled": False,
        "/rtx/ambientOcclusion/denoiserMode": 1,
        "/rtx/raytracing/subpixel/mode": 0,
        "/rtx/raytracing/cached/enabled": False,
        "/rtx-transient/dlssg/enabled": False,
        "/rtx-transient/dldenoiser/enabled": False,
        "/rtx/post/dlss/execMode": 0,
        "/rtx/pathtracing/maxSamplesPerLaunch": 1_000_000,
        "/rtx/viewTile/limit": 1_000_000,
    },
    "balanced": {
        "/rtx/rtpt/cached/enabled": False,
        "/rtx/rtpt/lightcache/cached/enabled": False,
        "/rtx/rtpt/translucency/virtualMotion/enabled": False,
        "/rtx/rtpt/maxBounces": 2,
        "/rtx/rtpt/splitGlass": False,
        "/rtx/rtpt/splitClearcoat": False,
        "/rtx/rtpt/splitRoughReflection": True,
        "/rtx/rtpt/useAmbientOcclusionForAmbientLight": False,
        "/rtx/sceneDb/ambientLightIntensity": 1.0,
        "/rtx/shadows/enabled": True,
        "/rtx/ambientOcclusion/enabled": False,
        "/rtx/ambientOcclusion/denoiserMode": 1,
        "/rtx/raytracing/subpixel/mode": 0,
        "/rtx/raytracing/cached/enabled": True,
        "/rtx-transient/dlssg/enabled": False,
        "/rtx-transient/dldenoiser/enabled": True,
        "/rtx/post/dlss/execMode": 1,
        "/rtx/pathtracing/maxSamplesPerLaunch": 1_000_000,
        "/rtx/viewTile/limit": 1_000_000,
    },
    "quality": {
        "/rtx/rtpt/maxBounces": 3,
        "/rtx/rtpt/cached/enabled": False,
        "/rtx/rtpt/lightcache/cached/enabled": False,
        "/rtx/rtpt/translucency/virtualMotion/enabled": False,
        "/rtx/rtpt/splitRoughReflection": True,
        "/rtx/rtpt/adaptiveSampling/disocclusion/enabled": True,
        "/rtx/rtpt/adaptiveSampling/disocclusion/spp": 4,
        "/rtx/sceneDb/ambientLightIntensity": 1.0,
        "/rtx/shadows/enabled": True,
        "/rtx/ambientOcclusion/enabled": True,
        "/rtx/ambientOcclusion/denoiserMode": 0,
        "/rtx/raytracing/subpixel/mode": 1,
        "/rtx/raytracing/cached/enabled": True,
        "/rtx-transient/dlssg/enabled": False,
        "/rtx/post/dlss/execMode": 2,
        "/rtx/pathtracing/maxSamplesPerLaunch": 1_000_000,
        "/rtx/viewTile/limit": 1_000_000,
    },
}


def get_kit_rendering_preset(preset_name: str) -> dict[str, Any]:
    """Return a deep copy of the requested rendering preset."""
    if preset_name not in {"performance", "balanced", "quality"}:
        raise ValueError(f"Unknown preset '{preset_name}'.")
    return deepcopy(_KIT_PRESETS[preset_name])
