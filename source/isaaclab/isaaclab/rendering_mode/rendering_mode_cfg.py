# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Literal

from isaaclab.utils import configclass


@configclass
class RenderingModeCfg:
    """Shared rendering mode profile for renderers and visualizers.

    This profile keeps Omniverse/RTX controls in one place using ``kit_*`` fields.
    """

    rendering_mode_preset: Literal["performance", "balanced", "quality"] | None = None
    """Optional built-in preset profile.

    Preset values are defined in 'isaaclab.rendering_mode.rendering_mode_presets'.
    """

    kit_enable_translucency: bool | None = None
    """Maps to '/rtx/translucency/enabled'."""

    kit_enable_reflections: bool | None = None
    """Maps to '/rtx/reflections/enabled'."""

    kit_enable_global_illumination: bool | None = None
    """Maps to '/rtx/indirectDiffuse/enabled'."""

    kit_antialiasing_mode: Literal["Off", "FXAA", "DLSS", "TAA", "DLAA"] | None = None
    """Optional anti-aliasing mode applied via Replicator settings helper."""

    kit_enable_dlssg: bool | None = None
    """Maps to '/rtx-transient/dlssg/enabled'."""

    kit_enable_dl_denoiser: bool | None = None
    """Maps to '/rtx-transient/dldenoiser/enabled'."""

    kit_dlss_mode: Literal[0, 1, 2, 3] | None = None
    """Maps to '/rtx/post/dlss/execMode'."""

    kit_enable_direct_lighting: bool | None = None
    """Maps to '/rtx/directLighting/enabled'."""

    kit_samples_per_pixel: int | None = None
    """Maps to '/rtx/directLighting/sampledLighting/samplesPerPixel'."""

    kit_enable_shadows: bool | None = None
    """Maps to '/rtx/shadows/enabled'."""

    kit_enable_ambient_occlusion: bool | None = None
    """Maps to '/rtx/ambientOcclusion/enabled'."""

    kit_dome_light_upper_lower_strategy: Literal[0, 3, 4] | None = None
    """Maps to '/rtx/domeLight/upperLowerStrategy'."""

    # TODO: Consider supporting additional raw backend settings dictionaries.
