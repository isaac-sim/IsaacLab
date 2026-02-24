# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Literal

from isaaclab.utils import configclass


@configclass
class RenderingQualityCfg:
    """Shared rendering quality profile for visualizers and renderers.

    This profile keeps backend-specific fields in one place using explicit prefixes:
    - 'kit_*' for Omniverse/RTX quality controls
    - 'newton_*' for Newton visual quality controls
    """

    kit_rendering_preset: Literal["performance", "balanced", "high"] | None = None
    """Optional built-in preset profile.

    Preset values are defined in 'isaaclab.rendering.rendering_quality.rendering_quality_presets'.
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

    newton_enable_shadows: bool | None = None
    """Overrides Newton visualizer shadow rendering."""

    newton_enable_sky: bool | None = None
    """Overrides Newton visualizer sky rendering."""

    newton_enable_wireframe: bool | None = None
    """Overrides Newton visualizer wireframe rendering."""

    newton_sky_upper_color: tuple[float, float, float] | None = None
    """Overrides Newton visualizer upper sky color."""

    newton_sky_lower_color: tuple[float, float, float] | None = None
    """Overrides Newton visualizer lower sky color."""

    newton_light_color: tuple[float, float, float] | None = None
    """Overrides Newton visualizer light color."""

    # TODO: Consider supporting additional raw backend settings dictionaries.
