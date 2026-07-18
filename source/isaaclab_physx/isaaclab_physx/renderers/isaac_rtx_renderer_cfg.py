# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Isaac RTX (Replicator) Renderer."""

from typing import Any, Literal

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.utils.configclass import configclass


@configclass
class IsaacRtxRendererGlobalSettingsCfg:
    """Global Isaac RTX renderer settings.

    These settings are applied to Kit/RTX carb settings. They are carried by
    :class:`IsaacRtxRendererCfg` so they are owned by the Isaac RTX backend, but
    they affect the process-global RTX renderer rather than a single camera.
    """

    enable_translucency: bool | None = None
    """Enable translucency for specular transmissive surfaces such as glass."""

    enable_reflections: bool | None = None
    """Enable reflections."""

    enable_global_illumination: bool | None = None
    """Enable diffuse global illumination."""

    antialiasing_mode: Literal["Off", "FXAA", "DLSS", "TAA", "DLAA"] | None = None
    """Anti-aliasing mode selected through Replicator."""

    enable_dlssg: bool | None = None
    """Enable DLSS frame generation."""

    enable_dl_denoiser: bool | None = None
    """Enable DL denoiser."""

    dlss_mode: Literal[0, 1, 2, 3] | None = None
    """DLSS performance/quality mode."""

    enable_direct_lighting: bool | None = None
    """Enable direct light contributions from lights."""

    samples_per_pixel: int | None = None
    """Direct lighting samples per pixel."""

    enable_shadows: bool | None = None
    """Enable shadow rendering."""

    enable_ambient_occlusion: bool | None = None
    """Enable ambient occlusion."""

    dome_light_upper_lower_strategy: Literal[0, 3, 4] | None = None
    """Dome light sampling strategy."""

    max_bounces: int | None = None
    """Maximum number of ray bounces for RTPT."""

    split_glass: bool | None = None
    """Enable separate glass ray splitting."""

    split_clearcoat: bool | None = None
    """Enable separate clearcoat ray splitting."""

    split_rough_reflection: bool | None = None
    """Enable separate rough-reflection ray splitting."""

    ambient_light_intensity: float | None = None
    """Scene ambient light intensity."""

    ambient_occlusion_denoiser_mode: Literal[0, 1] | None = None
    """Ambient occlusion denoiser mode."""

    subpixel_mode: int | None = None
    """RTX subpixel mode."""

    enable_cached_raytracing: bool | None = None
    """Enable cached ray tracing."""

    max_samples_per_launch: int | None = None
    """Path tracing maximum samples per launch."""

    view_tile_limit: int | None = None
    """Maximum number of view tiles."""

    carb_settings: dict[str, Any] | None = None
    """Raw carb settings applied after named fields."""

    rendering_mode: Literal["performance", "balanced", "quality"] | None = None
    """Legacy rendering-mode preset to apply before field overrides."""


@configclass
class IsaacRtxRendererCfg(RendererCfg):
    """Configuration for Isaac RTX renderer using Omniverse Replicator.

    Holds the Replicator/RTX-specific knobs (semantic segmentation, instance
    segmentation, semantic filtering, depth clipping) used by the RTX rendering
    pipeline.
    """

    renderer_type: str = "isaac_rtx"
    """Type identifier for Isaac RTX renderer."""

    global_settings: IsaacRtxRendererGlobalSettingsCfg = IsaacRtxRendererGlobalSettingsCfg()
    """Global Kit/RTX quality settings applied before RTX Hydra attach."""

    semantic_filter: str | list[str] = "*:*"
    """A string or a list specifying a semantic filter predicate. Defaults to ``"*:*"``.

    If a string, it should be a disjunctive normal form of (semantic type, labels). For examples:

    * ``"typeA : labelA & !labelB | labelC , typeB: labelA ; typeC: labelE"``:
      All prims with semantic type "typeA" and label "labelA" but not "labelB" or with label "labelC".
      Also, all prims with semantic type "typeB" and label "labelA", or with semantic type "typeC" and label "labelE".
    * ``"typeA : * ; * : labelA"``: All prims with semantic type "typeA" or with label "labelA"

    If a list of strings, each string should be a semantic type. The segmentation for prims with
    semantics of the specified types will be retrieved. For example, if the list is ["class"], only
    the segmentation for prims with semantics of type "class" will be retrieved.

    .. seealso::

        For more information on the semantics filter, see the documentation on `Replicator Semantics Schema Editor`_.

    .. _Replicator Semantics Schema Editor: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/semantics_schema_editor.html#semantics-filtering
    """

    colorize_semantic_segmentation: bool = True
    """Whether to colorize the semantic segmentation images. Defaults to True.

    If True, semantic segmentation is converted to an image where semantic IDs are mapped to colors
    and returned as a ``uint8`` 4-channel array. If False, the output is returned as a ``int32`` array.
    """

    colorize_instance_id_segmentation: bool = True
    """Whether to colorize the instance ID segmentation images. Defaults to True.

    If True, instance id segmentation is converted to an image where instance IDs are mapped to colors.
    and returned as a ``uint8`` 4-channel array. If False, the output is returned as a ``int32`` array.
    """

    colorize_instance_segmentation: bool = True
    """Whether to colorize the instance segmentation images. Defaults to True.

    If True, instance segmentation is converted to an image where instance IDs are mapped to colors.
    and returned as a ``uint8`` 4-channel array. If False, the output is returned as a ``int32`` array.
    """

    semantic_segmentation_mapping: dict = {}
    """Dictionary mapping semantics to specific colours

    Eg.

    .. code-block:: python

        {
            "class:cube_1": (255, 36, 66, 255),
            "class:cube_2": (255, 184, 48, 255),
            "class:cube_3": (55, 255, 139, 255),
            "class:table": (255, 237, 218, 255),
            "class:ground": (100, 100, 100, 255),
            "class:robot": (61, 178, 255, 255),
        }

    """

    depth_clipping_behavior: Literal["max", "zero", "none"] = "none"
    """Clipping behavior for the camera for values exceed the maximum value. Defaults to "none".

    - ``"max"``: Values are clipped to the maximum value.
    - ``"zero"``: Values are clipped to zero.
    - ``"none"``: No clipping is applied. Values will be returned as ``inf``.
    """
