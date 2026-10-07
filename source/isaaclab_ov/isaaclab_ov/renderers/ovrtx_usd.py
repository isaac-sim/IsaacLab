# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD manipulation for OVRTX: Render scope building, camera injection, and stage prim activation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from pxr import Gf, Sdf, Usd, UsdGeom

if TYPE_CHECKING:
    from isaaclab.renderers.camera_render_spec import CameraRenderSpec

    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXCameraRenderData

logger = logging.getLogger(__name__)


# Maps camera data types to (prim name, source name). Shared sources use one config.
# OVRTX 0.4 keys ``frame.render_vars`` by source name; 0.5+ keys them by the RenderVar prim path.
_RENDER_VAR_BY_DATA_TYPE: dict[str, tuple[str, str]] = {
    "rgb": ("LdrColor", "LdrColor"),
    "rgba": ("LdrColor", "LdrColor"),
    # Simple shading uses LdrColor in per-product RTX Minimal mode.
    "simple_shading_constant_diffuse": ("LdrColor", "LdrColor"),
    "simple_shading_diffuse_mdl": ("LdrColor", "LdrColor"),
    "simple_shading_full_mdl": ("LdrColor", "LdrColor"),
    "rgb_hdr": ("HdrColor", "HdrColor"),
    "albedo": ("albedo", "DiffuseAlbedoSD"),
    "depth": ("depth", "DistanceToImagePlaneSD"),
    "distance_to_image_plane": ("depth", "DistanceToImagePlaneSD"),
    # This source requires a distinct render-var prim.
    "distance_to_camera": ("DistanceToCameraSD", "DistanceToCameraSD"),
    "normals": ("NormalSD", "NormalSD"),
    "motion_vectors": ("TargetMotionSD", "TargetMotionSD"),
    "semantic_segmentation": ("semantic", "SemanticSegmentation"),
    "instance_segmentation": (
        "NonStableInstanceSegmentation",
        "NonStableInstanceSegmentation",
    ),
}

# Maps simple shading data types to the render product's ``omni:rtx:minimal:mode``.
_RTX_MINIMAL_MODES = {
    "simple_shading_constant_diffuse": 1,
    "simple_shading_diffuse_mdl": 2,
    "simple_shading_full_mdl": 3,
}

_COLOR_DATA_TYPES = frozenset({"rgb", "rgba"})

# Segmentation ID-map vars are authored alongside the pixel AOVs, not as camera data types.
_SEGMENTATION_MAP_RENDER_VARS: tuple[tuple[str, str], ...] = (
    ("StableIdSemanticIdMap", "StableIdSemanticIdMap"),
    ("StableIdMap", "StableIdMap"),
    ("SemanticIdMap", "SemanticIdMap"),
)

_RENDER_VAR_PRIM_NAME_BY_SOURCE: Mapping[str, str] = MappingProxyType(
    {source: name for name, source in (*_RENDER_VAR_BY_DATA_TYPE.values(), *_SEGMENTATION_MAP_RENDER_VARS)}
)


def get_render_var_configs(data_types: list[str], render_scope_name: str) -> list[tuple[str, str, str]]:
    """Return render-var configs for the requested camera outputs.

    Shared sources are de-duplicated. Segmentation requests also add their ID-map vars.

    Args:
        data_types: Requested camera data types.
        render_scope_name: Root scope containing the render product and its render vars.

    Returns:
        Render-var configs as (absolute prim path, prim name, source name) tuples.

    Raises:
        ValueError: If no outputs are requested, an output is unsupported, or outputs are incompatible.
    """
    if not data_types:
        raise ValueError("OVRTX render products require at least one output.")
    unsupported = set(data_types) - _RENDER_VAR_BY_DATA_TYPE.keys()
    if unsupported:
        raise ValueError(f"Unsupported OVRTX output types: {sorted(unsupported)}.")
    simple_shading = list(dict.fromkeys(data_type for data_type in data_types if data_type in _RTX_MINIMAL_MODES))
    color = list(dict.fromkeys(data_type for data_type in data_types if data_type in _COLOR_DATA_TYPES))

    if simple_shading and color:
        raise ValueError(
            f"OVRTX cannot render simple shading {simple_shading} together with {color} on one render product:"
            " both read the 'LdrColor' render var, and simple shading additionally requires RTX Minimal mode."
            " Request them from separate cameras."
        )
    if len(simple_shading) > 1:
        raise ValueError(
            f"OVRTX supports at most one simple shading data type per render product, got {simple_shading}."
            " RTX Minimal mode is a per-render-product setting. Request them from separate cameras."
        )

    render_vars = list(dict.fromkeys(_RENDER_VAR_BY_DATA_TYPE[data_type] for data_type in data_types))

    # Author the ID-to-label map render vars needed to decode the segmentation info dicts.
    # instance_segmentation needs StableIdSemanticIdMap + StableIdMap to resolve each pixel to a prim path.
    if "instance_segmentation" in data_types:
        render_vars.append(_SEGMENTATION_MAP_RENDER_VARS[0])
        render_vars.append(_SEGMENTATION_MAP_RENDER_VARS[1])
    # SemanticIdMap resolves the semantic-ID-to-label mapping and is shared by both semantic_segmentation and
    # instance_segmentation, so it is authored once when either output is requested.
    if "semantic_segmentation" in data_types or "instance_segmentation" in data_types:
        render_vars.append(_SEGMENTATION_MAP_RENDER_VARS[2])
    return [(f"/{render_scope_name}/Vars/{name}", name, source) for name, source in render_vars]


def render_var_prim_names_by_source() -> Mapping[str, str]:
    """Return the scope-independent RenderVar prim name of every OVRTX render-var source.

    Returns:
        Read-only mapping of render-var source name to its RenderVar prim name.
    """
    return _RENDER_VAR_PRIM_NAME_BY_SOURCE


def render_var_prim_paths_by_source(render_scope_name: str) -> Mapping[str, str]:
    """Return the authored RenderVar prim path of every OVRTX render-var source.

    Args:
        render_scope_name: Root scope containing the render product and its render vars.

    Returns:
        Read-only mapping of render-var source name to the absolute path of the ``RenderVar``
        prim this module authors for it.
    """
    return MappingProxyType(
        {source: f"/{render_scope_name}/Vars/{name}" for source, name in _RENDER_VAR_PRIM_NAME_BY_SOURCE.items()}
    )


def build_render_product_as_string(
    spec: CameraRenderSpec,
    render_data: OVRTXCameraRenderData,
    *,
    device_id: int,
    enable_shadows: bool = False,
) -> str:
    """Build a complete render product USD layer as a string.

    The layer sets the camera scope as its default prim for referencing into OVRTX.
    The initial camera relationship targets only environment zero, whose camera is guaranteed to
    exist in the trimmed stage. Multi-environment rendering rewrites the relationship with every
    resolved camera path after runtime cloning.

    Args:
        spec: Camera configuration, environment count, and camera paths. ISP configurations
            automatically request the HDR render variable in addition to the configured outputs.
        render_data: Camera's render scope and product identity.
        device_id: CUDA device index the render product is pinned to, so its render var buffers are
            allocated on the same device as the Warp kernels that read them.
        enable_shadows: Whether lights cast shadows. Defaults to False. Only honored for the
            ``simple_shading_*`` data types, which are the ones that select RTX Minimal mode.

    Returns:
        Render product USD layer, including the USDA header and default prim metadata.
    """
    data_types = list(spec.cfg.data_types)
    if spec.cfg.isp_cfg is not None and "rgb_hdr" not in data_types:
        data_types.append("rgb_hdr")
    tiled_width, tiled_height = render_data.num_cols * render_data.width, render_data.num_rows * render_data.height
    render_var_configs = get_render_var_configs(data_types, render_data.render_scope_name)
    minimal_mode = next(
        (_RTX_MINIMAL_MODES[data_type] for data_type in data_types if data_type in _RTX_MINIMAL_MODES), None
    )

    background_color = spec.cfg.background_color
    if background_color is None:
        bg_type_line = 'token omni:rtx:background:source:type = "domeLight"'
    else:
        r, g, b = background_color
        bg_type_line = (
            f'token omni:rtx:background:source:type = "color"\n'
            f"        color3f omni:rtx:background:source:color = ({r}, {g}, {b})"
        )

    # Minimal is the only OVRTX render mode with a shadow switch, so ``enable_shadows`` is authored
    # only there. The path-traced modes always trace shadows: ``omni:rtx:shadows:enabled`` exists as
    # a setting name and authors without error, but no path-tracing backend reads it.
    if minimal_mode is None:
        render_mode_lines = ['token omni:rtx:rendermode = "RealTimePathTracing"']
    else:
        render_mode_lines = [
            'token omni:rtx:rendermode = "Minimal"',
            f"int omni:rtx:minimal:mode = {minimal_mode}",
            f"bool omni:rtx:minimal:castShadows = {'true' if enable_shadows else 'false'}",
        ]

    render_mode_block = "\n        ".join(render_mode_lines)
    ordered_vars = ", ".join(f"<{path}>" for path, _, _ in render_var_configs)
    render_var_defs = "\n".join(
        f'''        def RenderVar "{name}"
        {{
            uniform string sourceName = "{source}"
        }}'''
        for _, name, source in render_var_configs
    )

    layer = f'''#usda 1.0
(defaultPrim = "{render_data.render_scope_name}")

def Scope "{render_data.render_scope_name}"
{{
    def RenderProduct "RenderProduct" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    ) {{
        rel camera = [<{render_data.camera_paths[0]}>]
        uint[] deviceIds = [{device_id}]
        {bg_type_line}
        float omni:rtx:rt:ambientLight:intensity = 1.0
        {render_mode_block}
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        rel orderedVars = [{ordered_vars}]
        uniform int2 resolution = ({tiled_width}, {tiled_height})
    }}

    def "Vars"
    {{
{render_var_defs}
    }}
}}
'''
    if spec.camera_prim_paths and not spec.render_settings:
        return layer
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().ImportFromString(layer)
    if not spec.camera_prim_paths:
        camera = UsdGeom.Camera.Define(stage, render_data.camera_paths[0])
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(20.955 * spec.cfg.height / spec.cfg.width)
        camera.CreateFocalLengthAttr(24.0)
        camera.CreateHorizontalApertureOffsetAttr(0.0)
        camera.CreateVerticalApertureOffsetAttr(0.0)
        camera.CreateClippingRangeAttr((0.01, 1.0e6))
        camera.AddTransformOp().Set(Gf.Matrix4d(1.0))
    product = stage.GetPrimAtPath(render_data.render_product_path)
    for name, (type_name, value) in spec.render_settings.items():
        product.CreateAttribute(name, getattr(Sdf.ValueTypeNames, type_name)).Set(value)
    return stage.GetRootLayer().ExportToString()
