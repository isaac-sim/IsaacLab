# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD manipulation for OVRTX: Render scope building, camera injection, and stage prim activation."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from pxr import Sdf, Usd, UsdGeom

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

_DEFAULT_RENDER_VAR = _RENDER_VAR_BY_DATA_TYPE["rgb"]

# Segmentation ID-map vars are authored alongside the pixel AOVs, not as camera data types.
_SEGMENTATION_MAP_RENDER_VARS: tuple[tuple[str, str], ...] = (
    ("StableIdSemanticIdMap", "StableIdSemanticIdMap"),
    ("StableIdMap", "StableIdMap"),
    ("SemanticIdMap", "SemanticIdMap"),
)

_RENDER_VAR_PRIM_NAME_BY_SOURCE: Mapping[str, str] = MappingProxyType(
    {source: name for name, source in (*_RENDER_VAR_BY_DATA_TYPE.values(), *_SEGMENTATION_MAP_RENDER_VARS)}
)


def _validate_data_type_combination(data_types: list[str]) -> None:
    """Reject data type combinations that a single OVRTX render product cannot serve.

    Args:
        data_types: Requested camera data types.

    Raises:
        ValueError: If color and simple-shading data types are combined, or if more than one
            simple-shading data type is requested.
    """
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


def get_render_var_config(data_types: list[str], render_scope_name: str) -> tuple[str, str, str]:
    """Return the first supported render-var configuration for ``data_types``.

    Args:
        data_types: Requested camera data types.
        render_scope_name: Root scope containing the render product and its render vars.

    Returns:
        The render-var config, defaulting to ``LdrColor`` when no entry is supported.
    """
    return get_render_var_configs(data_types, render_scope_name)[0]


def get_render_var_configs(data_types: list[str], render_scope_name: str) -> list[tuple[str, str, str]]:
    """Return render-var configs for the requested camera data types.

    Shared sources are de-duplicated. Unsupported data types are logged and skipped; if no
    supported type remains, ``LdrColor`` is used. Segmentation requests also add their ID-map vars.

    Args:
        data_types: Requested camera data types.
        render_scope_name: Root scope containing the render product and its render vars.

    Returns:
        Render-var configs as (absolute prim path, prim name, source name) tuples.

    Raises:
        ValueError: If ``data_types`` contains incompatible outputs.
    """
    data_types = data_types if data_types else ["rgb"]
    _validate_data_type_combination(data_types)

    render_vars: list[tuple[str, str]] = []
    unsupported: list[str] = []
    for data_type in data_types:
        config = _RENDER_VAR_BY_DATA_TYPE.get(data_type)
        if config is None:
            unsupported.append(data_type)
        elif config not in render_vars:
            render_vars.append(config)

    if unsupported:
        logger.warning(
            "OVRTX does not support the requested data type(s) %s; no render var is authored for them.", unsupported
        )
    if not render_vars:
        render_vars.append(_DEFAULT_RENDER_VAR)

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


def build_render_scope_usd(
    spec: CameraRenderSpec,
    render_data: OVRTXCameraRenderData,
    *,
    device_id: int | None = None,
    enable_shadows: bool = False,
) -> str:
    """Build the camera's USD scope containing its RenderProduct and Vars.

    The initial camera relationship targets only environment zero, whose camera exists in the
    trimmed stage. Multi-environment rendering rewrites it after runtime cloning.

    Args:
        spec: Camera configuration, environment count, and camera paths. ISP configurations
            automatically request the HDR render variable in addition to the configured outputs.
        render_data: Camera's render scope and product identity.
        device_id: CUDA device index the render product is pinned to via ``deviceIds``. When ``None``,
            OVRTX assigns the device automatically.
        enable_shadows: Whether lights cast shadows. Defaults to False. Only honored in RTX Minimal
            mode, selected by ``simple_shading_*`` data types; the path-traced modes always cast shadows.

    Returns:
        The USD snippet for the render scope, without a layer header or metadata.
    """
    data_types = list(spec.cfg.data_types or ["rgb"])
    if spec.cfg.isp_cfg is not None and "rgb_hdr" not in data_types:
        data_types.append("rgb_hdr")
    tiled_width, tiled_height = _tiled_resolution(spec.num_instances, spec.cfg.width, spec.cfg.height)
    camera_path = f"/World/envs/env_0/{spec.camera_path_relative_to_env_0}"
    render_var_configs = get_render_var_configs(data_types, render_data.render_scope_name)
    minimal_mode = next(
        (_RTX_MINIMAL_MODES[data_type] for data_type in data_types if data_type in _RTX_MINIMAL_MODES), None
    )

    # OVRTX reads ``deviceIds`` as CUDA indices and returns render var buffers on that device. Left
    # unauthored it picks its own device, which on a multi-GPU machine can differ from the device the
    # consuming Warp kernels run on -- an illegal access without peer access, silent garbage with it.
    device_ids_line = "" if device_id is None else f"\n        uint[] deviceIds = [{device_id}]"

    background_color = getattr(spec.cfg, "background_color", None)
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

    return f'''
def Scope "{render_data.render_scope_name}"
{{
    def RenderProduct "{render_data.render_product_name}" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    ) {{
        rel camera = [<{camera_path}>]{device_ids_line}
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


def _tiled_resolution(num_envs: int, width: int, height: int) -> tuple[int, int]:
    """Compute tiled width and height from env count and per-env resolution (same as Camera)."""
    num_cols = math.ceil(math.sqrt(num_envs))
    num_rows = math.ceil(num_envs / num_cols)
    return num_cols * width, num_rows * height


def build_render_product_as_string(
    spec: CameraRenderSpec,
    render_data: OVRTXCameraRenderData,
    *,
    device_id: int | None = None,
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
            allocated on the same device as the Warp kernels that read them. When ``None``, OVRTX
            assigns the device automatically.
        enable_shadows: Whether lights cast shadows. Defaults to False. Only honored for the
            ``simple_shading_*`` data types, which are the ones that select RTX Minimal mode.

    Returns:
        Render product USD layer, including the USDA header and default prim metadata.
    """
    camera_content = build_render_scope_usd(spec, render_data, device_id=device_id, enable_shadows=enable_shadows)
    return f'#usda 1.0\n(defaultPrim = "{render_data.render_scope_name}")\n' + camera_content


def create_scene_partition_attributes(
    stage,
    num_envs: int = 1,
) -> None:
    """Create scene partition attributes for env roots and cameras.

    Camera prims are discovered by USD type (``UsdGeom.Camera``) rather than by name, so this works regardless of
    where the camera is placed in the hierarchy.

    Args:
        stage: USD stage to modify.
        num_envs: Number of environments.
    """
    # Collect the attribute paths and scene partition tokens to update.
    attr_updates: list[tuple[Sdf.Path, str]] = []
    for env_idx in range(num_envs):
        env_path = f"/World/envs/env_{env_idx}"
        env_prim = stage.GetPrimAtPath(env_path)
        if not env_prim.IsValid():
            logger.warning("Failed to get env root prim at '%s'", env_path)
            continue

        scene_partition = f"env_{env_idx}"

        for prim in Usd.PrimRange(env_prim):
            if prim.GetPath() == env_prim.GetPath():
                attr_path = prim.GetPath().AppendProperty("primvars:omni:scenePartition")
            elif prim.IsA(UsdGeom.Camera):
                attr_path = prim.GetPath().AppendProperty("omni:scenePartition")
            else:
                continue
            attr_updates.append((attr_path, scene_partition))

    root_layer = stage.GetRootLayer()
    type_name = Sdf.ValueTypeNames.Token
    variability = Sdf.VariabilityUniform
    is_custom = True

    # Create the attributes and set the default values.
    with Sdf.ChangeBlock():
        for attr_path, scene_partition in attr_updates:
            Sdf.JustCreatePrimAttributeInLayer(root_layer, attr_path, type_name, variability, is_custom)
            root_layer.GetAttributeAtPath(attr_path).default = scene_partition
            logger.debug("Set scene partition '%s' on '%s'", scene_partition, attr_path.GetPrimPath())


def _collect_prims_to_deactivate(parent_prim: Usd.Prim, source_paths: frozenset[Sdf.Path]) -> list[Sdf.Path]:
    """Collect child prims under ``parent_prim`` for deactivation.

    For each child:

    * If the child is a source, keep the full subtree and stop descending.
    * If the child is an ancestor of some source, recurse to deactivate non-source siblings deeper in the tree.
    * Otherwise, deactivate the child prim (including descendants).

    Args:
        parent_prim: Parent prim whose children are considered.
        source_paths: The paths to the cloning sources.

    Returns:
        Paths of prims to deactivate on the root layer.
    """
    prim_paths: list[Sdf.Path] = []

    for child in parent_prim.GetChildren():
        child_path = child.GetPath()

        # If the child is a source, keep it and stop walking down the tree.
        if child_path in source_paths:
            continue

        # If the child is an ancestor of some source, recurse to deactivate non-source siblings deeper in the tree.
        if any(source.HasPrefix(child_path) for source in source_paths):
            prim_paths.extend(_collect_prims_to_deactivate(child, source_paths))
            continue

        # Otherwise, deactivate the child prim (including descendants).
        if child.IsActive():
            prim_paths.append(child_path)

    return prim_paths


def export_stage_to_string(
    stage: Usd.Stage, num_envs: int, source_paths: tuple[str, ...], keep_env_roots: bool = True
) -> str:
    """Export the USD stage as a USDA string for OVRTX loading.

    When ``num_envs`` is 1, the full stage is exported unchanged. Otherwise the stage is trimmed so OVRTX receives
    only the prototype geometry it replicates at clone time. Non-source env descendants are deactivated on an
    anonymous session layer used only for export, so the input stage remains unchanged.

    When ``keep_env_roots`` is True (the legacy ``renderer.clone_usd`` path) the non-source env root prims stay
    active so the exported stage retains a slot for every env. The ovstage ``stage.clone`` path passes False, which
    additionally trims the non-source env roots themselves; ``stage.clone`` recreates them and the RenderProduct's
    camera relationship is re-authored after clone.

    Args:
        stage: USD stage to export.
        num_envs: Number of parallel environments on the stage.
        source_paths: The paths to source prims to keep in the exported stage.
        keep_env_roots: Whether to keep the non-source env root prims active in the exported stage. Pass False for
            the ovstage clone path, which repopulates env roots itself.

    Returns:
        USDA text of the (possibly trimmed) stage.
    """
    if num_envs <= 1:
        return stage.ExportToString()

    export_session = Sdf.Layer.CreateAnonymous()
    export_session.subLayerPaths = [stage.GetSessionLayer().identifier]
    export_stage = Usd.Stage.Open(stage.GetRootLayer(), export_session)
    envs_path = Sdf.Path("/World/envs")
    envs_prim = export_stage.GetPrimAtPath(envs_path)
    if not envs_prim.IsValid():
        raise RuntimeError(f"Failed to get prim at path: {envs_path}")

    source_path_set = frozenset(map(Sdf.Path, source_paths))
    prim_paths: list[Sdf.Path] = []

    if keep_env_roots:
        for child in envs_prim.GetChildren():
            # Legacy code path: keep env roots so we can query their xforms after opening stage
            child_path = child.GetPath()
            if child_path not in source_path_set:
                prim_paths.extend(_collect_prims_to_deactivate(child, source_path_set))
    else:
        # Ovstage code path: strip env roots, their xforms are queried beforehand.
        prim_paths = _collect_prims_to_deactivate(envs_prim, source_path_set)

    with Sdf.ChangeBlock():
        for prim_path in prim_paths:
            Sdf.CreatePrimInLayer(export_session, prim_path).active = False
            logger.debug("Deactivated prim: %s", prim_path)
    logger.info("Deactivated %d prims in total", len(prim_paths))
    return export_stage.ExportToString()
