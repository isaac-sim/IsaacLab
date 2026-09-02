# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD manipulation for OVRTX: Render scope building, camera injection, and stage prim activation."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from pxr import Sdf, Usd, UsdGeom

logger = logging.getLogger(__name__)


# Maps camera data types to (prim path, prim name, source name). Shared sources use one config.
# OVRTX 0.4 keys ``frame.render_vars`` by source name; 0.5+ keys them by the RenderVar prim path.
_RENDER_VAR_BY_DATA_TYPE: dict[str, tuple[str, str, str]] = {
    "rgb": ("/Render/Vars/LdrColor", "LdrColor", "LdrColor"),
    "rgba": ("/Render/Vars/LdrColor", "LdrColor", "LdrColor"),
    # Simple shading uses LdrColor in per-product RTX Minimal mode.
    "simple_shading_constant_diffuse": ("/Render/Vars/LdrColor", "LdrColor", "LdrColor"),
    "simple_shading_diffuse_mdl": ("/Render/Vars/LdrColor", "LdrColor", "LdrColor"),
    "simple_shading_full_mdl": ("/Render/Vars/LdrColor", "LdrColor", "LdrColor"),
    "rgb_hdr": ("/Render/Vars/HdrColor", "HdrColor", "HdrColor"),
    "albedo": ("/Render/Vars/albedo", "albedo", "DiffuseAlbedoSD"),
    "depth": ("/Render/Vars/depth", "depth", "DistanceToImagePlaneSD"),
    "distance_to_image_plane": ("/Render/Vars/depth", "depth", "DistanceToImagePlaneSD"),
    # This source requires a distinct render-var prim.
    "distance_to_camera": ("/Render/Vars/DistanceToCameraSD", "DistanceToCameraSD", "DistanceToCameraSD"),
    "normals": ("/Render/Vars/NormalSD", "NormalSD", "NormalSD"),
    "motion_vectors": ("/Render/Vars/TargetMotionSD", "TargetMotionSD", "TargetMotionSD"),
    "semantic_segmentation": ("/Render/Vars/semantic", "semantic", "SemanticSegmentation"),
    "instance_segmentation": (
        "/Render/Vars/NonStableInstanceSegmentation",
        "NonStableInstanceSegmentation",
        "NonStableInstanceSegmentation",
    ),
}

# Data types produced by putting the whole render product into RTX Minimal mode.
_SIMPLE_SHADING_DATA_TYPES = frozenset(
    {
        "simple_shading_constant_diffuse",
        "simple_shading_diffuse_mdl",
        "simple_shading_full_mdl",
    }
)

_COLOR_DATA_TYPES = frozenset({"rgb", "rgba"})

_DEFAULT_RENDER_VAR = _RENDER_VAR_BY_DATA_TYPE["rgb"]

# Segmentation ID-map vars are authored alongside the pixel AOVs, not as camera data types.
_SEGMENTATION_MAP_RENDER_VARS: tuple[tuple[str, str, str], ...] = (
    ("/Render/Vars/StableIdSemanticIdMap", "StableIdSemanticIdMap", "StableIdSemanticIdMap"),
    ("/Render/Vars/StableIdMap", "StableIdMap", "StableIdMap"),
    ("/Render/Vars/SemanticIdMap", "SemanticIdMap", "SemanticIdMap"),
)

_RENDER_VAR_PRIM_PATH_BY_SOURCE: Mapping[str, str] = MappingProxyType(
    {source: path for path, _, source in (*_RENDER_VAR_BY_DATA_TYPE.values(), *_SEGMENTATION_MAP_RENDER_VARS)}
)


def _validate_data_type_combination(data_types: list[str]) -> None:
    """Reject data type combinations that a single OVRTX render product cannot serve.

    Args:
        data_types: Requested camera data types.

    Raises:
        ValueError: If color and simple-shading data types are combined, or if more than one
            simple-shading data type is requested.
    """
    simple_shading = list(
        dict.fromkeys(data_type for data_type in data_types if data_type in _SIMPLE_SHADING_DATA_TYPES)
    )
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


def get_render_var_config(data_types: list[str]) -> tuple[str, str, str]:
    """Return the first supported render-var configuration for ``data_types``.

    Args:
        data_types: Requested camera data types.

    Returns:
        The render-var config, defaulting to ``LdrColor`` when no entry is supported.
    """
    return get_render_var_configs(data_types)[0]


def get_render_var_configs(data_types: list[str]) -> list[tuple[str, str, str]]:
    """Return render-var configs for the requested camera data types.

    Shared sources are de-duplicated. Unsupported data types are logged and skipped; if no
    supported type remains, ``LdrColor`` is used. Segmentation requests also add their ID-map vars.

    Args:
        data_types: Requested camera data types.

    Returns:
        Render-var configs to author on the render product.

    Raises:
        ValueError: If ``data_types`` contains incompatible outputs.
    """
    data_types = data_types if data_types else ["rgb"]
    _validate_data_type_combination(data_types)

    render_vars: list[tuple[str, str, str]] = []
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
    return render_vars


def render_var_prim_paths_by_source() -> Mapping[str, str]:
    """Return the authored RenderVar prim path of every OVRTX render-var source.

    Returns:
        Read-only mapping of render-var source name to the absolute path of the ``RenderVar``
        prim this module authors for it.
    """
    return _RENDER_VAR_PRIM_PATH_BY_SOURCE


def build_render_scope_usd(
    camera_paths: list[str],
    render_product_name: str,
    render_var_path: str,
    render_var_name: str,
    source_name: str,
    tiled_width: int,
    tiled_height: int,
    minimal_mode: int | None = None,
    render_var_configs: list[tuple[str, str, str]] | None = None,
    background_color: tuple[float, float, float] | None = None,
    device_id: int | None = None,
    enable_shadows: bool = False,
) -> str:
    """Build the Render scope USD string (def Scope Render, RenderProduct, Vars).

    Args:
        camera_paths: List of camera prim paths.
        render_product_name: Name of the render product.
        render_var_path: Path of the render variable.
        render_var_name: Name of the render variable.
        source_name: Name of the source.
        tiled_width: Width of the tiled image.
        tiled_height: Height of the tiled image.
        minimal_mode: RTX minimal mode. None if not requested. Valid values are 1, 2, 3.
        render_var_configs: Render variables to author. Uses the single render var arguments if not provided.
        background_color: Solid background color as normalized RGB floats ``(r, g, b)`` in ``[0, 1]``.
            When set, the render product uses a solid color background instead of the dome light.
            When ``None``, the default dome-light background is used.
        device_id: CUDA device index the render product is pinned to via ``deviceIds``. When ``None``,
            OVRTX assigns the device automatically.
        enable_shadows: Whether lights cast shadows. Defaults to False. Only honored in RTX Minimal
            mode, that is when ``minimal_mode`` is set; the path-traced modes always cast shadows.

    Returns:
        The USD string for the render scope.
    """
    camera_rel_list = ", ".join([f"<{p}>" for p in camera_paths])
    # OVRTX reads ``deviceIds`` as CUDA indices and returns render var buffers on that device. Left
    # unauthored it picks its own device, which on a multi-GPU machine can differ from the device the
    # consuming Warp kernels run on -- an illegal access without peer access, silent garbage with it.
    device_ids_line = "" if device_id is None else f"\n        uint[] deviceIds = [{device_id}]"

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
    if render_var_configs is None:
        render_var_configs = [(render_var_path, render_var_name, source_name)]
    ordered_vars = ", ".join(f"<{path}>" for path, _, _ in render_var_configs)
    render_var_defs = "\n".join(
        f'''        def RenderVar "{name}"
        {{
            uniform string sourceName = "{source}"
        }}'''
        for _, name, source in render_var_configs
    )

    return f'''
def Scope "Render"
{{
    def RenderProduct "{render_product_name}" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    ) {{
        rel camera = [{camera_rel_list}]{device_ids_line}
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
    width: int,
    height: int,
    num_envs: int,
    data_types: list[str],
    camera_path: str,
    minimal_mode: int | None = None,
    background_color: tuple[float, float, float] | None = None,
    device_id: int | None = None,
    enable_shadows: bool = False,
) -> tuple[str, str]:
    """Build the render product USD snippet as a string.

    This string is meant to be appended to an exported stage (ASCII) before loading into OVRTX.
    The initial camera relationship targets only the exact source camera that exists in the trimmed
    stage. Multi-environment rendering rewrites the relationship with every exact camera path after
    runtime cloning.

    Args:
        width: Tile width from sensor config [px].
        height: Tile height from sensor config [px].
        num_envs: Number of environments from scene.
        data_types: Data types from sensor config.
        camera_path: Exact source camera prim path.
        minimal_mode: RTX minimal mode. None if not requested. Valid values are 1, 2, 3.
        background_color: Solid background color as normalized RGB floats ``(r, g, b)`` in ``[0, 1]``.
            When set, the render product uses a solid color background instead of the dome light.
            When ``None``, the default dome-light background is used.
        device_id: CUDA device index the render product is pinned to, so its render var buffers are
            allocated on the same device as the Warp kernels that read them. When ``None``, OVRTX
            assigns the device automatically.
        enable_shadows: Whether lights cast shadows. Defaults to False. Only honored for the
            ``simple_shading_*`` data types, which are the ones that select RTX Minimal mode.

    Returns:
        Tuple of (render product USD snippet as a string, absolute render product prim path).
    """
    data_types = data_types if data_types else ["rgb"]
    tiled_width, tiled_height = _tiled_resolution(num_envs, width, height)

    render_product_name = "RenderProduct"
    render_product_path = f"/Render/{render_product_name}"

    render_var_configs = get_render_var_configs(data_types)
    render_var_path, render_var_name, source_name = render_var_configs[0]

    camera_content = build_render_scope_usd(
        [camera_path],
        render_product_name,
        render_var_path,
        render_var_name,
        source_name,
        tiled_width,
        tiled_height,
        minimal_mode,
        render_var_configs,
        background_color,
        device_id,
        enable_shadows,
    )
    return camera_content, render_product_path


def create_scene_partition_attributes(
    stage,
    camera_paths_by_env: Mapping[str, str],
) -> None:
    """Create scene partition attributes on exact environment and camera prims.

    Args:
        stage: USD stage to modify.
        camera_paths_by_env: Exact environment root paths to their exact camera paths.

    Raises:
        ValueError: If a camera is outside its environment.
        RuntimeError: If a declared environment or camera prim does not exist.
    """
    # Collect the attribute paths and scene partition tokens to update.
    attr_updates: list[tuple[Sdf.Path, str]] = []
    for env_path, camera_path in camera_paths_by_env.items():
        if camera_path != env_path and not camera_path.startswith(env_path.rstrip("/") + "/"):
            raise ValueError(f"Camera '{camera_path}' is not under environment '{env_path}'.")
        env_prim = stage.GetPrimAtPath(env_path)
        if not env_prim.IsValid():
            raise RuntimeError(f"Failed to get environment root prim at '{env_path}'.")
        scene_partition = env_path.rstrip("/").rsplit("/", 1)[-1]
        attr_updates.append((env_prim.GetPath().AppendProperty("primvars:omni:scenePartition"), scene_partition))

        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"Failed to get camera prim at '{camera_path}'.")
        attr_updates.append((camera_prim.GetPath().AppendProperty("omni:scenePartition"), scene_partition))

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
    stage: Usd.Stage, env_paths: Sequence[str], source_paths: tuple[str, ...], keep_env_roots: bool = True
) -> str:
    """Export the USD stage as a USDA string for OVRTX loading.

    The stage is trimmed so OVRTX receives only the environment roots and prototype geometry declared by the
    plan. Non-source prims are deactivated on an anonymous session layer used only for export, so the input stage
    remains unchanged.

    When ``keep_env_roots`` is True (the legacy ``renderer.clone_usd`` path) the non-source env root prims stay
    active so the exported stage retains a slot for every env. The ovstage ``stage.clone`` path passes False, which
    additionally trims the non-source env roots themselves; ``stage.clone`` recreates them and the RenderProduct's
    camera relationship is re-authored after clone.

    Args:
        stage: USD stage to export.
        env_paths: Exact parallel environment root paths on the stage.
        source_paths: The paths to source prims to keep in the exported stage.
        keep_env_roots: Whether to keep the non-source env root prims active in the exported stage. Pass False for
            the ovstage clone path, which repopulates env roots itself.

    Returns:
        USDA text of the (possibly trimmed) stage.
    """
    export_session = Sdf.Layer.CreateAnonymous()
    export_session.subLayerPaths = [stage.GetSessionLayer().identifier]
    export_stage = Usd.Stage.Open(stage.GetRootLayer(), export_session)
    env_parent_path = Sdf.Path(env_paths[0]).GetParentPath()
    if any(Sdf.Path(path).GetParentPath() != env_parent_path for path in env_paths[1:]):
        raise ValueError("OVRTX requires every environment root to share one parent.")
    env_parent = export_stage.GetPrimAtPath(env_parent_path)
    if not env_parent.IsValid():
        raise RuntimeError(f"Failed to get prim at path: {env_parent_path}")

    source_path_set = frozenset(map(Sdf.Path, source_paths))
    prim_paths: list[Sdf.Path] = []

    if keep_env_roots:
        env_path_set = frozenset(map(Sdf.Path, env_paths))
        for env_path in env_path_set:
            if not export_stage.GetPrimAtPath(env_path).IsValid():
                raise RuntimeError(f"Failed to get environment root prim at '{env_path}'.")
        for env_prim in env_parent.GetChildren():
            env_path = env_prim.GetPath()
            if env_path in source_path_set:
                continue
            if env_path in env_path_set or any(source.HasPrefix(env_path) for source in source_path_set):
                prim_paths.extend(_collect_prims_to_deactivate(env_prim, source_path_set))
            elif env_prim.IsActive():
                prim_paths.append(env_path)
    else:
        # Ovstage code path: strip env roots, their xforms are queried beforehand.
        prim_paths = _collect_prims_to_deactivate(env_parent, source_path_set)

    with Sdf.ChangeBlock():
        for prim_path in prim_paths:
            Sdf.CreatePrimInLayer(export_session, prim_path).active = False
            logger.debug("Deactivated prim: %s", prim_path)
    logger.info("Deactivated %d prims in total", len(prim_paths))
    return export_stage.ExportToString()
