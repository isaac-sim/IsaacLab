# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD manipulation for OVRTX: Render scope building, camera injection, and stage prim activation."""

from __future__ import annotations

import logging
import math

from pxr import Sdf, Usd, UsdGeom

logger = logging.getLogger(__name__)


# Render var authored for each supported camera data type, as
# ``(render_var_path, render_var_name, source_name)``. OVRTX keys the render vars of a frame by
# ``source_name``, so data types reading the same source (``rgb``/``rgba``, ``depth``/
# ``distance_to_image_plane``) collapse onto a single authored render var.
_RENDER_VAR_BY_DATA_TYPE: dict[str, tuple[str, str, str]] = {
    "rgb": ("/Render/Vars/LdrColor", "LdrColor", "LdrColor"),
    "rgba": ("/Render/Vars/LdrColor", "LdrColor", "LdrColor"),
    # Simple shading is not a distinct source: it is ``LdrColor`` rendered while the render product
    # is in RTX Minimal mode, which is why it cannot share a product with ``rgb``/``rgba``.
    "simple_shading_constant_diffuse": ("/Render/Vars/LdrColor", "LdrColor", "LdrColor"),
    "simple_shading_diffuse_mdl": ("/Render/Vars/LdrColor", "LdrColor", "LdrColor"),
    "simple_shading_full_mdl": ("/Render/Vars/LdrColor", "LdrColor", "LdrColor"),
    "rgb_hdr": ("/Render/Vars/HdrColor", "HdrColor", "HdrColor"),
    "albedo": ("/Render/Vars/albedo", "albedo", "DiffuseAlbedoSD"),
    "depth": ("/Render/Vars/depth", "depth", "DistanceToImagePlaneSD"),
    "distance_to_image_plane": ("/Render/Vars/depth", "depth", "DistanceToImagePlaneSD"),
    # Distance to camera reads a different source than image-plane depth, so it gets its own prim
    # instead of reusing ``/Render/Vars/depth``; both can then be authored on one render product.
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
_SIMPLE_SHADING_DATA_TYPES = frozenset({
    "simple_shading_constant_diffuse",
    "simple_shading_diffuse_mdl",
    "simple_shading_full_mdl",
})

_COLOR_DATA_TYPES = frozenset({"rgb", "rgba"})

_DEFAULT_RENDER_VAR = _RENDER_VAR_BY_DATA_TYPE["rgb"]


def _validate_data_type_combination(data_types: list[str]) -> None:
    """Reject data type combinations that a single OVRTX render product cannot serve.

    Args:
        data_types: Requested camera data types.

    Raises:
        ValueError: If color and simple-shading data types are combined, or if more than one
            simple-shading data type is requested. Both cases silently retarget the shared
            ``LdrColor`` render var, leaving the other outputs empty or wrongly shaded.
    """
    simple_shading = [data_type for data_type in data_types if data_type in _SIMPLE_SHADING_DATA_TYPES]
    color = [data_type for data_type in data_types if data_type in _COLOR_DATA_TYPES]

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
    """Return the primary ``(render_var_path, render_var_name, source_name)`` for ``data_types``.

    The primary render var is the one authored for the first supported entry of ``data_types``. It
    seeds the single-render-var arguments of :func:`build_render_scope_usd`; use
    :func:`get_render_var_configs` to author every requested output.

    Args:
        data_types: Requested camera data types.

    Returns:
        The primary render var config. Defaults to ``LdrColor`` when no entry is supported.
    """
    return get_render_var_configs(data_types)[0]


def get_render_var_configs(data_types: list[str]) -> list[tuple[str, str, str]]:
    """Return the render var configs needed to serve every requested data type.

    Each config is a ``(render_var_path, render_var_name, source_name)`` tuple. One render var is
    authored per requested data type, in request order, de-duplicated by config so data types that
    read the same source (``rgb``/``rgba``, ``depth``/``distance_to_image_plane``) share one entry.
    Data types OVRTX does not support (e.g. ``instance_id_segmentation_fast``) are logged and
    skipped; when that leaves nothing, ``LdrColor`` is authored so the render product stays valid.

    The following ID-map render vars are appended when applicable. They carry the metadata that the
    segmentation AOVs are decoded against rather than pixels of their own:

    * ``SemanticIdMap`` — when ``"semantic_segmentation"`` is requested, so the
      semantic-ID-to-label mapping can be decoded for ``camera.data.info``.
    * ``StableIdSemanticIdMap``, ``StableIdMap``, ``SemanticIdMap`` — when
      ``"instance_segmentation"`` is requested, so the instance-ID-to-prim-path
      (``idToLabels``) and instance-ID-to-semantic (``idToSemantics``) mappings can be decoded.

    Args:
        data_types: Requested camera data types.

    Returns:
        The render var configs to author on the render product.

    Raises:
        ValueError: If ``data_types`` combines outputs one render product cannot serve. See
            :func:`_validate_data_type_combination`.
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
        render_vars.append(("/Render/Vars/StableIdSemanticIdMap", "StableIdSemanticIdMap", "StableIdSemanticIdMap"))
        render_vars.append(("/Render/Vars/StableIdMap", "StableIdMap", "StableIdMap"))
    # SemanticIdMap resolves the semantic-ID-to-label mapping and is shared by both semantic_segmentation and
    # instance_segmentation, so it is authored once when either output is requested.
    if "semantic_segmentation" in data_types or "instance_segmentation" in data_types:
        render_vars.append(("/Render/Vars/SemanticIdMap", "SemanticIdMap", "SemanticIdMap"))
    return render_vars


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

    Returns:
        The USD string for the render scope.
    """
    camera_rel_list = ", ".join([f"<{p}>" for p in camera_paths])

    if background_color is None:
        bg_type_line = 'token omni:rtx:background:source:type = "domeLight"'
    else:
        r, g, b = background_color
        bg_type_line = (
            f'token omni:rtx:background:source:type = "color"\n'
            f"        color3f omni:rtx:background:source:color = ({r}, {g}, {b})"
        )

    if minimal_mode is None:
        render_mode_lines = ['token omni:rtx:rendermode = "RealTimePathTracing"']
    else:
        render_mode_lines = [
            'token omni:rtx:rendermode = "Minimal"',
            f"int omni:rtx:minimal:mode = {minimal_mode}",
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
        rel camera = [{camera_rel_list}]
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
    minimal_mode: int | None = None,
    camera_rel_path: str = "Camera",
    background_color: tuple[float, float, float] | None = None,
) -> tuple[str, str]:
    """Build the render product USD snippet as a string.

    This string is meant to be appended to an exported stage (ASCII) before loading into OVRTX.
    The initial camera relationship targets only environment zero, whose camera is guaranteed to
    exist in the trimmed stage. Multi-environment rendering rewrites the relationship with every
    resolved camera path after runtime cloning.

    Args:
        width: Tile width from sensor config [px].
        height: Tile height from sensor config [px].
        num_envs: Number of environments from scene.
        data_types: Data types from sensor config.
        minimal_mode: RTX minimal mode. None if not requested. Valid values are 1, 2, 3.
        camera_rel_path: Camera prim path relative to the env root (e.g. ``"Camera"`` or ``"Robot/head_cam"``).
        background_color: Solid background color as normalized RGB floats ``(r, g, b)`` in ``[0, 1]``.
            When set, the render product uses a solid color background instead of the dome light.
            When ``None``, the default dome-light background is used.

    Returns:
        Tuple of (render product USD snippet as a string, absolute render product prim path).
    """
    data_types = data_types if data_types else ["rgb"]
    tiled_width, tiled_height = _tiled_resolution(num_envs, width, height)

    camera_paths = [f"/World/envs/env_0/{camera_rel_path}"]
    render_product_name = "RenderProduct"
    render_product_path = f"/Render/{render_product_name}"

    render_var_configs = get_render_var_configs(data_types)
    render_var_path, render_var_name, source_name = render_var_configs[0]

    camera_content = build_render_scope_usd(
        camera_paths,
        render_product_name,
        render_var_path,
        render_var_name,
        source_name,
        tiled_width,
        tiled_height,
        minimal_mode,
        render_var_configs,
        background_color,
    )
    return camera_content, render_product_path


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


def _set_prims_active_on_layer(layer: Sdf.Layer, prim_paths: list[Sdf.Path], active: bool) -> None:
    """Activate or deactivate prims on the given layer.

    Args:
        layer: Layer to modify the prims on.
        prim_paths: Paths of prims to activate or deactivate.
        active: Whether to activate or deactivate the prims.
    """
    action_str = "Activated" if active else "Deactivated"

    with Sdf.ChangeBlock():
        for prim_path in prim_paths:
            # If a prim already exists at the given path it will be returned unmodified.
            prim_spec = Sdf.CreatePrimInLayer(layer, prim_path)
            prim_spec.active = active
            logger.debug("%s prim: %s", action_str, prim_path)

    logger.info("%s %d prims in total", action_str, len(prim_paths))


def export_stage_to_string(
    stage: Usd.Stage, num_envs: int, source_paths: tuple[str, ...], keep_env_roots: bool = True
) -> str:
    """Export the USD stage as a USDA string for OVRTX loading.

    When ``num_envs`` is 1, the full stage is exported unchanged. Otherwise the stage is trimmed so OVRTX receives
    only the prototype geometry it replicates at clone time. Non-source env descendants are temporarily deactivated
    on the root layer during export and restored afterwards; ``stage.ExportToString`` re-composes the stage, so
    deactivated prims drop out of the exported text and their paths are absent when the clone path repopulates them.

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

    envs_path = Sdf.Path("/World/envs")
    envs_prim = stage.GetPrimAtPath(envs_path)
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

    root_layer = stage.GetRootLayer()

    # Temporarily deactivate the prims so that the stage is exported without them.
    _set_prims_active_on_layer(root_layer, prim_paths, active=False)

    try:
        return stage.ExportToString()
    finally:
        # Restore the active state of the prims.
        _set_prims_active_on_layer(root_layer, prim_paths, active=True)
