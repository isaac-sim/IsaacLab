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


def get_render_var_config(data_types: list[str]) -> tuple[str, str, str]:
    """Return (render_var_path, render_var_name, source_name) from data_types."""
    use_depth = any(dt in ["depth", "distance_to_image_plane", "distance_to_camera"] for dt in data_types)
    use_albedo = "albedo" in data_types
    use_semantic = "semantic_segmentation" in data_types
    use_rgb = any(dt in ["rgb", "rgba"] for dt in data_types)
    use_hdr = "rgb_hdr" in data_types

    if use_depth and not (use_rgb or use_albedo or use_semantic):
        return "/Render/Vars/depth", "depth", "DistanceToImagePlaneSD"
    if use_albedo and not (use_rgb or use_semantic):
        return "/Render/Vars/albedo", "albedo", "DiffuseAlbedoSD"
    if use_semantic and not (use_rgb or use_albedo):
        return "/Render/Vars/semantic", "semantic", "SemanticSegmentation"
    if use_hdr and not use_rgb:
        return "/Render/Vars/HdrColor", "HdrColor", "HdrColor"
    return "/Render/Vars/LdrColor", "LdrColor", "LdrColor"


def get_render_var_configs(data_types: list[str]) -> list[tuple[str, str, str]]:
    """Return render var configs needed for the requested data types.

    Returns the single render var resolved by :func:`get_render_var_config`,
    plus ``HdrColor`` when both ``"rgb"`` (or ``"rgba"``) and ``"rgb_hdr"`` are
    in ``data_types`` so PPISP can consume the HDR AOV alongside the LDR
    destination on the same render product. Other multi-AOV combinations are
    not supported.
    """
    data_types = data_types if data_types else ["rgb"]
    render_vars: list[tuple[str, str, str]] = [get_render_var_config(data_types)]
    use_rgb = any(dt in ["rgb", "rgba"] for dt in data_types)
    if use_rgb and "rgb_hdr" in data_types:
        render_vars.append(("/Render/Vars/HdrColor", "HdrColor", "HdrColor"))
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

    Returns:
        The USD string for the render scope.
    """
    camera_rel_list = ", ".join([f"<{p}>" for p in camera_paths])

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
        token omni:rtx:background:source:type = "domeLight"
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
) -> tuple[str, str]:
    """Build the render product USD snippet as a string.

    This string is meant to be appended to an exported stage (ASCII) before loading into OVRTX.

    Args:
        width: Tile width from sensor config [px].
        height: Tile height from sensor config [px].
        num_envs: Number of environments from scene.
        data_types: Data types from sensor config.
        minimal_mode: RTX minimal mode. None if not requested. Valid values are 1, 2, 3.
        camera_rel_path: Camera prim path relative to the env root (e.g. ``"Camera"`` or ``"Robot/head_cam"``).

    Returns:
        Tuple of (render product USD snippet as a string, absolute render product prim path).
    """
    data_types = data_types if data_types else ["rgb"]
    tiled_width, tiled_height = _tiled_resolution(num_envs, width, height)

    camera_paths = [f"/World/envs/env_{i}/{camera_rel_path}" for i in range(num_envs)]
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


def export_stage_to_string(stage: Usd.Stage, num_envs: int, source_paths: tuple[str, ...]) -> str:
    """Export the USD stage as a USDA string for OVRTX loading.

    When ``num_envs`` is 1, the full stage is exported unchanged. Otherwise the stage is trimmed so OVRTX receives only
    the prototype geometry it will replicate with ``clone_usd``.

    Args:
        stage: USD stage to export.
        num_envs: Number of parallel environments on the stage.
        source_paths: The paths to source prims to keep in the exported stage.

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

    for child in envs_prim.GetChildren():
        # All env roots will be kept in the stage. If an env root is a source, keep the full subtree and don't walk down
        # the subtree, otherwise walk down the subtree to collect descendant prims that are not sources to deactivate.
        child_path = child.GetPath()
        if child_path not in source_path_set:
            prim_paths.extend(_collect_prims_to_deactivate(child, source_path_set))

    root_layer = stage.GetRootLayer()

    # Temporarily deactivate the prims so that the stage is exported without them.
    _set_prims_active_on_layer(root_layer, prim_paths, active=False)

    try:
        return stage.ExportToString()
    finally:
        # Restore the active state of the prims.
        _set_prims_active_on_layer(root_layer, prim_paths, active=True)
