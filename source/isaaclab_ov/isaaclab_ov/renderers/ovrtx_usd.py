# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD manipulation for OVRTX: Render scope building, camera injection, and stage prim activation."""

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

    if use_depth and not (use_rgb or use_albedo or use_semantic):
        return "/Render/Vars/depth", "depth", "DistanceToImagePlaneSD"
    if use_albedo and not (use_rgb or use_semantic):
        return "/Render/Vars/albedo", "albedo", "DiffuseAlbedoSD"
    if use_semantic and not (use_rgb or use_albedo):
        return "/Render/Vars/semantic", "semantic", "SemanticSegmentation"
    return "/Render/Vars/LdrColor", "LdrColor", "LdrColor"


def build_render_scope_usd(
    camera_paths: list[str],
    render_product_name: str,
    render_var_path: str,
    render_var_name: str,
    source_name: str,
    tiled_width: int,
    tiled_height: int,
    minimal_mode: int | None = None,
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
        rel orderedVars = <{render_var_path}>
        uniform int2 resolution = ({tiled_width}, {tiled_height})
    }}

    def "Vars"
    {{
        def RenderVar "{render_var_name}"
        {{
            uniform string sourceName = "{source_name}"
        }}
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

    render_var_path, render_var_name, source_name = get_render_var_config(data_types)

    camera_content = build_render_scope_usd(
        camera_paths,
        render_product_name,
        render_var_path,
        render_var_name,
        source_name,
        tiled_width,
        tiled_height,
        minimal_mode,
    )
    return camera_content, render_product_path


def create_scene_partition_attributes(
    stage,
    num_envs: int = 1,
    use_ovrtx_cloning: bool = True,
    enable_scene_partition_workaround: bool = False,
) -> None:
    """Create scene partition attributes for env roots and cameras.

    If use_ovrtx_cloning is True, only env_0 is exported for OVRTX; env_1..env_{n-1} are deactivated before export.
    OVRTX clones env_0 internally and _update_scene_partitions_after_clone sets partition attributes on the clones.
    So we only need to set attributes on env_0 here.

    Camera prims are discovered by USD type (``UsdGeom.Camera``) rather than by name, so this works regardless of
    where the camera is placed in the hierarchy.

    Args:
        stage: USD stage to modify.
        num_envs: Number of environments.
        use_ovrtx_cloning: Whether OVRTX cloning is enabled.
        enable_scene_partition_workaround: Whether to enable the scene partition workaround for OVRTX 0.2.0 because it
            doesn't support primvar inheritance.
    """
    env_indices = [0] if use_ovrtx_cloning else range(num_envs)
    for env_idx in env_indices:
        env_path = f"/World/envs/env_{env_idx}"
        env_prim = stage.GetPrimAtPath(env_path)
        if not env_prim.IsValid():
            logger.warning("Failed to get env root prim at '%s'", env_path)
            continue

        scene_partition = f"env_{env_idx}"
        env_prim.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token).Set(scene_partition)
        logger.debug("Set scene partition '%s' on env root '%s'", scene_partition, env_prim.GetPath())

        for prim in Usd.PrimRange(env_prim):
            if prim.GetPath() == env_prim.GetPath():
                continue
            if prim.IsA(UsdGeom.Camera):
                prim.CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set(scene_partition)
                logger.debug("Set scene partition '%s' on camera '%s'", scene_partition, prim.GetPath())
            elif enable_scene_partition_workaround:
                prim.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token).Set(scene_partition)
                logger.debug("Set scene partition '%s' on prim '%s'", scene_partition, prim.GetPath())


def export_stage_to_string(stage, num_envs: int, use_ovrtx_cloning: bool = True) -> str:
    """Export the stage to a string; when num_envs > 1, only env_0 is exported for OVRTX cloning.

    When num_envs > 1, deactivates env_1..env_{num_envs-1} before export and reactivates
    them after, so the exported content contains only env_0. The stage is modified in place.

    Args:
        stage: USD stage to export.
        num_envs: Number of environments.
        use_ovrtx_cloning: Whether OVRTX cloning is enabled.

    Returns:
        The exported stage as a string.
    """
    deactivated_prims = []
    if use_ovrtx_cloning and num_envs > 1:
        logger.info("Deactivating %d environment roots...", num_envs - 1)
        for env_idx in range(1, num_envs):
            env_path = f"/World/envs/env_{env_idx}"
            prim = stage.GetPrimAtPath(env_path)
            if prim.IsValid() and prim.IsActive():
                prim.SetActive(False)
                deactivated_prims.append(prim)
                logger.debug("Deactivated environment root: %s", env_path)

        logger.info("Deactivated %d environment roots in total", len(deactivated_prims))

    try:
        return stage.ExportToString()
    finally:
        if deactivated_prims:
            logger.info("Reactivating %d environment roots...", len(deactivated_prims))
            for prim in deactivated_prims:
                if prim.IsValid():
                    prim.SetActive(True)
