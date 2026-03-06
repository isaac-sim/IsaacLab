# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD manipulation for OVRTX: Render scope building, camera injection, and stage prim activation."""

import logging
import math
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from pxr import Sdf, Usd

if TYPE_CHECKING:
    from .ovrtx_renderer_cfg import OVRTXRendererCfg

logger = logging.getLogger(__name__)


def get_render_var_config(data_types: list[str], simple_shading_mode: bool) -> tuple[str, str, str]:
    """Return (render_var_path, render_var_name, source_name) from data_types and shading mode."""
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
    if simple_shading_mode:
        return "/Render/Vars/SimpleShading", "SimpleShading", "SimpleShadingSD"
    return "/Render/Vars/LdrColor", "LdrColor", "LdrColor"


def build_render_scope_usd(
    camera_paths: list[str],
    render_product_name: str,
    render_var_path: str,
    render_var_name: str,
    source_name: str,
    tiled_width: int,
    tiled_height: int,
    simple_shading_mode: bool = False,
) -> str:
    """Build the Render scope USD string (def Scope Render, RenderProduct, Vars)."""
    render_mode = "Minimal" if simple_shading_mode else "RealTimePathTracing"
    logger.info("Rendering mode: %s (omni:rtx:rendermode=%s)", render_var_name, render_mode)
    if simple_shading_mode:
        logger.info("Simple shading mode: ENABLED")
    else:
        logger.info("Simple shading mode: DISABLED (using full RTX path tracing)")
    camera_rel_list = ", ".join([f"<{p}>" for p in camera_paths])
    return f'''
def Scope "Render"
{{
    def RenderProduct "{render_product_name}" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    ) {{
        rel camera = [{camera_rel_list}]
        token omni:rtx:background:source:type = "domeLight"
        token omni:rtx:rendermode = "{render_mode}"
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
    """Compute tiled width and height from env count and per-env resolution (same as TiledCamera)."""
    num_cols = math.ceil(math.sqrt(num_envs))
    num_rows = math.ceil(num_envs / num_cols)
    return num_cols * width, num_rows * height


def inject_cameras_into_usd(
    usd_scene_path: str,
    cfg: "OVRTXRendererCfg",
    width: int,
    height: int,
    num_envs: int,
    data_types: list[str],
) -> tuple[str, str]:
    """Inject camera and render product definitions into an existing USD file.

    Reads the USD file, appends a Render scope (cameras + RenderProduct + Vars),
    writes to a temp file in cfg.temp_usd_dir, and returns (path_to_combined_usd, render_product_path).

    Args:
        usd_scene_path: Path to the base USD scene.
        cfg: OVRTX renderer config (simple_shading_mode, temp_usd_dir, temp_usd_suffix).
        width: Tile width from sensor config.
        height: Tile height from sensor config.
        num_envs: Number of environments from scene.
        data_types: Data types from sensor config.
    """
    with open(usd_scene_path) as f:
        original_usd = f.read()

    data_types = data_types if data_types else ["rgb"]
    tiled_width, tiled_height = _tiled_resolution(num_envs, width, height)

    camera_paths = [f"/World/envs/env_{i}/Camera" for i in range(num_envs)]
    render_product_name = "RenderProduct"
    render_product_path = f"/Render/{render_product_name}"

    render_var_path, render_var_name, source_name = get_render_var_config(data_types, cfg.simple_shading_mode)

    camera_content = build_render_scope_usd(
        camera_paths,
        render_product_name,
        render_var_path,
        render_var_name,
        source_name,
        tiled_width,
        tiled_height,
        simple_shading_mode=cfg.simple_shading_mode,
    )
    combined_usd = original_usd.rstrip() + "\n\n" + camera_content

    Path(cfg.temp_usd_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=cfg.temp_usd_suffix, delete=False, dir=cfg.temp_usd_dir) as f:
        f.write(combined_usd)
        temp_path = f.name
    logger.info("Created combined USD: %s", temp_path)
    return temp_path, render_product_path


def create_cloning_attributes(
    stage, camera_prim_name: str = "Camera", num_envs: int = 1, use_cloning: bool = True
) -> int:
    """Create OVRTX cloning attributes (scene partition, xform) on env_0 only.

    Only env_0 is exported for OVRTX; env_1..env_{n-1} are deactivated before export.
    OVRTX clones env_0 internally and _update_scene_partitions_after_clone sets
    partition attributes on the clones. So we only need to set attributes on env_0 here.

    Args:
        stage: USD stage to modify.
        camera_prim_name: Name of the camera prim under each env (e.g. "Camera").

    Returns:
        Total number of objects (non-camera prims) that received partition attributes.
    """
    total_objects = 0
    env_indices = [0] if use_cloning else range(num_envs)
    for env_idx in env_indices:
        env_path = f"/World/envs/env_{env_idx}"
        env_prim = stage.GetPrimAtPath(env_path)
        if not env_prim.IsValid():
            continue
        partition_name = f"env_{env_idx}"
        attr = env_prim.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token)
        attr.Set(partition_name)
        for prim in Usd.PrimRange(env_prim):
            if prim.GetPath() == env_prim.GetPath() or "Camera" in prim.GetPath().pathString:
                continue
            obj_attr = prim.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token)
            obj_attr.Set(partition_name)
            total_objects += 1
        camera_path = f"{env_path}/{camera_prim_name}"
        camera_prim = stage.GetPrimAtPath(camera_path)
        if camera_prim.IsValid():
            camera_prim.CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set(partition_name)
    return total_objects


def export_stage_for_ovrtx(stage, export_path: str, num_envs: int, use_cloning: bool = True) -> str:
    """Export the stage to a USD file; when num_envs > 1, only env_0 is exported for OVRTX cloning.

    When num_envs > 1, deactivates env_1..env_{num_envs-1} before export and reactivates
    them after, so the file contains only env_0. The stage is modified in place.

    Args:
        stage: USD stage to export.
        export_path: Path for the exported file.
        num_envs: Number of environments.

    Returns:
        export_path (same as input).
    """
    deactivated = []
    if use_cloning and num_envs > 1:
        logger.info("Deactivating %d cloned environments...", num_envs - 1)
        for env_idx in range(1, num_envs):
            env_path = f"/World/envs/env_{env_idx}"
            prim = stage.GetPrimAtPath(env_path)
            if prim.IsValid() and prim.IsActive():
                prim.SetActive(False)
                deactivated.append(prim)
                if env_idx <= 3 or env_idx == num_envs - 1:
                    logger.info("Deactivated: %s", env_path)
        if num_envs > 5:
            logger.info("... (deactivated %d environments total)", len(deactivated))

    try:
        stage.Export(export_path)
        return export_path
    finally:
        if deactivated:
            logger.info("Reactivating %d environments...", len(deactivated))
            for prim in deactivated:
                if prim.IsValid():
                    prim.SetActive(True)
