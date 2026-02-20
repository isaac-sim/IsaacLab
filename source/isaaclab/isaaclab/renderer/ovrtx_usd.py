# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD manipulation for OVRTX: Render scope building, camera injection, and stage prim activation."""

import tempfile
from pathlib import Path

from pxr import Sdf, Usd


def get_render_var_config(
    data_types: list[str], simple_shading_mode: bool
) -> tuple[str, str, str]:
    """Return (render_var_path, render_var_name, source_name) from data_types and shading mode."""
    use_depth = any(
        dt in ["depth", "distance_to_image_plane", "distance_to_camera"] for dt in data_types
    )
    use_albedo = "albedo" in data_types
    use_semantic = "semantic_segmentation" in data_types
    use_rgb = any(dt in ["rgb", "rgba"] for dt in data_types)

    if use_depth and not (use_rgb or use_albedo or use_semantic):
        return "/Render/Vars/depth", "depth", "DistanceToImagePlaneSD"
    if use_albedo and not (use_rgb or use_semantic):
        return "/Render/Vars/albedo", "albedo", "DiffuseAlbedoSD"
    if use_semantic and not (use_rgb or use_albedo):
        return "/Render/Vars/semantic", "semantic", "SemanticSegmentationSD"
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
) -> str:
    """Build the Render scope USD string (def Scope Render, RenderProduct, Vars)."""
    camera_rel_list = ", ".join([f"<{p}>" for p in camera_paths])
    return f'''
def Scope "Render"
{{
    def RenderProduct "{render_product_name}" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    ) {{
        rel camera = [{camera_rel_list}]
        token omni:rtx:background:source:type = "domeLight"
        token omni:rtx:rendermode = "RealTimePathTracing"
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


def inject_cameras_into_usd(
    usd_scene_path: str,
    num_envs: int,
    tiled_width: int,
    tiled_height: int,
    data_types: list[str],
    simple_shading_mode: bool,
) -> tuple[str, str]:
    """Inject camera and render product definitions into an existing USD file.

    Reads the USD file, appends a Render scope (cameras + RenderProduct + Vars),
    writes to a temp file, and returns (path_to_combined_usd, render_product_path).
    """
    with open(usd_scene_path, "r") as f:
        original_usd = f.read()

    camera_paths = [f"/World/envs/env_{i}/Camera" for i in range(num_envs)]
    render_product_name = "RenderProduct"
    render_product_path = f"/Render/{render_product_name}"

    render_var_path, render_var_name, source_name = get_render_var_config(
        data_types, simple_shading_mode
    )
    print(f"  Rendering mode: {render_var_name}")

    camera_content = build_render_scope_usd(
        camera_paths,
        render_product_name,
        render_var_path,
        render_var_name,
        source_name,
        tiled_width,
        tiled_height,
    )
    combined_usd = original_usd.rstrip() + "\n\n" + camera_content

    Path("/tmp/ovrtx_test").mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".usda", delete=False, dir="/tmp/ovrtx_test"
    ) as f:
        f.write(combined_usd)
        temp_path = f.name
    print(f"   Created combined USD: {temp_path}")
    return temp_path, render_product_path


def set_scene_partition_attributes(
    stage, num_envs: int, camera_prim_name: str = "Camera"
) -> int:
    """Set OVRTX scene partition and xform attributes on the stage.

    Sets primvars:omni:scenePartition on each environment and all descendant prims,
    omni:scenePartition and omni:resetXformStack on camera prims, so OVRTX can
    partition rendering and write world transforms via bind_attribute.

    Args:
        stage: USD stage to modify.
        num_envs: Number of environments (/World/envs/env_0 .. env_{num_envs-1}).
        camera_prim_name: Name of the camera prim under each env (e.g. "Camera").

    Returns:
        Total number of objects (non-camera prims) that received partition attributes.
    """
    total_objects = 0
    for env_idx in range(num_envs):
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
            type_name = prim.GetTypeName()
            if not type_name.startswith("Render"):
                reset_xform_attr = prim.CreateAttribute("omni:resetXformStack", Sdf.ValueTypeNames.Bool)
                reset_xform_attr.Set(True)
            total_objects += 1

        camera_path = f"{env_path}/{camera_prim_name}"
        camera_prim = stage.GetPrimAtPath(camera_path)
        if camera_prim.IsValid():
            camera_prim.CreateAttribute("omni:scenePartition", Sdf.ValueTypeNames.Token).Set(partition_name)
            camera_prim.CreateAttribute("omni:resetXformStack", Sdf.ValueTypeNames.Bool).Set(True)

    return total_objects


def export_stage_for_ovrtx(
    stage,
    export_path: str,
    num_envs: int,
    use_ovrtx_cloning: bool,
) -> str:
    """Export the stage to a USD file, optionally with only env_0 for OVRTX cloning.

    If use_ovrtx_cloning and num_envs > 1, deactivates env_1..env_{num_envs-1}
    before export and reactivates them after, so the file contains only env_0.
    The stage is modified in place (deactivate then reactivate).

    Args:
        stage: USD stage to export.
        export_path: Path for the exported file.
        num_envs: Number of environments.
        use_ovrtx_cloning: If True and num_envs > 1, deactivate non-base envs for export.

    Returns:
        export_path (same as input).
    """
    deactivated = []
    if num_envs > 1 and use_ovrtx_cloning:
        deactivated = deactivate_cloned_envs(stage, num_envs)
    try:
        stage.Export(export_path)
        return export_path
    finally:
        if deactivated:
            reactivate_prims(deactivated)


def deactivate_cloned_envs(stage, num_envs: int) -> list:
    """Deactivate all cloned environments (env_1 onwards) on the USD stage.

    Used so only env_0 is exported when using OVRTX internal cloning.
    Returns the list of deactivated prims (for reactivate_prims later).
    """
    deactivated = []
    print(f"[OVRTX OPTIMIZE] Deactivating {num_envs - 1} cloned environments...")
    for env_idx in range(1, num_envs):
        env_path = f"/World/envs/env_{env_idx}"
        prim = stage.GetPrimAtPath(env_path)
        if prim.IsValid() and prim.IsActive():
            prim.SetActive(False)
            deactivated.append(prim)
            if env_idx <= 3 or env_idx == num_envs - 1:
                print(f"  Deactivated: {env_path}")
    if num_envs > 5:
        print(f"  ... (deactivated {len(deactivated)} environments total)")
    return deactivated


def reactivate_prims(prims: list) -> None:
    """Reactivate previously deactivated prims on the USD stage."""
    print(f"[OVRTX OPTIMIZE] Reactivating {len(prims)} environments...")
    for prim in prims:
        if prim.IsValid():
            prim.SetActive(True)
