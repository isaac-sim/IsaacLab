# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD string construction for OVRTX render products."""

from __future__ import annotations

import math


def get_render_var_config(data_types: list[str]) -> tuple[str, str, str]:
    """Return (render_var_path, render_var_name, source_name) from data_types."""
    use_depth = any(dt in ["depth", "distance_to_image_plane", "distance_to_camera"] for dt in data_types)
    use_distance_to_camera = "distance_to_camera" in data_types and not any(
        dt in ["depth", "distance_to_image_plane"] for dt in data_types
    )
    use_albedo = "albedo" in data_types
    use_semantic = "semantic_segmentation" in data_types
    use_instance_seg = "instance_segmentation" in data_types
    use_normals = "normals" in data_types
    use_motion_vectors = "motion_vectors" in data_types
    use_rgb = any(dt in ["rgb", "rgba"] for dt in data_types)
    use_hdr = "rgb_hdr" in data_types

    if use_depth and not (
        use_rgb or use_albedo or use_semantic or use_instance_seg or use_normals or use_motion_vectors
    ):
        source = "DistanceToCameraSD" if use_distance_to_camera else "DistanceToImagePlaneSD"
        return "/Render/Vars/depth", "depth", source
    if use_albedo and not (use_rgb or use_semantic or use_instance_seg or use_normals or use_motion_vectors):
        return "/Render/Vars/albedo", "albedo", "DiffuseAlbedoSD"
    if use_semantic and not (use_rgb or use_albedo or use_normals or use_motion_vectors):
        return "/Render/Vars/semantic", "semantic", "SemanticSegmentation"
    if use_instance_seg and not (
        use_rgb or use_albedo or use_semantic or use_normals or use_depth or use_hdr or use_motion_vectors
    ):
        return (
            "/Render/Vars/NonStableInstanceSegmentation",
            "NonStableInstanceSegmentation",
            "NonStableInstanceSegmentation",
        )
    if use_normals and not (
        use_rgb or use_albedo or use_semantic or use_instance_seg or use_depth or use_motion_vectors
    ):
        return "/Render/Vars/NormalSD", "NormalSD", "NormalSD"
    if use_motion_vectors and not (
        use_rgb or use_albedo or use_semantic or use_instance_seg or use_depth or use_normals
    ):
        return "/Render/Vars/TargetMotionSD", "TargetMotionSD", "TargetMotionSD"
    if use_hdr and not use_rgb:
        return "/Render/Vars/HdrColor", "HdrColor", "HdrColor"
    return "/Render/Vars/LdrColor", "LdrColor", "LdrColor"


def get_render_var_configs(data_types: list[str]) -> list[tuple[str, str, str]]:
    """Return render var configs needed for the requested data types.

    Each config is a ``(render_var_path, render_var_name, source_name)`` tuple as defined by
    :func:`get_render_var_config`. Always includes the single render var resolved by
    :func:`get_render_var_config`, plus the following extras when applicable:

    * ``HdrColor`` — when both ``"rgb"`` (or ``"rgba"``) and ``"rgb_hdr"`` are requested, so
      PPISP can consume the HDR AOV alongside the LDR destination on the same render product.
    * ``SemanticIdMap`` — when ``"semantic_segmentation"`` is requested, so the
      semantic-ID-to-label mapping can be decoded for ``camera.data.info``.
    * ``StableIdSemanticIdMap``, ``StableIdMap``, ``SemanticIdMap`` — when
      ``"instance_segmentation"`` is requested, so the instance-ID-to-prim-path
      (``idToLabels``) and instance-ID-to-semantic (``idToSemantics``) mappings can be decoded.

    Other multi-AOV combinations are not supported.
    """
    data_types = data_types if data_types else ["rgb"]
    render_vars: list[tuple[str, str, str]] = [get_render_var_config(data_types)]
    use_rgb = any(dt in ["rgb", "rgba"] for dt in data_types)
    if use_rgb and "rgb_hdr" in data_types:
        render_vars.append(("/Render/Vars/HdrColor", "HdrColor", "HdrColor"))
    # Author the ID-to-label map render vars needed to decode the segmentation info dicts. These are keyed off
    # the requested data types (not the single AOV resolved by get_render_var_config) so they are still authored
    # when segmentation is combined with other outputs. instance_segmentation needs StableIdSemanticIdMap +
    # StableIdMap to resolve each pixel to a prim path.
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


def build_render_product_as_string(
    width: int,
    height: int,
    num_envs: int,
    data_types: list[str],
    camera_path: str,
    minimal_mode: int | None = None,
    background_color: tuple[float, float, float] | None = None,
) -> tuple[str, str]:
    """Build the render product USD snippet as a string.

    This string is meant to be appended to an exported stage (ASCII) before loading into OVRTX.
    The initial camera relationship targets one camera authored in the prototype stage.
    Multi-environment rendering rewrites the relationship with every resolved camera path after
    runtime cloning.

    Args:
        width: Tile width from sensor config [px].
        height: Tile height from sensor config [px].
        num_envs: Number of environments from scene.
        data_types: Data types from sensor config.
        camera_path: Absolute path of a camera authored in the prototype stage.
        minimal_mode: RTX minimal mode. None if not requested. Valid values are 1, 2, 3.
        background_color: Solid background color as normalized RGB floats ``(r, g, b)`` in ``[0, 1]``.
            When set, the render product uses a solid color background instead of the dome light.
            When ``None``, the default dome-light background is used.

    Returns:
        Tuple of (render product USD snippet as a string, absolute render product prim path).
    """
    data_types = data_types if data_types else ["rgb"]
    num_cols = math.ceil(math.sqrt(num_envs))
    tiled_width, tiled_height = num_cols * width, math.ceil(num_envs / num_cols) * height

    camera_paths = [camera_path]
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
