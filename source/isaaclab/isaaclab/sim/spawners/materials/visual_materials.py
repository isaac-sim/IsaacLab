# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from pxr import Sdf, Usd, UsdShade

from isaaclab.sim.utils import clone, safe_set_attribute_on_usd_prim
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR
from isaaclab.utils.string import to_camel_case

if TYPE_CHECKING:
    from . import visual_materials_cfg


@clone
def spawn_preview_surface(
    prim_path: str,
    cfg: visual_materials_cfg.PreviewSurfaceCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Create a preview surface prim and override the settings with the given config.

    A preview surface is a physically-based surface that handles simple shaders while supporting
    both *specular* and *metallic* workflows. All color inputs are in linear color space (RGB).
    For more information, see the `documentation <https://openusd.org/release/spec_usdpreviewsurface.html>`__.

    The material is authored using the standard OpenUSD :class:`UsdShade` schema and can therefore
    be consumed by any renderer that supports ``UsdPreviewSurface``.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    del translation, orientation
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    material = UsdShade.Material.Define(stage, prim_path)
    shader = UsdShade.Shader.Define(stage, f"{prim_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    material.CreateDisplacementOutput().ConnectToSource(shader.CreateOutput("displacement", Sdf.ValueTypeNames.Token))
    _author_cfg_inputs(shader.GetPrim(), cfg, camel_case=True)
    return shader.GetPrim()


@clone
def spawn_from_mdl_file(
    prim_path: str,
    cfg: visual_materials_cfg.MdlFileCfg | visual_materials_cfg.GlassMdlCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Load a material from its MDL file and override the settings with the given config.

    NVIDIA's `Material Definition Language (MDL) <https://www.nvidia.com/en-us/design-visualization/technologies/material-definition-language/>`__
    is a language for defining physically-based materials. The MDL file format is a binary format
    that can be loaded by Omniverse and other applications such as Adobe Substance Designer.
    To learn more about MDL, see the `documentation <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials.html>`_.

    The shader network is authored directly with :mod:`UsdShade`.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    del translation, orientation
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    material_name = cfg.mdl_path.rsplit("/", 1)[-1].removesuffix(".mdl")
    material = UsdShade.Material.Define(stage, prim_path)
    shader = UsdShade.Shader.Define(stage, f"{prim_path}/Shader")
    shader.SetSourceAsset(Sdf.AssetPath(cfg.mdl_path.format(NVIDIA_NUCLEUS_DIR=NVIDIA_NUCLEUS_DIR)), "mdl")
    shader.SetSourceAssetSubIdentifier(material_name, "mdl")
    output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    output.SetRenderType("material")
    material.CreateSurfaceOutput("mdl").ConnectToSource(output)
    material.CreateDisplacementOutput("mdl").ConnectToSource(output)
    material.CreateVolumeOutput("mdl").ConnectToSource(output)
    _author_cfg_inputs(shader.GetPrim(), cfg, camel_case=False, ignored=("mdl_path",))
    return shader.GetPrim()


def _author_cfg_inputs(prim: Usd.Prim, cfg, *, camel_case: bool, ignored: tuple[str, ...] = ()) -> None:
    """Author material-specific config fields as shader inputs."""
    ignored = (*ignored, "func", "visible", "semantic_tags", "copy_from_source", "spawn_path")
    for name, value in cfg.to_dict().items():
        if name not in ignored and value is not None:
            input_name = to_camel_case(name, to="cC") if camel_case else name
            if name in {"diffuse_color", "emissive_color", "diffuse_color_constant", "glass_color"}:
                prim.CreateAttribute(f"inputs:{input_name}", Sdf.ValueTypeNames.Color3f)
            safe_set_attribute_on_usd_prim(prim, f"inputs:{name}", value, camel_case=camel_case)
