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

if TYPE_CHECKING:
    from . import visual_materials_cfg


@clone
def spawn_preview_surface(prim_path: str, cfg: visual_materials_cfg.PreviewSurfaceCfg) -> Usd.Prim:
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
    # get stage handle
    stage = get_current_stage()

    # spawn material if it doesn't exist.
    if not stage.GetPrimAtPath(prim_path).IsValid():
        material = UsdShade.Material.Define(stage, prim_path)
        shader = UsdShade.Shader.Define(stage, f"{prim_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float)
        surface = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        displacement = shader.CreateOutput("displacement", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(surface)
        material.CreateDisplacementOutput().ConnectToSource(displacement)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    # obtain prim
    prim = stage.GetPrimAtPath(f"{prim_path}/Shader")
    # check prim is valid
    if not prim.IsValid():
        raise ValueError(f"Failed to create preview surface material at path: '{prim_path}'.")
    # apply properties
    cfg = cfg.to_dict()  # type: ignore
    del cfg["func"]
    for attr_name, attr_value in cfg.items():
        safe_set_attribute_on_usd_prim(prim, f"inputs:{attr_name}", attr_value, camel_case=True)

    return prim


@clone
def spawn_from_mdl_file(
    prim_path: str, cfg: visual_materials_cfg.MdlFileCfg | visual_materials_cfg.GlassMdlCfg
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
    # get stage handle
    stage = get_current_stage()

    # spawn material if it doesn't exist.
    if not stage.GetPrimAtPath(prim_path).IsValid():
        # extract material name from path
        material_name = cfg.mdl_path.split("/")[-1].split(".")[0]
        mdl_url = cfg.mdl_path.format(NVIDIA_NUCLEUS_DIR=NVIDIA_NUCLEUS_DIR)
        material = UsdShade.Material.Define(stage, prim_path)
        shader = UsdShade.Shader.Define(stage, f"{prim_path}/Shader")
        shader.SetSourceAsset(Sdf.AssetPath(mdl_url), "mdl")
        shader.SetSourceAssetSubIdentifier(material_name, "mdl")
        shader_out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        shader_out.SetRenderType("material")
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader_out)
        material.CreateDisplacementOutput("mdl").ConnectToSource(shader_out)
        material.CreateVolumeOutput("mdl").ConnectToSource(shader_out)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    # obtain prim
    prim = stage.GetPrimAtPath(f"{prim_path}/Shader")
    # check prim is valid
    if not prim.IsValid():
        raise ValueError(f"Failed to create MDL material at path: '{prim_path}'.")
    # apply properties
    cfg = cfg.to_dict()  # type: ignore
    del cfg["func"]
    del cfg["mdl_path"]
    for attr_name, attr_value in cfg.items():
        safe_set_attribute_on_usd_prim(prim, f"inputs:{attr_name}", attr_value, camel_case=False)
    # return prim
    return prim
