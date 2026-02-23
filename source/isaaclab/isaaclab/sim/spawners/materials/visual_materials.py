# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pxr import Usd, UsdShade

from isaaclab.sim.utils import clone, safe_set_attribute_on_usd_prim
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR
from isaaclab.utils.version import has_kit

if TYPE_CHECKING:
    from . import visual_materials_cfg

# import logger
logger = logging.getLogger(__name__)


@clone
def spawn_preview_surface(prim_path: str, cfg: visual_materials_cfg.PreviewSurfaceCfg) -> Usd.Prim:
    """Create a preview surface prim and override the settings with the given config.

    A preview surface is a physically-based surface that handles simple shaders while supporting
    both *specular* and *metallic* workflows. All color inputs are in linear color space (RGB).
    For more information, see the `documentation <https://openusd.org/release/spec_usdpreviewsurface.html>`__.

    The function calls the USD command `CreateShaderPrimFromSdrCommand`_ to create the prim.

    .. _CreateShaderPrimFromSdrCommand: https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/omni.usd.commands/omni.usd.commands.CreateShaderPrimFromSdrCommand.html

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
    # check if Kit is available (required for shader creation commands)
    if not has_kit():
        raise RuntimeError(
            f"Cannot spawn preview surface material '{prim_path}' in kitless mode. "
            "This functionality requires Kit/Isaac Sim as it uses omni.usd.commands."
        )

    # get stage handle
    stage = get_current_stage()

    # spawn material if it doesn't exist.
    if not stage.GetPrimAtPath(prim_path).IsValid():
        # note: we don't use Omniverse's CreatePreviewSurfaceMaterialPrimCommand
        # since it does not support USD stage as an argument. The created material
        # in that case is always the one from USD Context which makes it difficult to
        # handle scene creation on a custom stage.
        material_prim = UsdShade.Material.Define(stage, prim_path)
        if material_prim:
            from omni.usd.commands import CreateShaderPrimFromSdrCommand

            shader_prim = CreateShaderPrimFromSdrCommand(
                parent_path=prim_path,
                identifier="UsdPreviewSurface",
                stage_or_context=stage,
                prim_name="Shader",
            ).do()
            # bind the shader graph to the material
            if shader_prim:
                surface_out = shader_prim.GetOutput("surface")
                if surface_out:
                    material_prim.CreateSurfaceOutput().ConnectToSource(surface_out)

                displacement_out = shader_prim.GetOutput("displacement")
                if displacement_out:
                    material_prim.CreateDisplacementOutput().ConnectToSource(displacement_out)
        else:
            raise ValueError(f"Failed to create preview surface shader at path: '{prim_path}'.")
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

    The function calls the USD command `CreateMdlMaterialPrim`_ to create the prim.

    .. _CreateMdlMaterialPrim: https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/omni.usd.commands/omni.usd.commands.CreateMdlMaterialPrimCommand.html

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
    # check if Kit is available (required for MDL material creation commands)
    if not has_kit():
        raise RuntimeError(
            f"Cannot spawn MDL material '{prim_path}' in kitless mode. "
            "This functionality requires Kit/Isaac Sim as it uses omni.usd.commands."
        )

    # get stage handle
    stage = get_current_stage()

    # spawn material if it doesn't exist.
    if not stage.GetPrimAtPath(prim_path).IsValid():
        # extract material name from path
        material_name = cfg.mdl_path.split("/")[-1].split(".")[0]
        from omni.usd.commands import CreateMdlMaterialPrimCommand

        CreateMdlMaterialPrimCommand(
            mtl_url=cfg.mdl_path.format(NVIDIA_NUCLEUS_DIR=NVIDIA_NUCLEUS_DIR),
            mtl_name=material_name,
            mtl_path=prim_path,
            stage=stage,
            select_new_prim=False,
        ).do()
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
