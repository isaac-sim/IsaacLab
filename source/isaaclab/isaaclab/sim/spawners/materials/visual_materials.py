# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
import omni.kit.commands
from pxr import Usd

from isaaclab.sim.utils import clone, safe_set_attribute_on_usd_prim
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

if TYPE_CHECKING:
    from . import visual_materials_cfg


@clone
def spawn_preview_surface(prim_path: str, cfg: visual_materials_cfg.PreviewSurfaceCfg) -> Usd.Prim:
    """Create a preview surface prim and override the settings with the given config.

    A preview surface is a physically-based surface that handles simple shaders while supporting
    both *specular* and *metallic* workflows. All color inputs are in linear color space (RGB).
    For more information, see the `documentation <https://openusd.org/release/spec_usdpreviewsurface.html>`__.

    The function calls the USD command `CreatePreviewSurfaceMaterialPrim`_ to create the prim.

    .. _CreatePreviewSurfaceMaterialPrim: https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/omni.usd.commands/omni.usd.commands.CreatePreviewSurfaceMaterialPrimCommand.html

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
    # spawn material if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path):
        omni.kit.commands.execute("CreatePreviewSurfaceMaterialPrim", mtl_path=prim_path, select_new_prim=False)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    # obtain prim
    prim = prim_utils.get_prim_at_path(f"{prim_path}/Shader")
    # apply properties
    cfg = cfg.to_dict()
    del cfg["func"]
    for attr_name, attr_value in cfg.items():
        safe_set_attribute_on_usd_prim(prim, f"inputs:{attr_name}", attr_value, camel_case=True)
    # return prim
    return prim


@clone
def spawn_from_mdl_file(prim_path: str, cfg: visual_materials_cfg.MdlMaterialCfg) -> Usd.Prim:
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
    # spawn material if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path):
        # extract material name from path
        material_name = cfg.mdl_path.split("/")[-1].split(".")[0]
        omni.kit.commands.execute(
            "CreateMdlMaterialPrim",
            mtl_url=cfg.mdl_path.format(NVIDIA_NUCLEUS_DIR=NVIDIA_NUCLEUS_DIR),
            mtl_name=material_name,
            mtl_path=prim_path,
            select_new_prim=False,
        )
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    # obtain prim
    prim = prim_utils.get_prim_at_path(f"{prim_path}/Shader")
    # apply properties
    cfg = cfg.to_dict()
    del cfg["func"]
    del cfg["mdl_path"]
    for attr_name, attr_value in cfg.items():
        safe_set_attribute_on_usd_prim(prim, f"inputs:{attr_name}", attr_value, camel_case=False)
    # return prim
    return prim
