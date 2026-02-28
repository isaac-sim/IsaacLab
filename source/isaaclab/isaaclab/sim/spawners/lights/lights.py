# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from pxr import Usd, UsdLux

from isaaclab.sim.utils import clone, create_prim, get_current_stage, safe_set_attribute_on_usd_prim

if TYPE_CHECKING:
    from . import lights_cfg


@clone
def spawn_light(
    prim_path: str,
    cfg: lights_cfg.LightCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a light prim at the specified prim path with the specified configuration.

    The created prim is based on the `USD.LuxLight <https://openusd.org/dev/api/class_usd_lux_light_a_p_i.html>`_ API.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration for the light source.
        translation: The translation of the prim. Defaults to None, in which case this is set to the origin.
        orientation: The orientation of the prim as (x, y, z, w). Defaults to None, in which case this
            is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Raises:
        ValueError:  When a prim already exists at the specified prim path.
    """
    # obtain stage handle
    stage = get_current_stage()
    # check if prim already exists
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    # create the prim
    prim = create_prim(
        prim_path, prim_type=cfg.prim_type, translation=translation, orientation=orientation, stage=stage
    )

    # convert to dict
    cfg = cfg.to_dict()
    # delete spawner func specific parameters
    del cfg["prim_type"]
    # delete custom attributes in the config that are not USD parameters
    non_usd_cfg_param_names = ["func", "copy_from_source", "visible", "semantic_tags", "spawn_path"]
    for param_name in non_usd_cfg_param_names:
        del cfg[param_name]
    # set into USD API
    for attr_name, value in cfg.items():
        # special operation for texture properties
        # note: this is only used for dome light
        if "texture" in attr_name:
            light_prim = UsdLux.DomeLight(prim)
            if attr_name == "texture_file":
                light_prim.CreateTextureFileAttr(value)
            elif attr_name == "texture_format":
                light_prim.CreateTextureFormatAttr(value)
            else:
                raise ValueError(f"Unsupported texture attribute: '{attr_name}'.")
        else:
            if attr_name == "visible_in_primary_ray":
                prim_prop_name = attr_name
            else:
                prim_prop_name = f"inputs:{attr_name}"
            # set the attribute
            safe_set_attribute_on_usd_prim(prim, prim_prop_name, value, camel_case=True)
    # return the prim
    return prim
