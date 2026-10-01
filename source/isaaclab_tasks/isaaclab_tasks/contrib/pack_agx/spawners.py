# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prototype material overrides for the textured AGX props."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pxr import Sdf, Usd, UsdShade

from isaaclab.sim import clone, spawn_from_usd

if TYPE_CHECKING:
    from .config.env_config import TexturedPropCfg


@clone
def spawn_prop(prim_path: str, cfg: TexturedPropCfg, *args, **kwargs) -> Usd.Prim:
    """Keep authored textures, overriding their metallic and roughness channels before cloning."""
    root = spawn_from_usd(prim_path, cfg, *args, **kwargs)
    values = {
        "metallic_texture_influence": 0.0,
        "reflection_roughness_texture_influence": 0.0,
        "metallic_constant": cfg.metallic,
        "reflection_roughness_constant": cfg.roughness,
        "albedo_brightness": cfg.brightness,
    }
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            for name, value in values.items():
                shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
    return root
