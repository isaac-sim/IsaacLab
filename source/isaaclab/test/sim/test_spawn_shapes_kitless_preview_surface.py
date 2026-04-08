# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from unittest import mock

import pytest
from pxr import UsdGeom, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import visual_materials
from isaaclab.sim.utils import prims as prim_utils


def test_spawn_shape_authors_preview_surface_and_display_color_in_kitless_mode():
    """Primitive shape spawners should still author PreviewSurface metadata in kitless mode."""
    stage = sim_utils.create_new_stage()
    stage.DefinePrim("/World", "Xform")

    cfg = sim_utils.CuboidCfg(
        size=(0.4, 0.5, 0.6),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.8),
    )

    with sim_utils.use_stage(stage):
        with (
            mock.patch.object(visual_materials, "has_kit", return_value=False),
            mock.patch.object(prim_utils, "has_kit", return_value=False),
        ):
            prim = cfg.func("/World/Cube", cfg)

    assert prim.IsValid()

    material = UsdShade.Material.Get(stage, "/World/Cube/geometry/material")
    assert material
    assert material.GetPrim().IsValid()

    mesh_prim = stage.GetPrimAtPath("/World/Cube/geometry/mesh")
    shader_prim = stage.GetPrimAtPath("/World/Cube/geometry/material/Shader")
    assert shader_prim.IsValid()
    assert shader_prim.GetAttribute("inputs:diffuseColor").Get() == (1.0, 0.0, 0.0)

    binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
    assert binding_api.GetDirectBinding().GetMaterialPath() == material.GetPath()

    shader = UsdShade.Shader(shader_prim)
    assert shader.GetInput("opacity").Get() == pytest.approx(0.8)

    mesh = UsdGeom.Gprim(mesh_prim)
    assert not mesh.GetDisplayColorPrimvar().HasAuthoredValue()
    assert not mesh.GetDisplayOpacityPrimvar().HasAuthoredValue()
