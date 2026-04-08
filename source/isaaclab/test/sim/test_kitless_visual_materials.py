# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from unittest import mock

import pytest
from pxr import UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import visual_materials
from isaaclab.sim.utils import prims as prim_utils


def test_spawn_preview_surface_authors_material_in_kitless_mode():
    """PreviewSurface should still be authored with pure USD APIs when Kit is unavailable."""
    stage = sim_utils.create_new_stage()
    stage.DefinePrim("/World", "Xform")

    with sim_utils.use_stage(stage):
        with mock.patch.object(visual_materials, "has_kit", return_value=False):
            cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.5, 0.75), roughness=0.2, metallic=0.4)
            shader_prim = cfg.func("/World/Looks/PreviewSurface", cfg)

    assert shader_prim.IsValid()
    assert str(shader_prim.GetPath()) == "/World/Looks/PreviewSurface/Shader"
    assert shader_prim.GetAttribute("info:id").Get() == "UsdPreviewSurface"
    assert shader_prim.GetAttribute("inputs:diffuseColor").Get() == cfg.diffuse_color
    assert shader_prim.GetAttribute("inputs:roughness").Get() == pytest.approx(cfg.roughness)
    assert shader_prim.GetAttribute("inputs:metallic").Get() == pytest.approx(cfg.metallic)

    material = UsdShade.Material.Get(stage, "/World/Looks/PreviewSurface")
    assert material
    assert material.GetPrim().IsValid()
    assert material.GetSurfaceOutput().HasConnectedSource()


def test_bind_visual_material_works_in_kitless_mode():
    """Visual material binding should fall back to USD MaterialBindingAPI in kitless mode."""
    stage = sim_utils.create_new_stage()
    stage.DefinePrim("/World", "Xform")
    mesh_prim = stage.DefinePrim("/World/Geom/Cube", "Cube")
    material = UsdShade.Material.Define(stage, "/World/Looks/TestMaterial")

    with sim_utils.use_stage(stage):
        with mock.patch.object(prim_utils, "has_kit", return_value=False):
            sim_utils.bind_visual_material("/World/Geom/Cube", "/World/Looks/TestMaterial", stage=stage)

    binding_api = UsdShade.MaterialBindingAPI(mesh_prim)
    direct_binding = binding_api.GetDirectBinding()
    assert direct_binding.GetMaterialPath() == material.GetPath()
    assert direct_binding.GetMaterialPurpose() == ""


@pytest.mark.parametrize("stronger_than_descendants", [True, False])
def test_bind_visual_material_preserves_requested_binding_strength_in_kitless_mode(stronger_than_descendants):
    stage = sim_utils.create_new_stage()
    stage.DefinePrim("/World", "Xform")
    mesh_prim = stage.DefinePrim("/World/Geom/Cube", "Cube")
    UsdShade.Material.Define(stage, "/World/Looks/TestMaterial")

    with sim_utils.use_stage(stage):
        with mock.patch.object(prim_utils, "has_kit", return_value=False):
            sim_utils.bind_visual_material(
                "/World/Geom/Cube",
                "/World/Looks/TestMaterial",
                stage=stage,
                stronger_than_descendants=stronger_than_descendants,
            )

    relationship = mesh_prim.GetRelationship("material:binding")
    assert relationship.GetTargets() == [stage.GetPrimAtPath("/World/Looks/TestMaterial").GetPath()]
    assert relationship.GetMetadata("bindMaterialAs") == (
        UsdShade.Tokens.strongerThanDescendants
        if stronger_than_descendants
        else UsdShade.Tokens.weakerThanDescendants
    )
