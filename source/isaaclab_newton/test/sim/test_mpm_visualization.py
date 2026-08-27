# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for USD-stage MPM particle visualization."""

from types import SimpleNamespace

import numpy as np
from isaaclab_newton.sim.spawners.mpm.visualization import create_mpm_particle_visualization

from pxr import Gf, Usd, UsdGeom, UsdShade

import isaaclab.sim as sim_utils


def _create_visualization(monkeypatch, visual_material):
    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(sim_utils, "get_current_stage", lambda: stage)

    prim_paths = create_mpm_particle_visualization(
        prim_paths=["/World/Particles/env_0"],
        positions=np.zeros((1, 2, 3), dtype=np.float32),
        widths=np.full(2, 0.01, dtype=np.float32),
        color=(0.1, 0.2, 0.3),
        visual_material=visual_material,
    )
    return stage, prim_paths[0]


def _assert_display_color_fallback(stage, prim_path):
    points = UsdGeom.Points(stage.GetPrimAtPath(prim_path))
    assert points.GetDisplayColorAttr().Get() == [Gf.Vec3f(0.1, 0.2, 0.3)]
    assert UsdShade.MaterialBindingAPI(points).GetDirectBindingRel().GetTargets() == []


def test_missing_visual_material_prim_falls_back_to_display_color(monkeypatch):
    """A material spawner that authors no valid USD material leaves the points unbound."""
    stage, prim_path = _create_visualization(
        monkeypatch,
        SimpleNamespace(func=lambda _prim_path, _cfg: None),
    )

    _assert_display_color_fallback(stage, prim_path)


def test_particle_visualization_uses_standard_material_binding_helper(monkeypatch):
    """A valid material is bound through Isaac Lab's visual-material helper."""
    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(sim_utils, "get_current_stage", lambda: stage)
    bindings = []
    monkeypatch.setattr(
        sim_utils,
        "bind_visual_material",
        lambda prim_path, material_path, stage: bindings.append((prim_path, material_path, stage)),
    )

    def spawn_material(prim_path, _cfg):
        UsdShade.Material.Define(stage, prim_path)

    visual_material = SimpleNamespace(func=spawn_material)
    prim_paths = create_mpm_particle_visualization(
        prim_paths=["/World/Particles/env_0"],
        positions=np.zeros((1, 2, 3), dtype=np.float32),
        widths=np.full(2, 0.01, dtype=np.float32),
        color=(0.1, 0.2, 0.3),
        visual_material=visual_material,
    )

    assert bindings == [(prim_paths[0], "/World/Particles/Looks/visualMaterial", stage)]
