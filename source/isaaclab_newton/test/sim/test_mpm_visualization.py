# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for USD-stage MPM particle visualization."""

from types import SimpleNamespace

import numpy as np
import pytest
from isaaclab_newton.sim.spawners.mpm.visualization import create_mpm_particle_visualization

from pxr import Gf, Usd, UsdGeom, UsdShade

import isaaclab.sim as sim_utils

_PRIM_PATHS = ["/World/envs/env_0/Sand/Particles", "/World/envs/env_1/Sand/Particles"]
"""Environment-namespaced points prims, matching the ``{ENV_REGEX_NS}/<asset>/Particles`` paths used by MPMObject."""

_POSITIONS = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
"""Distinct per-environment positions, so a prim showing another environment's slice is caught."""

_WIDTHS = np.array([0.01, 0.02, 0.03], dtype=np.float32)
_COLOR = (0.1, 0.2, 0.3)


def _spawn_material(prim_path, _cfg):
    UsdShade.Material.Define(sim_utils.get_current_stage(), prim_path)


def _create_visualization(monkeypatch, visual_material=None):
    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(sim_utils, "get_current_stage", lambda: stage)

    prim_paths = create_mpm_particle_visualization(
        prim_paths=_PRIM_PATHS,
        positions=_POSITIONS,
        widths=_WIDTHS,
        color=_COLOR,
        visual_material=visual_material,
    )
    return stage, prim_paths


def test_each_environment_renders_its_own_particle_slice(monkeypatch):
    """Every environment gets a points prim carrying its own positions, plus the shared widths and color."""
    stage, prim_paths = _create_visualization(monkeypatch)

    assert prim_paths == _PRIM_PATHS
    for env_idx, prim_path in enumerate(prim_paths):
        points = UsdGeom.Points(stage.GetPrimAtPath(prim_path))
        np.testing.assert_array_equal(np.asarray(points.GetPointsAttr().Get()), _POSITIONS[env_idx])
        np.testing.assert_array_equal(np.asarray(points.GetWidthsAttr().Get()), _WIDTHS)
        assert points.GetDisplayColorAttr().Get() == [Gf.Vec3f(*_COLOR)]


def test_particle_clouds_ignore_the_inherited_environment_transform(monkeypatch):
    """Points are authored in the world frame, so the prims must reset the environment's xform stack."""
    stage, prim_paths = _create_visualization(monkeypatch)

    for prim_path in prim_paths:
        assert UsdGeom.Points(stage.GetPrimAtPath(prim_path)).GetResetXformStack()


def test_prim_path_count_must_match_environment_count(monkeypatch):
    """A path-per-environment mismatch is rejected instead of silently dropping environments."""
    monkeypatch.setattr(sim_utils, "get_current_stage", Usd.Stage.CreateInMemory)

    with pytest.raises(ValueError, match="one particle visualization prim path per environment"):
        create_mpm_particle_visualization(
            prim_paths=_PRIM_PATHS,
            positions=_POSITIONS[:1],
            widths=_WIDTHS,
            color=_COLOR,
        )


def test_missing_visual_material_prim_falls_back_to_display_color(monkeypatch):
    """A material spawner that authors no valid USD material leaves the points unbound."""
    stage, prim_paths = _create_visualization(monkeypatch, SimpleNamespace(func=lambda _prim_path, _cfg: None))

    for prim_path in prim_paths:
        points = UsdGeom.Points(stage.GetPrimAtPath(prim_path))
        assert points.GetDisplayColorAttr().Get() == [Gf.Vec3f(*_COLOR)]
        assert UsdShade.MaterialBindingAPI(points).GetDirectBinding().GetMaterialPath().isEmpty


def test_visual_material_is_bound_to_every_particle_cloud(monkeypatch):
    """A spawned material is bound to each environment's points prim from a sibling ``Looks`` scope."""
    stage, prim_paths = _create_visualization(monkeypatch, SimpleNamespace(func=_spawn_material))

    expected_material_paths = [
        "/World/envs/env_0/Sand/Looks/visualMaterial",
        "/World/envs/env_1/Sand/Looks/visualMaterial",
    ]
    for prim_path, material_path in zip(prim_paths, expected_material_paths, strict=True):
        binding = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim_path)).GetDirectBinding()
        assert str(binding.GetMaterialPath()) == material_path
