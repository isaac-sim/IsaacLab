# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from typing import Literal

import numpy as np
import pytest
import trimesh

from pxr import UsdGeom

from isaaclab import cloner as lab_cloner
from isaaclab.sim import PreviewSurfaceCfg, build_simulation_context, get_first_matching_child_prim
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

pytestmark = pytest.mark.integration

_DEVICES = ["cuda:0", "cpu"]
_EXPECTED_PLANE_UVS = [(-65.0, -65.0), (65.0, -65.0), (65.0, 65.0), (-65.0, 65.0)]


def _obtain_collision_mesh(mesh_prim_path: str, mesh_type: Literal["Mesh", "Plane"]) -> trimesh.Trimesh | None:
    """Get the collision mesh from the terrain, or None for an infinite plane collider."""
    mesh_prim = get_first_matching_child_prim(mesh_prim_path, lambda prim: prim.GetTypeName() == mesh_type)
    assert mesh_prim.IsValid()
    if mesh_type != "Mesh":
        return None
    mesh_prim = UsdGeom.Mesh(mesh_prim)
    vertices = np.asarray(mesh_prim.GetPointsAttr().Get())
    faces = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def _assert_mesh_size(mesh: trimesh.Trimesh, expected_size_x: float, expected_size_y: float):
    bounds = mesh.bounds
    actual_size = abs(bounds[1] - bounds[0])
    assert actual_size[0] == pytest.approx(expected_size_x)
    assert actual_size[1] == pytest.approx(expected_size_y)


def _assert_bounded_ground_plane(stage, prim_path: str):
    """The visual mesh is bounded to the environment grid while the collision Plane stays infinite."""
    environment = stage.GetPrimAtPath(f"{prim_path}/Environment")
    assert tuple(environment.GetAttribute("xformOp:scale").Get()) == pytest.approx((2.6, 2.6, 1.0))
    visual_mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{prim_path}/Environment/Geometry"))
    assert [tuple(uv) for uv in UsdGeom.PrimvarsAPI(visual_mesh).GetPrimvar("st").Get()] == _EXPECTED_PLANE_UVS
    assert _obtain_collision_mesh(prim_path, mesh_type="Plane") is None


def test_visual_material_defaults():
    """Resolves omitted visual materials by terrain type while preserving an explicit None."""
    generator_cfg = TerrainImporterCfg(prim_path="/World/generated")
    assert isinstance(generator_cfg.visual_material, PreviewSurfaceCfg)
    assert generator_cfg.visual_material.diffuse_color == (0.0, 0.0, 0.0)
    assert TerrainImporterCfg(prim_path="/World/plane", terrain_type="plane").visual_material is None
    assert TerrainImporterCfg(prim_path="/World/bare", visual_material=None).visual_material is None


@pytest.mark.parametrize("device", _DEVICES)
def test_terrain_importer_env_origins(device):
    """Grid env origins of a plane terrain match Lab's grid_transforms for assorted spacings and env counts."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        for index, (env_spacing, num_envs) in enumerate([(1.0, 1), (1.0, 4), (4.325, 125), (4.325, 379), (8.0, 1024)]):
            terrain_importer = TerrainImporter(
                TerrainImporterCfg(
                    num_envs=num_envs, env_spacing=env_spacing, prim_path=f"/World/ground_{index}", terrain_type="plane"
                )
            )
            lab_grid_origins, _ = lab_cloner.grid_transforms(num_envs, spacing=env_spacing)
            np.testing.assert_allclose(
                terrain_importer.env_origins.cpu().numpy(), lab_grid_origins, rtol=1e-5, atol=1e-5
            )


@pytest.mark.parametrize("device", _DEVICES)
def test_terrain_generation(device):
    """Generated terrains are imported as a collision mesh with the configured footprint.

    A heightfield collider tag is only authored when every sub-terrain opts into heightfield conversion.
    """
    generator_cfg = ROUGH_TERRAINS_CFG.copy()
    generator_cfg.num_rows, generator_cfg.num_cols, generator_cfg.border_width = 1, 2, 2.0
    generator_cfg.seed = 0
    opt_out_cfg = next(iter(generator_cfg.sub_terrains.values()))
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        for name, convert_all in (("mixed", False), ("heightfield", True)):
            opt_out_cfg.convert_to_heightfield = convert_all
            terrain_importer = TerrainImporter(
                TerrainImporterCfg(prim_path=f"/World/{name}", terrain_generator=generator_cfg, num_envs=1)
            )
            mesh_prim_path = f"/World/{name}/terrain"
            assert terrain_importer.terrain_prim_paths == [mesh_prim_path]
            assert terrain_importer.env_origins.shape == (1, 3)

            mesh = _obtain_collision_mesh(mesh_prim_path, mesh_type="Mesh")
            _assert_mesh_size(
                mesh,
                generator_cfg.size[0] * generator_cfg.num_rows + 2 * generator_cfg.border_width,
                generator_cfg.size[1] * generator_cfg.num_cols + 2 * generator_cfg.border_width,
            )
            resolution = sim.stage.GetPrimAtPath(mesh_prim_path).GetAttribute("newton:heightfield:resolution")
            assert resolution.HasAuthoredValue() == convert_all
            if convert_all:
                assert resolution.Get() == pytest.approx(generator_cfg.horizontal_scale)

        # duplicate terrain names are rejected
        with pytest.raises(ValueError, match="already exists"):
            terrain_importer.import_ground_plane("terrain")


@pytest.mark.parametrize("device", _DEVICES)
def test_plane(device):
    """Plane terrains use a bounded visual mesh, tint it from the visual material, and keep an infinite collider."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        terrain_importer = TerrainImporter(
            TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
                num_envs=4096,
                env_spacing=4.0,
                visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        )
        mesh_prim_path = "/World/ground/terrain"
        assert terrain_importer.terrain_prim_paths == [mesh_prim_path]
        _assert_bounded_ground_plane(sim.stage, mesh_prim_path)
        tint = sim.stage.GetAttributeAtPath(f"{mesh_prim_path}/Looks/theGrid/Shader.inputs:diffuse_tint").Get()
        assert tuple(tint) == pytest.approx((1.0, 0.0, 0.0))

        # direct imports use the same bounded default instead of the legacy 2,000 km visual mesh
        terrain_importer.import_ground_plane("direct")
        _assert_bounded_ground_plane(sim.stage, "/World/ground/direct")


def test_usd():
    """Imports terrain from a USD and checks the resulting collision mesh has the size authored in the file."""
    with build_simulation_context(auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        terrain_importer = TerrainImporter(
            TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="usd",
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
                num_envs=1,
                env_spacing=1.0,
            )
        )
        mesh_prim_path = "/World/ground/terrain"
        assert terrain_importer.terrain_prim_paths == [mesh_prim_path]
        _assert_mesh_size(_obtain_collision_mesh(mesh_prim_path, mesh_type="Mesh"), 96, 96)
