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
import torch
import trimesh

from pxr import UsdGeom

import isaaclab.terrains as terrain_gen
from isaaclab.sim import PreviewSurfaceCfg, build_simulation_context, get_first_matching_child_prim
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

pytestmark = pytest.mark.integration


def test_terrain_generation():
    """Generates assorted terrains and tests that the resulting mesh has the correct size."""
    with build_simulation_context(device="cuda:0", auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Handler for terrains importing
        terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            max_init_terrain_level=None,
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            num_envs=1,
        )
        terrain_importer = TerrainImporter(terrain_importer_cfg)

        # check if mesh prim path exists
        mesh_prim_path = terrain_importer.cfg.prim_path + "/terrain"
        assert mesh_prim_path in terrain_importer.terrain_prim_paths

        # obtain underling mesh
        mesh = _obtain_collision_mesh(mesh_prim_path, mesh_type="Mesh")
        assert mesh is not None

        # calculate expected size from config
        cfg = terrain_importer.cfg.terrain_generator
        assert cfg is not None
        expectedSizeX = cfg.size[0] * cfg.num_rows + 2 * cfg.border_width
        expectedSizeY = cfg.size[1] * cfg.num_cols + 2 * cfg.border_width

        # get size from mesh bounds
        bounds = mesh.bounds
        actualSize = abs(bounds[1] - bounds[0])

        assert actualSize[0] == pytest.approx(expectedSizeX)
        assert actualSize[1] == pytest.approx(expectedSizeY)


def test_visual_material_defaults():
    """Resolves omitted visual materials by terrain type while preserving an explicit None."""
    generator_cfg = TerrainImporterCfg(prim_path="/World/generated")
    assert isinstance(generator_cfg.visual_material, PreviewSurfaceCfg)
    assert generator_cfg.visual_material.diffuse_color == (0.0, 0.0, 0.0)

    plane_cfg = TerrainImporterCfg(prim_path="/World/plane", terrain_type="plane")
    assert plane_cfg.visual_material is None

    unmaterialized_generator_cfg = TerrainImporterCfg(
        prim_path="/World/unmaterialized", terrain_type="generator", visual_material=None
    )
    assert unmaterialized_generator_cfg.visual_material is None


def test_plane():
    """Generates a plane and tests that the resulting mesh has the correct size."""
    with build_simulation_context(device="cuda:0", auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        # Handler for terrains importing
        terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            num_envs=4096,
            env_spacing=4.0,
            visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        )
        terrain_importer = TerrainImporter(terrain_importer_cfg)

        # The importer lays the environments out on a 64 x 64 grid with the configured spacing.
        origins = terrain_importer.env_origins
        assert origins.shape == (4096, 3)
        assert origins.device == torch.device("cuda:0")
        torch.testing.assert_close(origins[0].cpu(), torch.tensor([126.0, -126.0, 0.0]))
        torch.testing.assert_close(origins[-1].cpu(), torch.tensor([-126.0, 126.0, 0.0]))

        # check if mesh prim path exists
        mesh_prim_path = terrain_importer.cfg.prim_path + "/terrain"
        assert mesh_prim_path in terrain_importer.terrain_prim_paths

        # The visual mesh is bounded to the environment grid while the collision Plane stays infinite.
        environment = sim.stage.GetPrimAtPath(f"{mesh_prim_path}/Environment")
        assert tuple(environment.GetAttribute("xformOp:scale").Get()) == pytest.approx((2.6, 2.6, 1.0))
        visual_mesh = UsdGeom.Mesh(sim.stage.GetPrimAtPath(f"{mesh_prim_path}/Environment/Geometry"))
        assert [tuple(uv) for uv in UsdGeom.PrimvarsAPI(visual_mesh).GetPrimvar("st").Get()] == [
            (-65.0, -65.0),
            (65.0, -65.0),
            (65.0, 65.0),
            (-65.0, 65.0),
        ]

        # Direct imports use the same bounded default instead of the legacy 2,000 km visual mesh.
        terrain_importer.import_ground_plane("direct")
        direct_environment = sim.stage.GetPrimAtPath(f"{terrain_importer.cfg.prim_path}/direct/Environment")
        assert tuple(direct_environment.GetAttribute("xformOp:scale").Get()) == pytest.approx((2.6, 2.6, 1.0))
        direct_mesh = UsdGeom.Mesh(
            sim.stage.GetPrimAtPath(f"{terrain_importer.cfg.prim_path}/direct/Environment/Geometry")
        )
        assert [tuple(uv) for uv in UsdGeom.PrimvarsAPI(direct_mesh).GetPrimvar("st").Get()] == [
            (-65.0, -65.0),
            (65.0, -65.0),
            (65.0, 65.0),
            (-65.0, 65.0),
        ]

        # obtain underling mesh
        mesh = _obtain_collision_mesh(mesh_prim_path, mesh_type="Plane")
        assert mesh is None


def test_usd():
    """Imports terrain from a usd and tests that the resulting mesh has the correct size."""
    with build_simulation_context(device="cuda:0", auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Handler for terrains importing
        terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
            num_envs=1,
            env_spacing=1.0,
        )
        terrain_importer = TerrainImporter(terrain_importer_cfg)

        # check if mesh prim path exists
        mesh_prim_path = terrain_importer.cfg.prim_path + "/terrain"
        assert mesh_prim_path in terrain_importer.terrain_prim_paths

        # obtain underling mesh
        mesh = _obtain_collision_mesh(mesh_prim_path, mesh_type="Mesh")
        assert mesh is not None

        # expect values from USD file
        expectedSizeX = 96
        expectedSizeY = 96

        # get size from mesh bounds
        bounds = mesh.bounds
        actualSize = abs(bounds[1] - bounds[0])

        assert actualSize[0] == pytest.approx(expectedSizeX)
        assert actualSize[1] == pytest.approx(expectedSizeY)


def _obtain_collision_mesh(mesh_prim_path: str, mesh_type: Literal["Mesh", "Plane"]) -> trimesh.Trimesh | None:
    """Get the collision mesh from the terrain."""
    # traverse the prim and get the collision mesh
    mesh_prim = get_first_matching_child_prim(mesh_prim_path, lambda prim: prim.GetTypeName() == mesh_type)
    # check it is valid
    assert mesh_prim.IsValid()

    if mesh_prim.GetTypeName() == "Mesh":
        # cast into UsdGeomMesh
        mesh_prim = UsdGeom.Mesh(mesh_prim)
        # store the mesh
        vertices = np.asarray(mesh_prim.GetPointsAttr().Get())
        faces = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        return None
