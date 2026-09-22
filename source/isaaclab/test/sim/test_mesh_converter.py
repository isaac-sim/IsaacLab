# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import os
import tempfile

import pytest
from isaaclab_physx.sim.schemas import (
    PhysxConvexDecompositionCfg,
    PhysxConvexHullCfg,
    PhysxSDFMeshCfg,
    PhysxTriangleMeshCfg,
    PhysxTriangleMeshSimplificationCfg,
)

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import MESH_APPROXIMATION_TOKENS, schemas_cfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def assets() -> dict[str, str]:
    """Download the duck test meshes once and map each file extension to its local path."""
    assets_dir = f"{ISAACLAB_NUCLEUS_DIR}/Tests/MeshConverter/duck"
    download_dir = tempfile.mkdtemp(suffix="_mesh_converter_test_assets")
    return {
        ext: retrieve_file_path(f"{assets_dir}/{name}", download_dir=download_dir)
        for ext, name in (("obj", "duck.obj"), ("stl", "duck.stl"), ("fbx", "duck.fbx"), ("mtl", "duck.mtl"))
    } | {"png": retrieve_file_path(f"{assets_dir}/duckCM.png", download_dir=download_dir)}


@pytest.fixture(autouse=True)
def stage():
    """Spawn each converted asset into a fresh stage."""
    return sim_utils.create_new_stage()


def _spawn(usd_path: str, prim_path: str = "/World/Object"):
    """Reference the converted USD under ``prim_path`` and return the prim."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim(prim_path, usd_path=usd_path)
    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsValid()
    return prim


def test_lazy_conversion_cache(assets):
    """Conversion is skipped for an unchanged asset and config, and re-run when the config changes."""
    config = MeshConverterCfg(asset_path=assets["obj"])
    converter = MeshConverter(config)
    created = os.stat(converter.usd_path).st_mtime_ns

    config.usd_dir = converter.usd_dir
    assert os.stat(MeshConverter(config).usd_path).st_mtime_ns == created

    config.make_instanceable = not config.make_instanceable
    assert os.stat(MeshConverter(config).usd_path).st_mtime_ns != created


@pytest.mark.parametrize("mesh_format", ["obj", "stl", "fbx"])
def test_convert_mesh(assets, mesh_format):
    """Each supported format converts to a Z-up, metric USD that is loadable twice and carries the cfg transform."""
    config = MeshConverterCfg(
        asset_path=assets[mesh_format],
        scale=(0.5, 1.5, 2.0),
        translation=(1.0, -2.0, 3.0),
        rotation=(0.0, 0.0, 0.7071068, 0.7071068),
    )
    converter = MeshConverter(config)

    converted = Usd.Stage.Open(converter.usd_path)
    assert UsdGeom.GetStageUpAxis(converted) == "Z"
    assert UsdGeom.GetStageMetersPerUnit(converted) == 1.0

    _spawn(converter.usd_path, "/World/Object")
    _spawn(converter.usd_path, "/World/Object2")
    geometry = sim_utils.get_current_stage().GetPrimAtPath("/World/Object/geometry")
    assert tuple(geometry.GetAttribute("xformOp:translate").Get()) == config.translation
    quat = geometry.GetAttribute("xformOp:orient").Get()
    assert (*quat.GetImaginary(), quat.GetReal()) == config.rotation
    assert tuple(geometry.GetAttribute("xformOp:scale").Get()) == config.scale


@pytest.mark.parametrize(
    ("mesh_collision_props", "collision_enabled"),
    [
        pytest.param(None, True, id="no_approximation"),
        pytest.param(PhysxConvexHullCfg(), True, id="convex_hull"),
        pytest.param(PhysxConvexDecompositionCfg(), True, id="convex_decomposition"),
        pytest.param(PhysxTriangleMeshCfg(), True, id="triangle_mesh"),
        pytest.param(PhysxTriangleMeshSimplificationCfg(), True, id="mesh_simplification"),
        pytest.param(PhysxSDFMeshCfg(), True, id="sdf"),
        pytest.param(schemas_cfg.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="boundingCube"), True, id="cube"),
        pytest.param(
            schemas_cfg.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="boundingSphere"), False, id="disabled"
        ),
    ],
)
def test_collider_settings(assets, mesh_collision_props, collision_enabled):
    """Collision enablement and the mesh approximation token are authored on the (instanceable) mesh prim."""
    config = MeshConverterCfg(
        asset_path=assets["obj"],
        collision_props=[schemas_cfg.UsdPhysicsCollisionCfg(collision_enabled=collision_enabled)],
        mesh_collision_props=None if mesh_collision_props is None else [mesh_collision_props],
    )
    converter = MeshConverter(config)

    _spawn(converter.usd_path)
    stage = sim_utils.get_current_stage()
    geometry = stage.GetPrimAtPath("/World/Object/geometry")
    assert geometry.IsInstanceable() == config.make_instanceable
    # de-instance to inspect the prototype's mesh prim
    geometry.SetInstanceable(False)
    mesh_prim = stage.GetPrimAtPath("/World/Object/geometry/mesh")

    assert UsdPhysics.CollisionAPI(mesh_prim).GetCollisionEnabledAttr().Get() == collision_enabled
    if collision_enabled and mesh_collision_props is not None:
        expected_token = MESH_APPROXIMATION_TOKENS[mesh_collision_props.mesh_approximation_name]
        assert UsdPhysics.MeshCollisionAPI(mesh_prim).GetApproximationAttr().Get() == expected_token
