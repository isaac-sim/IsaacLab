# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the assemble trocar task configuration."""

import subprocess
import sys

import numpy as np
from isaaclab_newton.sim.schemas import MujocoCollisionCfg
from isaaclab_physx.sim.schemas import PhysxCollisionCfg

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_tasks.contrib.assemble_trocar.config.assemble_trocar_assets import (
    _bake_negative_mesh_collider_scales,
    _expand_merged_mesh_colliders,
    spawn_usd_with_newton_compatible_colliders,
)
from isaaclab_tasks.contrib.assemble_trocar.g129_dex3_env_cfg import G1AssembleTrocarEnvCfg


def test_task_config_import_does_not_preload_pxr():
    """RLinf must be able to parse the task config before starting SimulationApp."""
    code = """
import sys
import isaaclab_tasks.contrib.assemble_trocar.g129_dex3_env_cfg
assert not any(name == "pxr" or name.startswith("pxr.") for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _define_triangle(stage: Usd.Stage, path: str) -> UsdGeom.Mesh:
    """Define one triangle mesh for USD adaptation tests."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    return mesh


def test_task_scopes_newton_compatible_colliders_to_robot_and_tray():
    """Only assets requiring compatibility edits use the task-local spawner."""
    cfg = G1AssembleTrocarEnvCfg()

    assert cfg.scene.robot.spawn.func is spawn_usd_with_newton_compatible_colliders
    assert cfg.scene.tray.spawn.func is spawn_usd_with_newton_compatible_colliders
    assert cfg.scene.trocar_1.spawn.func is not spawn_usd_with_newton_compatible_colliders
    assert cfg.scene.trocar_2.spawn.func is not spawn_usd_with_newton_compatible_colliders


def test_expand_merged_mesh_collider():
    """An Xform mesh-merge collider becomes a regular mesh collider."""
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Asset")
    merged = UsdGeom.Xform.Define(stage, "/Asset/Collision").GetPrim()
    UsdPhysics.CollisionAPI.Apply(merged).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(merged).CreateApproximationAttr("convexHull")
    mesh = _define_triangle(stage, "/Asset/Collision/Mesh").GetPrim()

    assert _expand_merged_mesh_colliders(stage, root.GetPath()) == 1
    assert not merged.HasAPI(UsdPhysics.CollisionAPI)
    assert mesh.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get()
    assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == "convexHull"


def test_bake_negative_mesh_collider_scale():
    """A mirrored collider preserves its geometry with a positive USD scale."""
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Asset")
    mesh = _define_triangle(stage, "/Asset/Mesh")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    scale_op = UsdGeom.Xformable(mesh).AddScaleOp()
    scale_op.Set((-1.0, 1.0, 1.0))

    assert _bake_negative_mesh_collider_scales(stage, root.GetPath()) == 1
    np.testing.assert_allclose(UsdGeom.Xformable(mesh).GetOrderedXformOps()[-1].Get(), (1.0, 1.0, 1.0))
    np.testing.assert_allclose(mesh.GetPointsAttr().Get()[0], (-1.0, 2.0, 3.0))
    assert list(mesh.GetFaceVertexIndicesAttr().Get()) == [2, 1, 0]


def test_bore_contact_tuning_preserves_physx_properties():
    """The Newton-only bore override must not replace the existing PhysX contact settings."""
    cfg = G1AssembleTrocarEnvCfg()
    collision_props = cfg.scene.trocar_1.spawn.collision_props

    assert isinstance(collision_props, dict)
    assert any(
        isinstance(fragment, PhysxCollisionCfg) and fragment.contact_offset == 0.001 and fragment.rest_offset == -0.001
        for fragment in collision_props["/.*"]
    )

    bore_patterns = [pattern for pattern in collision_props if pattern != "/.*"]
    assert len(bore_patterns) == 1
    bore_props = collision_props[bore_patterns[0]]
    assert len(bore_props) == 1
    assert isinstance(bore_props[0], MujocoCollisionCfg)
    assert bore_props[0].priority == 1
    assert bore_props[0].solref == (0.002, 1.0)


def test_bore_friction_remap_is_newton_specific():
    """The startup term records the PhysX average-combine equivalent for Newton."""
    cfg = G1AssembleTrocarEnvCfg()
    term = cfg.events.newton_bore_friction

    assert term.mode == "startup"
    assert term.params["asset_cfg"].name == "trocar_1"
    assert term.params["source_mu"] == 1.2
    assert term.params["target_mu"] == 0.65
