# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from isaaclab_newton.physics import MJWarpSolverCfg

from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg


def test_g1_rough_newton_has_sufficient_constraint_capacity():
    env_cfg = G1RoughEnvCfg()

    solver_cfg = env_cfg.sim.physics.newton_mjwarp.solver_cfg

    assert isinstance(solver_cfg, MJWarpSolverCfg)
    assert solver_cfg.njmax == 300


@pytest.mark.parametrize("robot", ["anymal_d", "cassie", "g1", "go2", "h1"])
@pytest.mark.parametrize("backend", ["newton_mjwarp", "newton_kamino", "ovphysx", "isaacsim_physx"])
def test_velocity_fixed_material_is_authored_before_initialization(robot, backend, tmp_path):
    """All presets bind the same standard USD friction values even on referenced instances."""
    import importlib

    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from isaaclab.sim.utils.stage import use_stage

    from isaaclab_tasks.utils.hydra import resolve_presets

    names = {
        "anymal_d": "AnymalDRoughEnvCfg",
        "cassie": "CassieRoughEnvCfg",
        "g1": "G1RoughEnvCfg",
        "go2": "UnitreeGo2RoughEnvCfg",
        "h1": "H1RoughEnvCfg",
    }
    module = importlib.import_module(f"isaaclab_tasks.core.velocity.config.{robot}.rough_env_cfg")
    cfg = resolve_presets(getattr(module, names[robot])(), selected=[backend])
    assert not hasattr(cfg.events, "physics_material")

    # Keep real preset/spawner behavior but replace the remote geometry with a referenced instance.
    geometry = Usd.Stage.CreateNew(str(tmp_path / "geometry.usda"))
    geometry.SetDefaultPrim(UsdGeom.Xform.Define(geometry, "/Geometry").GetPrim())
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(geometry, "/Geometry/Collision").GetPrim())
    geometry.GetRootLayer().Save()
    source = Usd.Stage.CreateNew(str(tmp_path / "robot.usda"))
    root = UsdGeom.Xform.Define(source, "/Robot").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    source.SetDefaultPrim(root)
    instance = UsdGeom.Xform.Define(source, "/Robot/Geometry").GetPrim()
    instance.GetReferences().AddReference(str(tmp_path / "geometry.usda"))
    instance.SetInstanceable(True)
    source.GetRootLayer().Save()
    spawn_cfg = cfg.scene.robot.spawn.replace(usd_path=str(tmp_path / "robot.usda"))
    stage = Usd.Stage.CreateInMemory()
    with use_stage(stage):
        spawn_from_usd("/Robot", spawn_cfg)
    assert stage.GetPrimAtPath("/Robot/Geometry").IsInstance() == (not spawn_cfg.make_uninstanceable)
    collider = stage.GetPrimAtPath("/Robot/Geometry/Collision")
    material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
    assert material
    physics = UsdPhysics.MaterialAPI(material.GetPrim())
    assert physics.GetStaticFrictionAttr().Get() == pytest.approx(0.8)
    assert physics.GetDynamicFrictionAttr().Get() == pytest.approx(0.6)
    assert physics.GetRestitutionAttr().Get() == 0
    if backend in {"isaacsim_physx", "ovphysx"}:
        assert material.GetPrim().GetAttribute("physxMaterial:frictionCombineMode").Get() == "multiply"
        assert material.GetPrim().GetAttribute("physxMaterial:restitutionCombineMode").Get() == "multiply"
    if backend.startswith("newton"):
        import newton

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        assert builder.shape_material_mu == pytest.approx([0.6])
