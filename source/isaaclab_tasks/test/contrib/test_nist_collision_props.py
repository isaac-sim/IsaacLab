# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Factory task's collision configuration authors what its legacy collision cfgs authored.

The task is loaded through its gym entry point with the Newton physics preset, the only preset that
carried legacy collision cfgs (the PhysX presets set no mesh-collision properties). Its robot, fixed, and held
assets are spawned kitless onto an in-memory stage from small stand-in assets whose colliders mirror
the real Factory assets (mesh colliders plus a sphere collider on the robot, an SDF-cooked mesh
collider on the assembly parts), and compared prim by prim against the legacy collision cfgs.
"""

import warnings

import pytest

from pxr import Usd, UsdGeom

pytestmark = [pytest.mark.unit, pytest.mark.kitless]

_ROBOT = """#usda 1.0
(
    defaultPrim = "panda"
)

def Xform "panda"
{
    def Xform "panda_link0"
    {
        def Mesh "collisions" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {
            uniform token physics:approximation = "convexHull"
        }
    }

    def Xform "force_sensor"
    {
        def Sphere "collisions" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
        }
    }
}
"""

_PART = """#usda 1.0
(
    defaultPrim = "part"
)

def Xform "part" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    def Mesh "collisions" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxSDFMeshCollisionAPI"]
    )
    {
        uniform token physics:approximation = "sdf"
    }
}
"""

_SDF = dict(sdf_max_resolution=256, sdf_narrow_band_inner=-0.005, sdf_narrow_band_outer=0.005)


def _legacy_collision_props(name: str):
    """The Newton-preset collision cfgs the Factory task used before the mesh-collision slot existed."""
    import isaaclab_newton.sim.schemas as newton_schemas

    import isaaclab.sim as sim_utils

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        if name == "robot":
            return sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
                mesh_collision_property=newton_schemas.NewtonMeshCollisionPropertiesCfg(
                    mesh_approximation_name="convexHull"
                ),
            )
        if name == "fixed_asset":
            return newton_schemas.NewtonSDFCollisionPropertiesCfg(rest_offset=0.0, contact_gap=0.005, **_SDF)
        return newton_schemas.NewtonSDFCollisionPropertiesCfg(
            contact_offset=0.0025, rest_offset=0.0, contact_gap=0.005, **_SDF
        )


def _authored(stage, root: str) -> dict:
    """Map each prim below ``root`` to its authored API schemas (incl. unregistered tokens) and attributes."""
    out = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        listop = prim.GetMetadata("apiSchemas")
        schemas = sorted(listop.GetAddedOrExplicitItems()) if listop else []
        attrs = {a.GetName(): a.Get() for a in prim.GetAttributes() if a.IsAuthored()}
        out[prim.GetPath().pathString.removeprefix(root)] = (schemas, attrs)
    return out


@pytest.fixture
def assets(tmp_path):
    """Stand-in assets keyed by the file name of the Factory asset they replace."""
    robot, part = tmp_path / "robot.usda", tmp_path / "part.usda"
    robot.write_text(_ROBOT)
    part.write_text(_PART)
    return {"franka_mimic.usd": str(robot), "bolt_m16.usd": str(part), "nut_m16.usd": str(part)}


def test_factory_newton_collision_props_match_the_legacy_cfgs(assets):
    """The Newton preset of the Factory task authors the same colliders as its legacy collision cfgs."""
    from isaaclab.sim.utils import stage as stage_utils

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    cfg = parse_env_cfg("IsaacContrib-Factory-Franka", overrides=["physics=newton_mjwarp"])
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    for name in ("robot", "fixed_asset", "held_asset"):
        spawn = getattr(cfg.scene, name).spawn
        spawn = spawn.replace(usd_path=assets[spawn.usd_path.rsplit("/", 1)[-1]], visual_material=None)
        legacy = spawn.replace(collision_props=_legacy_collision_props(name), mesh_collision_props=None)
        with stage_utils.use_stage(stage), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spawn.func(f"/World/{name}", spawn)
        assert not [w for w in caught if "is deprecated" in str(w.message)], "the task still uses legacy cfgs"
        with stage_utils.use_stage(stage), warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy.func(f"/World/{name}_legacy", legacy)

        assert _authored(stage, f"/World/{name}") == _authored(stage, f"/World/{name}_legacy")
