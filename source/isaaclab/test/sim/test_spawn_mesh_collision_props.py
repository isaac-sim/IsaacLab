# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Routing of the ``mesh_collision_props`` spawner slot on the USD, shape, and mesh spawners, and the
accuracy of the deprecation guidance that points users at spawner slots.

These tests author on an in-memory USD stage and do not launch Isaac Sim / Kit.
"""

import dataclasses
import importlib
import inspect
import logging
import pkgutil
import re
import warnings

import pytest

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import stage as stage_utils

pytestmark = [pytest.mark.unit, pytest.mark.kitless]

_ASSET = """#usda 1.0
(
    defaultPrim = "Robot"
)

def Xform "Robot"
{
    def Xform "link0"
    {
        def Mesh "collisions" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        )
        {
            uniform token physics:approximation = "convexHull"
        }

        def Mesh "visuals"
        {
        }
    }

    def Xform "link1"
    {
        def Sphere "collisions" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
        }
    }
}
"""


@pytest.fixture
def asset_path(tmp_path):
    """A two-link asset: a mesh collider and a visual mesh under ``link0``, a sphere collider under ``link1``."""
    path = tmp_path / "robot.usda"
    path.write_text(_ASSET)
    return str(path)


@pytest.fixture
def stage():
    """An in-memory stage made current for the spawners."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    with stage_utils.use_stage(stage):
        yield stage


def _approximation(stage, path: str) -> str | None:
    """Return the authored ``physics:approximation`` of a prim, or None when it has no ``MeshCollisionAPI``."""
    prim = stage.GetPrimAtPath(path)
    if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        return None
    return UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()


def _authored(stage, root: str) -> dict:
    """Map each prim below ``root`` (relative path) to its applied API schemas and authored attributes."""
    out = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        attrs = {a.GetName(): a.Get() for a in prim.GetAttributes() if a.IsAuthored()}
        out[prim.GetPath().pathString.removeprefix(root)] = (list(prim.GetAppliedSchemas()), attrs)
    return out


"""
USD file spawners.
"""


def test_usd_file_bare_fragment_reaches_every_collider(stage, asset_path):
    """The shorthand targets every collider under the asset and leaves non-colliders untouched."""
    cfg = sim_utils.UsdFileCfg(
        usd_path=asset_path,
        mesh_collision_props=sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexDecomposition"),
    )
    cfg.func("/World/Robot", cfg)

    assert _approximation(stage, "/World/Robot/link0/collisions") == "convexDecomposition"
    assert _approximation(stage, "/World/Robot/link1/collisions") == "convexDecomposition"
    assert _approximation(stage, "/World/Robot/link0/visuals") is None
    assert _approximation(stage, "/World/Robot") is None


def test_usd_file_mapping_narrows_the_colliders(stage, asset_path):
    """A mapping key anchored at the spawn prim selects which colliders are tuned."""
    cfg = sim_utils.UsdFileCfg(
        usd_path=asset_path,
        mesh_collision_props={
            "/link1/.*": [sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="boundingSphere")]
        },
    )
    cfg.func("/World/Robot", cfg)

    assert _approximation(stage, "/World/Robot/link0/collisions") == "convexHull"
    assert _approximation(stage, "/World/Robot/link1/collisions") == "boundingSphere"


def test_usd_file_pattern_without_colliders_warns_and_authors_nothing(stage, asset_path, caplog):
    """A pattern that matches only non-colliders logs a warning instead of creating ``MeshCollisionAPI``."""
    cfg = sim_utils.UsdFileCfg(
        usd_path=asset_path,
        mesh_collision_props={"/link0/visuals": [sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none")]},
    )
    with caplog.at_level(logging.WARNING):
        cfg.func("/World/Robot", cfg)

    assert _approximation(stage, "/World/Robot/link0/visuals") is None
    assert "No mesh-collision targets" in caplog.text


@pytest.mark.parametrize(
    "make_legacy_mesh_cfg,make_fragments",
    [
        (
            lambda: sim_utils.schemas.MeshCollisionBaseCfg(mesh_approximation_name="convexHull"),
            lambda: sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull"),
        ),
        (
            lambda: sim_utils.schemas.BoundingSpherePropertiesCfg(),
            lambda: [sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="boundingSphere")],
        ),
    ],
    ids=["convex-hull", "bounding-sphere"],
)
def test_usd_file_slot_matches_the_legacy_nested_field(stage, asset_path, make_legacy_mesh_cfg, make_fragments):
    """The slot authors exactly what the legacy ``CollisionBaseCfg.mesh_collision_property`` authored."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = sim_utils.UsdFileCfg(
            usd_path=asset_path,
            collision_props=sim_utils.CollisionBaseCfg(mesh_collision_property=make_legacy_mesh_cfg()),
        )
        legacy.func("/World/Legacy", legacy)
    fragment = sim_utils.UsdFileCfg(usd_path=asset_path, mesh_collision_props=make_fragments())
    fragment.func("/World/Fragment", fragment)
    # negative control: the comparison must see a missing slot
    untouched = sim_utils.UsdFileCfg(usd_path=asset_path)
    untouched.func("/World/Untouched", untouched)

    assert _authored(stage, "/World/Fragment") == _authored(stage, "/World/Legacy")
    assert _authored(stage, "/World/Untouched") != _authored(stage, "/World/Legacy")


def test_usd_file_slot_accepts_a_legacy_mesh_collision_cfg(stage, asset_path):
    """A legacy mesh-collision cfg in the slot routes to the legacy writer on every collider."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = sim_utils.UsdFileCfg(
            usd_path=asset_path,
            mesh_collision_props=sim_utils.schemas.MeshCollisionBaseCfg(mesh_approximation_name="boundingCube"),
        )
        cfg.func("/World/Robot", cfg)

    assert _approximation(stage, "/World/Robot/link0/collisions") == "boundingCube"
    assert _approximation(stage, "/World/Robot/link1/collisions") == "boundingCube"
    assert _approximation(stage, "/World/Robot/link0/visuals") is None


"""
Shape and mesh spawners.
"""


def test_shape_slot_targets_the_geometry_collider(stage):
    """On a shape spawner the slot authors on the collider geometry prim."""
    cfg = sim_utils.CuboidCfg(
        size=(0.1, 0.1, 0.1),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        mesh_collision_props=sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull"),
    )
    cfg.func("/World/Cube", cfg)

    assert _approximation(stage, "/World/Cube/geometry/mesh") == "convexHull"
    assert _approximation(stage, "/World/Cube") is None


def test_shape_slot_without_collider_authors_nothing(stage):
    """Without ``collision_props`` the geometry is not a collider, so the slot has no target."""
    cfg = sim_utils.CuboidCfg(
        size=(0.1, 0.1, 0.1),
        mesh_collision_props=sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull"),
    )
    cfg.func("/World/Cube", cfg)

    assert _approximation(stage, "/World/Cube/geometry/mesh") is None


def test_mesh_slot_overrides_the_default_approximation(stage):
    """On a mesh spawner the slot overrides the approximation the spawner picks for the shape."""
    default = sim_utils.MeshSphereCfg(radius=0.1, collision_props=sim_utils.UsdPhysicsCollisionCfg())
    default.func("/World/Default", default)
    tuned = sim_utils.MeshSphereCfg(
        radius=0.1,
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        mesh_collision_props={"": [sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull")]},
    )
    tuned.func("/World/Tuned", tuned)

    assert _approximation(stage, "/World/Default/geometry/mesh") == "boundingSphere"
    assert _approximation(stage, "/World/Tuned/geometry/mesh") == "convexHull"


def test_mesh_slot_is_rejected_for_deformable_bodies(stage):
    """Deformable bodies collide through their simulation mesh, so the slot is rejected."""
    cfg = sim_utils.MeshCuboidCfg(
        size=(0.1, 0.1, 0.1),
        volume_deformable_props=sim_utils.schemas.OmniPhysicsDeformableBodyCfg(),
        mesh_collision_props=sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull"),
    )
    with pytest.raises(ValueError, match="mesh_collision_props"):
        cfg.func("/World/Deformable", cfg)


"""
Deprecation guidance.
"""

# Rigid-body families are tuned on every rigid-object spawner; the articulation, joint, tendon, and
# deformable families only on the file spawners.
_RIGID_FAMILY_BASES = ("CollisionBaseCfg", "MeshCollisionBaseCfg", "RigidBodyBaseCfg", "MassPropertiesCfg")
_SLOT_PATTERN = re.compile(r"`{0,2}\b([a-z_]+_props)\b`{0,2}\s+slot")


def _spawner_cfgs(owner: type) -> list[type]:
    """The slot-owning base spawner cfg and every subclass of it defined in :mod:`isaaclab.sim.spawners`."""
    import isaaclab.sim.spawners as spawners

    for module in pkgutil.walk_packages(spawners.__path__, spawners.__name__ + "."):
        if module.name.endswith("_cfg"):
            importlib.import_module(module.name)
    found, pending = [owner], [owner]
    while pending:
        for subclass in pending.pop().__subclasses__():
            if subclass not in found:
                found.append(subclass)
                pending.append(subclass)
    return found


def _deprecated_schema_cfgs() -> list[type]:
    """Every deprecated schema cfg class of the core, PhysX, and Newton schema modules."""
    modules = [importlib.import_module("isaaclab.sim.schemas.schemas_cfg")]
    for name in ("isaaclab_physx.sim.schemas.schemas_cfg", "isaaclab_newton.sim.schemas.schemas_cfg"):
        modules.append(pytest.importorskip(name))
    return [
        cls
        for module in modules
        for cls in vars(module).values()
        if inspect.isclass(cls) and cls.__module__ == module.__name__ and ".. deprecated::" in (cls.__doc__ or "")
    ]


def test_deprecation_guidance_names_only_existing_spawner_slots():
    """Every spawner slot a deprecated schema cfg points to (warning or docstring) exists on its spawners."""
    from isaaclab.sim.spawners.from_files.from_files_cfg import FileCfg
    from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg

    rigid_spawners, file_spawners = _spawner_cfgs(RigidObjectSpawnerCfg), _spawner_cfgs(FileCfg)
    assert sim_utils.CuboidCfg in rigid_spawners and sim_utils.UrdfFileCfg in file_spawners
    checked = 0
    missing = []
    for cls in _deprecated_schema_cfgs():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cls()
        messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        guidance = " ".join(messages) + " " + (cls.__doc__ or "").split(".. deprecated::", 1)[1]
        rigid_family = any(base.__name__ in _RIGID_FAMILY_BASES for base in cls.__mro__)
        spawners = rigid_spawners if rigid_family else file_spawners
        for slot in set(_SLOT_PATTERN.findall(guidance)):
            checked += 1
            for spawner in spawners:
                if slot not in {field.name for field in dataclasses.fields(spawner)}:
                    missing.append(f"{cls.__name__} -> {spawner.__name__}.{slot}")
    assert checked, "no spawner slot was named in any deprecation guidance; the pattern is stale"
    assert not missing, "deprecation guidance names spawner slots that do not exist:\n" + "\n".join(missing)
