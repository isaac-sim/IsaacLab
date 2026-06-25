# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest

from pxr import PhysxSchema, Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


def _new_sim():
    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    return sim_utils.get_current_stage()


def _make_prim_with_schemas(stage, path, schema_tokens):
    """Define an Xform and stamp ``apiSchemas`` metadata with the given multi-instance tokens."""
    UsdGeom.Xform.Define(stage, path)
    prim = stage.GetPrimAtPath(path)
    token_op = Sdf.TokenListOp()
    token_op.explicitItems = schema_tokens
    prim.SetMetadata("apiSchemas", token_op)
    return prim


def _make_xform(stage, path="/World/Tendon"):
    UsdGeom.Xform.Define(stage, path)
    return stage.GetPrimAtPath(path)


def _make_fixed_tendon_prim(stage, path, instance="default"):
    """Create a prim with a multi-instance PhysxTendonAxisRootAPI applied."""
    prim = _make_xform(stage, path)
    PhysxSchema.PhysxTendonAxisRootAPI.Apply(prim, instance)
    return prim


def _make_spatial_tendon_prim(stage, path, instance="default"):
    """Create a prim with a multi-instance PhysxTendonAttachmentRootAPI applied."""
    prim = _make_xform(stage, path)
    PhysxSchema.PhysxTendonAttachmentRootAPI.Apply(prim, instance)
    return prim


def _tendon_attr_prefix(prim, schema_substr):
    """Return the applied-schema name used by the writer as the authored-attribute prefix.

    The legacy writer authors ``f"{schema_name}:{camelCase(field)}"`` where ``schema_name`` is
    the entry returned by ``prim.GetAppliedSchemas()`` (e.g. ``PhysxTendonAxisRootAPI:t0``).
    """
    for schema_name in prim.GetAppliedSchemas():
        if schema_substr in schema_name:
            return schema_name
    raise AssertionError(f"no applied schema containing {schema_substr!r} on {prim.GetPath()}")


# -------------------------------------------------------------------------------------
# Fixed-tendon marker + metadata defaults
# -------------------------------------------------------------------------------------


def test_fixed_tendon_fragment_metadata_defaults():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import FixedTendonFragment, SchemaFragment

    cfg = PhysxFixedTendonCfg(stiffness=1.0)
    assert isinstance(cfg, FixedTendonFragment) and isinstance(cfg, SchemaFragment)
    assert cfg.func == "isaaclab_physx.sim.schemas:apply_fixed_tendon"
    assert cfg.stiffness == 1.0 and cfg.damping is None


def test_spatial_tendon_fragment_metadata_defaults():
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg

    from isaaclab.sim.schemas import SchemaFragment, SpatialTendonFragment

    cfg = PhysxSpatialTendonCfg(stiffness=2.0)
    assert isinstance(cfg, SpatialTendonFragment) and isinstance(cfg, SchemaFragment)
    assert cfg.func == "isaaclab_physx.sim.schemas:apply_spatial_tendon"
    assert cfg.stiffness == 2.0 and cfg.damping is None


# -------------------------------------------------------------------------------------
# PhysxFixedTendonCfg writes the multi-instance namespace
# -------------------------------------------------------------------------------------


def test_physx_fixed_tendon_fragment_writes_instanced_namespace():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg, apply_fixed_tendon

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_fixed_tendon_prim(stage, "/World/FT", instance="t0")
    apply_fixed_tendon(PhysxFixedTendonCfg(stiffness=3.0, damping=0.5), "/World/FT", stage)
    prefix = _tendon_attr_prefix(prim, "PhysxTendonAxisRootAPI")
    assert abs(prim.GetAttribute(f"{prefix}:stiffness").Get() - 3.0) < 1e-6
    assert abs(prim.GetAttribute(f"{prefix}:damping").Get() - 0.5) < 1e-6
    # the ``func`` plumbing field must not be authored as an attribute
    assert not prim.HasAttribute(f"{prefix}:func")


# -------------------------------------------------------------------------------------
# PhysxSpatialTendonCfg writes the multi-instance namespace
# -------------------------------------------------------------------------------------


def test_apply_fixed_tendon_writes_all_instances():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg, apply_fixed_tendon

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_prim_with_schemas(stage, "/World/FTmulti", ["PhysxTendonAxisRootAPI:t0", "PhysxTendonAxisRootAPI:t1"])
    assert apply_fixed_tendon(PhysxFixedTendonCfg(stiffness=9.0), "/World/FTmulti", stage) is True
    for inst in ("t0", "t1"):
        assert abs(prim.GetAttribute(f"PhysxTendonAxisRootAPI:{inst}:stiffness").Get() - 9.0) < 1e-6


def test_apply_fixed_tendon_descends_to_child_prims():
    # tendon schemas are authored on child joint prims, not the articulation root the spawner
    # targets; applying at the root must descend to every descendant carrying the schema.
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg, apply_fixed_tendon

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    UsdGeom.Xform.Define(stage, "/World/Robot")  # root: no tendon schema
    child = _make_fixed_tendon_prim(stage, "/World/Robot/joint", instance="t0")  # child joint carries it
    # apply at the ROOT, not the joint
    assert apply_fixed_tendon(PhysxFixedTendonCfg(stiffness=8.0), "/World/Robot", stage) is True
    prefix = _tendon_attr_prefix(child, "PhysxTendonAxisRootAPI")
    assert abs(child.GetAttribute(f"{prefix}:stiffness").Get() - 8.0) < 1e-6


def test_apply_spatial_tendon_descends_to_child_prims():
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg, apply_spatial_tendon

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    UsdGeom.Xform.Define(stage, "/World/Robot2")  # root: no tendon schema
    child = _make_prim_with_schemas(stage, "/World/Robot2/joint", ["PhysxTendonAttachmentRootAPI:s0"])
    assert apply_spatial_tendon(PhysxSpatialTendonCfg(stiffness=5.0), "/World/Robot2", stage) is True
    assert abs(child.GetAttribute("PhysxTendonAttachmentRootAPI:s0:stiffness").Get() - 5.0) < 1e-6


def test_physx_spatial_tendon_fragment_writes_instanced_namespace():
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg, apply_spatial_tendon

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_spatial_tendon_prim(stage, "/World/ST", instance="s0")
    apply_spatial_tendon(PhysxSpatialTendonCfg(stiffness=4.0, limit_stiffness=0.25), "/World/ST", stage)
    prefix = _tendon_attr_prefix(prim, "PhysxTendonAttachmentRootAPI")
    assert abs(prim.GetAttribute(f"{prefix}:stiffness").Get() - 4.0) < 1e-6
    assert abs(prim.GetAttribute(f"{prefix}:limitStiffness").Get() - 0.25) < 1e-6
    assert not prim.HasAttribute(f"{prefix}:func")


def test_apply_spatial_tendon_writes_all_instances():
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg, apply_spatial_tendon

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_prim_with_schemas(
        stage,
        "/World/STmulti",
        ["PhysxTendonAttachmentRootAPI:r0", "PhysxTendonAttachmentLeafAPI:l0"],
    )
    assert apply_spatial_tendon(PhysxSpatialTendonCfg(stiffness=4.0), "/World/STmulti", stage) is True
    assert abs(prim.GetAttribute("PhysxTendonAttachmentRootAPI:r0:stiffness").Get() - 4.0) < 1e-6
    assert abs(prim.GetAttribute("PhysxTendonAttachmentLeafAPI:l0:stiffness").Get() - 4.0) < 1e-6


# -------------------------------------------------------------------------------------
# apply_fixed_tendon_properties dispatch (tune-not-apply, multi-fragment)
# -------------------------------------------------------------------------------------


def test_apply_fixed_tendon_properties_dispatches_fragments():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import apply_fixed_tendon_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_fixed_tendon_prim(stage, "/World/FT2", instance="t0")
    apply_fixed_tendon_properties(
        "/World/FT2",
        [PhysxFixedTendonCfg(stiffness=5.0), PhysxFixedTendonCfg(damping=0.75)],
        stage,
    )
    prefix = _tendon_attr_prefix(prim, "PhysxTendonAxisRootAPI")
    assert abs(prim.GetAttribute(f"{prefix}:stiffness").Get() - 5.0) < 1e-6
    assert abs(prim.GetAttribute(f"{prefix}:damping").Get() - 0.75) < 1e-6


def test_apply_spatial_tendon_properties_dispatches_fragments():
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg

    from isaaclab.sim.schemas import apply_spatial_tendon_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_spatial_tendon_prim(stage, "/World/ST2", instance="s0")
    apply_spatial_tendon_properties(
        "/World/ST2",
        [PhysxSpatialTendonCfg(stiffness=6.0), PhysxSpatialTendonCfg(offset=0.1)],
        stage,
    )
    prefix = _tendon_attr_prefix(prim, "PhysxTendonAttachmentRootAPI")
    assert abs(prim.GetAttribute(f"{prefix}:stiffness").Get() - 6.0) < 1e-6
    assert abs(prim.GetAttribute(f"{prefix}:offset").Get() - 0.1) < 1e-6


# -------------------------------------------------------------------------------------
# Public imports
# -------------------------------------------------------------------------------------


def test_public_imports():
    from isaaclab_physx.sim.schemas import (  # noqa: F401
        PhysxFixedTendonCfg,
        PhysxSpatialTendonCfg,
        apply_fixed_tendon,
        apply_spatial_tendon,
    )

    from isaaclab.sim.schemas import (  # noqa: F401
        FixedTendonFragment,
        SpatialTendonFragment,
        apply_fixed_tendon_properties,
        apply_spatial_tendon_properties,
    )


# -------------------------------------------------------------------------------------
# core writer parity: invalid-prim guard + aggregated return
# -------------------------------------------------------------------------------------


def test_apply_fixed_tendon_raises_on_invalid_prim():
    from isaaclab.sim.schemas import apply_fixed_tendon_properties

    _new_sim()
    stage = sim_utils.get_current_stage()
    with pytest.raises(ValueError):
        apply_fixed_tendon_properties("/World/DoesNotExist", [], stage)


def test_apply_spatial_tendon_raises_on_invalid_prim():
    from isaaclab.sim.schemas import apply_spatial_tendon_properties

    _new_sim()
    stage = sim_utils.get_current_stage()
    with pytest.raises(ValueError):
        apply_spatial_tendon_properties("/World/DoesNotExist", [], stage)


def test_apply_fixed_tendon_aggregates_fragment_results():
    from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg, apply_fixed_tendon_properties

    stage = _new_sim()
    _make_prim_with_schemas(stage, "/World/Agg", ["PhysxTendonAxisRootAPI:inst0"])

    # a fragment whose applier reports failure makes the aggregate False
    failing = UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)
    failing.func = lambda cfg, prim_path, stage=None: False
    assert apply_fixed_tendon_properties("/World/Agg", [failing], stage) is False

    ok = UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)
    ok.func = lambda cfg, prim_path, stage=None: True
    assert apply_fixed_tendon_properties("/World/Agg", [ok], stage) is True


def test_apply_fixed_tendon_raises_on_invalid_prim_backend():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg, apply_fixed_tendon

    _new_sim()
    stage = sim_utils.get_current_stage()
    with pytest.raises(ValueError):
        apply_fixed_tendon(PhysxFixedTendonCfg(stiffness=1.0), "/World/DoesNotExist", stage)


def test_apply_mujoco_fixed_tendon_raises_on_invalid_prim():
    from isaaclab_newton.sim.schemas import MujocoFixedTendonCfg, apply_mujoco_fixed_tendon

    _new_sim()
    stage = sim_utils.get_current_stage()
    with pytest.raises(ValueError):
        apply_mujoco_fixed_tendon(MujocoFixedTendonCfg(stiffness=1.0), "/World/DoesNotExist", stage)


# -------------------------------------------------------------------------------------
# MujocoFixedTendonCfg — Newton fragment for the mjc: namespace
# -------------------------------------------------------------------------------------


def test_mujoco_fixed_tendon_metadata():
    from isaaclab_newton.sim.schemas import MujocoFixedTendonCfg

    from isaaclab.sim.schemas import FixedTendonFragment

    cfg = MujocoFixedTendonCfg(stiffness=2.0)
    assert isinstance(cfg, FixedTendonFragment)
    # not namespace-driven: the custom applier writes mjc:* itself, so _usd_namespace stays None
    assert type(cfg)._usd_namespace is None
    assert cfg.func == "isaaclab_newton.sim.schemas:apply_mujoco_fixed_tendon"
    assert not hasattr(cfg, "rest_length") and not hasattr(cfg, "limit_stiffness")


def test_apply_mujoco_fixed_tendon_writes_mjc_namespace():
    from isaaclab_newton.sim.schemas import MujocoFixedTendonCfg, apply_mujoco_fixed_tendon

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    stage.DefinePrim("/World/MjcT", "MjcTendon")
    assert apply_mujoco_fixed_tendon(MujocoFixedTendonCfg(stiffness=2.0, damping=0.25), "/World/MjcT", stage) is True
    prim = stage.GetPrimAtPath("/World/MjcT")
    assert abs(prim.GetAttribute("mjc:stiffness").Get() - 2.0) < 1e-6
    assert abs(prim.GetAttribute("mjc:damping").Get() - 0.25) < 1e-6
    assert not prim.HasAttribute("mjc:func")


def test_apply_mujoco_fixed_tendon_returns_false_on_non_mjc_prim():
    from isaaclab_newton.sim.schemas import MujocoFixedTendonCfg, apply_mujoco_fixed_tendon

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    UsdGeom.Xform.Define(stage, "/World/NotMjc")
    assert apply_mujoco_fixed_tendon(MujocoFixedTendonCfg(stiffness=2.0), "/World/NotMjc", stage) is False
    prim = stage.GetPrimAtPath("/World/NotMjc")
    assert not prim.HasAttribute("mjc:stiffness")


# -------------------------------------------------------------------------------------
# legacy-vs-fragment equivalence (the fragment API must be a behavioral no-op swap)
# -------------------------------------------------------------------------------------


def test_legacy_and_fragment_fixed_tendon_produce_identical_attrs():
    """The fragment API must author the same tendon attributes as the legacy writer.

    Verified end-to-end on the Shadow Hand (the real tendon user,
    ``FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1)``); replicated here on a synthetic
    root + descendant-joint structure so it runs deterministically without asset-server access. Also
    exercises the descend-to-child-prims behavior, since the schemas live on descendants of the
    applied prim path (as they do on a real articulation).
    """
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg, PhysxFixedTendonPropertiesCfg

    from isaaclab.sim.schemas import apply_fixed_tendon_properties, modify_fixed_tendon_properties

    stage = _new_sim()

    def _build(root):
        # tendon schemas on descendant joints (multi-instance), mirroring the Shadow Hand layout
        UsdGeom.Xform.Define(stage, root)
        _make_prim_with_schemas(stage, f"{root}/J0", ["PhysxTendonAxisRootAPI:t0", "PhysxTendonAxisRootAPI:t1"])
        _make_prim_with_schemas(stage, f"{root}/nested/J1", ["PhysxTendonAxisRootAPI:t0"])

    _build("/World/legacy")
    _build("/World/fragment")

    # apply each path at the ROOT; both must descend to the child joints
    modify_fixed_tendon_properties("/World/legacy", PhysxFixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1))
    apply_fixed_tendon_properties("/World/fragment", [PhysxFixedTendonCfg(limit_stiffness=30.0, damping=0.1)])

    def _collect(root):
        attrs = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            for schema_name in prim.GetAppliedSchemas():
                if "PhysxTendonAxisRootAPI" not in schema_name:
                    continue
                for suffix in ("limitStiffness", "damping"):
                    attr = prim.GetAttribute(f"{schema_name}:{suffix}")
                    if attr and attr.HasAuthoredValue():
                        rel = prim.GetPath().pathString[len(root) :]  # key relative to root so paths compare
                        attrs[f"{rel}|{schema_name}:{suffix}"] = attr.Get()
        return attrs

    legacy = _collect("/World/legacy")
    fragment = _collect("/World/fragment")

    assert legacy, "legacy writer authored no tendon attributes (test would be vacuous)"
    assert legacy.keys() == fragment.keys()
    for key, value in legacy.items():
        assert abs(fragment[key] - value) < 1e-6


def test_spawn_from_file_with_empty_tendon_lists_is_noop(tmp_path):
    # an empty tendon list is type-valid for the slot; the spawner shim must route it through the
    # fragment path (a no-op) rather than handing [] to the legacy modify_*_tendon_properties writer.
    asset = tmp_path / "mini.usda"
    src = Usd.Stage.CreateNew(str(asset))
    UsdGeom.Xform.Define(src, "/Root")
    src.SetDefaultPrim(src.GetPrimAtPath("/Root"))
    src.GetRootLayer().Save()
    del src

    _new_sim()
    cfg = sim_utils.UsdFileCfg(usd_path=str(asset), fixed_tendons_props=[], spatial_tendons_props=[])
    cfg.func("/World/Asset", cfg)  # must not raise
    assert sim_utils.get_current_stage().GetPrimAtPath("/World/Asset").IsValid()
