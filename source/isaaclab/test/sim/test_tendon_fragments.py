# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import dataclasses

import pytest

from pxr import PhysxSchema, Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

pytestmark = pytest.mark.integration


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


def _make_spatial_tendon_leaf_prim(stage, path, instance="default"):
    """Create a prim with a multi-instance PhysxTendonAttachmentLeafAPI applied."""
    prim = _make_xform(stage, path)
    PhysxSchema.PhysxTendonAttachmentLeafAPI.Apply(prim, instance)
    return prim


def _tendon_attr_prefix(prim, schema_substr):
    """Return the canonical PhysX property prefix for an applied tendon schema.

    Applied schema tokens use ``PhysxTendon...API:<instance>``, while all PhysX tendon
    properties use the schema-declared ``physxTendon:<instance>`` namespace.
    """
    for schema_name in prim.GetAppliedSchemas():
        if schema_substr in schema_name:
            instance_name = schema_name.split(":", maxsplit=1)[1]
            return f"physxTendon:{instance_name}"
    raise AssertionError(f"no applied schema containing {schema_substr!r} on {prim.GetPath()}")


# -------------------------------------------------------------------------------------
# Fixed-tendon marker + metadata defaults
# -------------------------------------------------------------------------------------


def test_fixed_tendon_fragment_metadata_defaults():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import FixedTendonFragment, SchemaFragment

    cfg = PhysxFixedTendonCfg(instance_names=None, stiffness=1.0)
    assert isinstance(cfg, FixedTendonFragment) and isinstance(cfg, SchemaFragment)
    assert type(cfg)._usd_applied_schema == "PhysxTendonAxisRootAPI"
    assert type(cfg)._usd_namespace == "physxTendon"
    assert cfg.func == "isaaclab.sim.schemas:apply_multi_apply"
    assert cfg.stiffness == 1.0 and cfg.damping is None


def test_spatial_tendon_fragment_metadata_defaults():
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg

    from isaaclab.sim.schemas import SchemaFragment, SpatialTendonFragment

    cfg = PhysxSpatialTendonCfg(instance_names=None, stiffness=2.0)
    assert isinstance(cfg, SpatialTendonFragment) and isinstance(cfg, SchemaFragment)
    assert type(cfg)._usd_applied_schema == "PhysxTendonAttachmentRootAPI"
    assert type(cfg)._usd_namespace == "physxTendon"
    assert cfg.func == "isaaclab.sim.schemas:apply_multi_apply"
    assert cfg.stiffness == 2.0 and cfg.damping is None


# -------------------------------------------------------------------------------------
# PhysxFixedTendonCfg writes the multi-instance namespace
# -------------------------------------------------------------------------------------


def test_physx_fixed_tendon_fragment_writes_instanced_namespace():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import apply_multi_apply

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_fixed_tendon_prim(stage, "/World/FT", instance="t0")
    apply_multi_apply(PhysxFixedTendonCfg(instance_names=None, stiffness=3.0, damping=0.5), "/World/FT", stage)
    prefix = _tendon_attr_prefix(prim, "PhysxTendonAxisRootAPI")
    assert abs(prim.GetAttribute(f"{prefix}:stiffness").Get() - 3.0) < 1e-6
    assert abs(prim.GetAttribute(f"{prefix}:damping").Get() - 0.5) < 1e-6
    # the ``func`` plumbing field must not be authored as an attribute
    assert not prim.HasAttribute(f"{prefix}:func")
    assert not prim.HasAttribute("PhysxTendonAxisRootAPI:t0:stiffness")


# -------------------------------------------------------------------------------------
# PhysxSpatialTendonCfg writes the multi-instance namespace
# -------------------------------------------------------------------------------------


def test_apply_fixed_tendon_writes_all_instances():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import apply_multi_apply

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_prim_with_schemas(stage, "/World/FTmulti", ["PhysxTendonAxisRootAPI:t0", "PhysxTendonAxisRootAPI:t1"])
    assert apply_multi_apply(PhysxFixedTendonCfg(instance_names=None, stiffness=9.0), "/World/FTmulti", stage) is True
    for inst in ("t0", "t1"):
        assert abs(prim.GetAttribute(f"physxTendon:{inst}:stiffness").Get() - 9.0) < 1e-6


def test_apply_fixed_tendon_writes_only_the_selected_instances():
    """Two tendons share a prim, so the instance -- not the prim -- selects which one is tuned."""
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import apply_multi_apply

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_prim_with_schemas(stage, "/World/FTpick", ["PhysxTendonAxisRootAPI:t0", "PhysxTendonAxisRootAPI:t1"])
    assert apply_multi_apply(PhysxFixedTendonCfg(instance_names="t0", stiffness=9.0), "/World/FTpick", stage) is True
    assert abs(prim.GetAttribute("physxTendon:t0:stiffness").Get() - 9.0) < 1e-6
    assert not prim.GetAttribute("physxTendon:t1:stiffness").HasAuthoredValue()


def test_apply_fixed_tendon_skips_a_prim_without_the_named_instance():
    """A prim carrying the schema but not the named instance is passed over, not written or raised on."""
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import apply_multi_apply

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_prim_with_schemas(stage, "/World/FTother", ["PhysxTendonAxisRootAPI:t0"])
    assert apply_multi_apply(PhysxFixedTendonCfg(instance_names="t1", stiffness=9.0), "/World/FTother", stage) is False
    assert not prim.GetAttribute("physxTendon:t0:stiffness").HasAuthoredValue()


def test_apply_fixed_tendon_properties_selects_one_instance_across_a_subtree():
    """Under the spawner's default subtree pattern each joint carries its own tendon.

    Naming one tendon must tune the joint that carries it and pass over the others; a name no joint
    carries tunes nothing, which the writer reports rather than raises.
    """
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import apply_fixed_tendon_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    UsdGeom.Xform.Define(stage, "/World/Hand")
    first = _make_fixed_tendon_prim(stage, "/World/Hand/J0", instance="ff")
    second = _make_fixed_tendon_prim(stage, "/World/Hand/J1", instance="mf")
    selected = [PhysxFixedTendonCfg(instance_names="mf", stiffness=9.0)]
    assert apply_fixed_tendon_properties("/World/Hand(/.*)?", selected, stage) is True
    assert abs(second.GetAttribute("physxTendon:mf:stiffness").Get() - 9.0) < 1e-6
    assert not first.GetAttribute("physxTendon:ff:stiffness").HasAuthoredValue()
    absent = [PhysxFixedTendonCfg(instance_names="typo", stiffness=9.0)]
    assert apply_fixed_tendon_properties("/World/Hand(/.*)?", absent, stage) is False


def test_apply_fixed_tendon_writer_descends_to_child_prims():
    # tendon schemas are authored on child joint prims, not the articulation root the spawner
    # targets. Targeting is owned by the core writer: its subtree expression descends to
    # every descendant carrying the schema, while the backend func is a per-prim tuner that
    # no-ops on a prim without the schema.
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import apply_fixed_tendon_properties, apply_multi_apply

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    UsdGeom.Xform.Define(stage, "/World/Robot")  # root: no tendon schema
    child = _make_fixed_tendon_prim(stage, "/World/Robot/joint", instance="t0")  # child joint carries it
    # the backend func is per-prim: applied at the root it tunes nothing
    assert apply_multi_apply(PhysxFixedTendonCfg(instance_names=None, stiffness=8.0), "/World/Robot", stage) is False
    # the core writer's subtree expression descends from the root to the joint
    assert (
        apply_fixed_tendon_properties(
            "/World/Robot(/.*)?", [PhysxFixedTendonCfg(instance_names=None, stiffness=8.0)], stage
        )
        is True
    )
    prefix = _tendon_attr_prefix(child, "PhysxTendonAxisRootAPI")
    assert abs(child.GetAttribute(f"{prefix}:stiffness").Get() - 8.0) < 1e-6


def test_apply_spatial_tendon_writer_descends_to_child_prims():
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg

    from isaaclab.sim.schemas import apply_multi_apply, apply_spatial_tendon_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    UsdGeom.Xform.Define(stage, "/World/Robot2")  # root: no tendon schema
    child = _make_prim_with_schemas(stage, "/World/Robot2/joint", ["PhysxTendonAttachmentRootAPI:s0"])
    # the backend func is per-prim: applied at the root it tunes nothing
    assert apply_multi_apply(PhysxSpatialTendonCfg(instance_names=None, stiffness=5.0), "/World/Robot2", stage) is False
    # the core writer's subtree expression descends from the root to the joint
    assert (
        apply_spatial_tendon_properties(
            "/World/Robot2(/.*)?", [PhysxSpatialTendonCfg(instance_names=None, stiffness=5.0)], stage
        )
        is True
    )
    assert abs(child.GetAttribute("physxTendon:s0:stiffness").Get() - 5.0) < 1e-6


def test_physx_spatial_tendon_fragment_writes_instanced_namespace():
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg

    from isaaclab.sim.schemas import apply_multi_apply

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_spatial_tendon_prim(stage, "/World/ST", instance="s0")
    apply_multi_apply(
        PhysxSpatialTendonCfg(instance_names=None, stiffness=4.0, limit_stiffness=0.25), "/World/ST", stage
    )
    prefix = _tendon_attr_prefix(prim, "PhysxTendonAttachmentRootAPI")
    assert abs(prim.GetAttribute(f"{prefix}:stiffness").Get() - 4.0) < 1e-6
    assert abs(prim.GetAttribute(f"{prefix}:limitStiffness").Get() - 0.25) < 1e-6
    assert not prim.HasAttribute(f"{prefix}:func")
    assert not prim.HasAttribute("PhysxTendonAttachmentRootAPI:s0:stiffness")


def test_multi_apply_fragment_metadata_matches_the_schema_definition():
    """The applier trusts the fragment's static data, so the schema registry checks it here.

    For each PhysX tendon fragment: the schema is registered and multiple-apply, every field is a
    property the schema declares, and ``<namespace>:<instance>:<property>`` spells each attribute
    exactly as USD would (``Usd.SchemaRegistry.MakeMultipleApplyNameInstance``).
    """
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg, PhysxSpatialTendonCfg

    from isaaclab.utils.string import to_camel_case

    registry = Usd.SchemaRegistry()
    for cfg_cls in (PhysxFixedTendonCfg, PhysxSpatialTendonCfg):
        schema_name = cfg_cls._usd_applied_schema
        definition = registry.FindAppliedAPIPrimDefinition(schema_name)
        assert definition is not None, f"{schema_name} is not registered"
        assert registry.IsMultipleApplyAPISchema(schema_name)
        templates = {n.rsplit(":", 1)[-1]: n for n in definition.GetPropertyNames() if "__INSTANCE_NAME__" in n}
        for f in dataclasses.fields(cfg_cls):
            if f.name in ("func", "instance_names"):
                continue
            prop = to_camel_case(f.name, "cC")
            assert prop in templates, f"{cfg_cls.__name__}.{f.name} is not a property of {schema_name}"
            expected = Usd.SchemaRegistry.MakeMultipleApplyNameInstance(templates[prop], "inst")
            assert f"{cfg_cls._usd_namespace}:inst:{prop}" == expected


def test_apply_spatial_tendon_writes_all_instances():
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg

    from isaaclab.sim.schemas import apply_multi_apply

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_prim_with_schemas(
        stage,
        "/World/STmulti",
        ["PhysxTendonAttachmentRootAPI:r0", "PhysxTendonAttachmentLeafAPI:l0"],
    )
    assert apply_multi_apply(PhysxSpatialTendonCfg(instance_names=None, stiffness=4.0), "/World/STmulti", stage) is True
    assert abs(prim.GetAttribute("physxTendon:r0:stiffness").Get() - 4.0) < 1e-6
    # Stiffness is a property of the tendon, declared only by the root schema; the leaf attachment
    # declares geometry and limits, so it is skipped rather than given an attribute PhysX ignores.
    assert not prim.HasAttribute("physxTendon:l0:stiffness")


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
        [
            PhysxFixedTendonCfg(instance_names=None, stiffness=5.0),
            PhysxFixedTendonCfg(instance_names=None, damping=0.75),
        ],
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
        [
            PhysxSpatialTendonCfg(instance_names=None, stiffness=6.0),
            PhysxSpatialTendonCfg(instance_names=None, offset=0.1),
        ],
        stage,
    )
    prefix = _tendon_attr_prefix(prim, "PhysxTendonAttachmentRootAPI")
    assert abs(prim.GetAttribute(f"{prefix}:stiffness").Get() - 6.0) < 1e-6
    assert abs(prim.GetAttribute(f"{prefix}:offset").Get() - 0.1) < 1e-6


# -------------------------------------------------------------------------------------
# Public imports
# -------------------------------------------------------------------------------------


def test_public_imports():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg, PhysxSpatialTendonCfg  # noqa: F401

    from isaaclab.sim.schemas import (  # noqa: F401
        FixedTendonFragment,
        SpatialTendonFragment,
        apply_fixed_tendon_properties,
        apply_multi_apply,
        apply_spatial_tendon_properties,
    )


# -------------------------------------------------------------------------------------
# core writer parity: invalid-prim guard + aggregated return
# -------------------------------------------------------------------------------------


def test_apply_fixed_tendon_warns_on_unmatched_path(caplog):
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import apply_fixed_tendon_properties

    _new_sim()
    stage = sim_utils.get_current_stage()
    with caplog.at_level("WARNING"):
        result = apply_fixed_tendon_properties(
            "/World/DoesNotExist", [PhysxFixedTendonCfg(instance_names=None, stiffness=1.0)], stage
        )
    assert result is False
    assert "/World/DoesNotExist" in caplog.text


def test_apply_spatial_tendon_warns_on_unmatched_path(caplog):
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg

    from isaaclab.sim.schemas import apply_spatial_tendon_properties

    _new_sim()
    stage = sim_utils.get_current_stage()
    with caplog.at_level("WARNING"):
        result = apply_spatial_tendon_properties(
            "/World/DoesNotExist", [PhysxSpatialTendonCfg(instance_names=None, stiffness=1.0)], stage
        )
    assert result is False
    assert "/World/DoesNotExist" in caplog.text


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


def test_apply_fixed_tendon_properties_narrows_to_exact_path():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import apply_fixed_tendon_properties

    stage = _new_sim()
    first = _make_fixed_tendon_prim(stage, "/World/Narrow/J0", instance="t1")
    second = _make_fixed_tendon_prim(stage, "/World/Narrow/J1", instance="t1")
    # the exact path of the first joint must tune only that joint
    assert (
        apply_fixed_tendon_properties(
            "/World/Narrow/J0", [PhysxFixedTendonCfg(instance_names=None, stiffness=50.0)], stage
        )
        is True
    )
    prefix = _tendon_attr_prefix(first, "PhysxTendonAxisRootAPI")
    assert abs(first.GetAttribute(f"{prefix}:stiffness").Get() - 50.0) < 1e-6
    second_prefix = _tendon_attr_prefix(second, "PhysxTendonAxisRootAPI")
    second_attr = second.GetAttribute(f"{second_prefix}:stiffness")
    assert not (second_attr and second_attr.HasAuthoredValue())


def test_apply_fixed_tendon_properties_bare_parent_path_does_not_descend(caplog):
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import apply_fixed_tendon_properties

    stage = _new_sim()
    UsdGeom.Xform.Define(stage, "/World/Parent")  # plain Xform: carries no tendon schema
    first = _make_fixed_tendon_prim(stage, "/World/Parent/J0", instance="t1")
    second = _make_fixed_tendon_prim(stage, "/World/Parent/J1", instance="t1")
    # a bare parent path (no ``(/.*)?`` suffix) matches only the parent, which is not a tendon target
    with caplog.at_level("WARNING"):
        result = apply_fixed_tendon_properties(
            "/World/Parent", [PhysxFixedTendonCfg(instance_names=None, stiffness=50.0)], stage
        )
    assert result is False
    for prim in (first, second):
        prefix = _tendon_attr_prefix(prim, "PhysxTendonAxisRootAPI")
        attr = prim.GetAttribute(f"{prefix}:stiffness")
        assert not (attr and attr.HasAuthoredValue())
    assert "/World/Parent" in caplog.text


def test_apply_fixed_tendon_raises_on_invalid_prim_backend():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import apply_multi_apply

    _new_sim()
    stage = sim_utils.get_current_stage()
    with pytest.raises(ValueError):
        apply_multi_apply(PhysxFixedTendonCfg(instance_names=None, stiffness=1.0), "/World/DoesNotExist", stage)


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
    # a MjcTendon is a prim, not a multiple-apply instance, so the fragment carries no instance coordinate
    assert not hasattr(cfg, "instance_names")
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
    apply_fixed_tendon_properties(
        "/World/fragment(/.*)?", [PhysxFixedTendonCfg(instance_names=None, limit_stiffness=30.0, damping=0.1)]
    )

    def _collect(root):
        attrs = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            for schema_name in prim.GetAppliedSchemas():
                if "PhysxTendonAxisRootAPI" not in schema_name:
                    continue
                instance_name = schema_name.split(":", maxsplit=1)[1]
                prefix = f"physxTendon:{instance_name}"
                for suffix in ("limitStiffness", "damping"):
                    attr = prim.GetAttribute(f"{prefix}:{suffix}")
                    if attr and attr.HasAuthoredValue():
                        rel = prim.GetPath().pathString[len(root) :]  # key relative to root so paths compare
                        attrs[f"{rel}|{prefix}:{suffix}"] = attr.Get()
        return attrs

    legacy = _collect("/World/legacy")
    fragment = _collect("/World/fragment")

    assert legacy, "legacy writer authored no tendon attributes (test would be vacuous)"
    assert legacy.keys() == fragment.keys()
    for key, value in legacy.items():
        assert abs(fragment[key] - value) < 1e-6


def test_legacy_and_fragment_spatial_tendon_produce_identical_attrs():
    """The legacy and fragment writers must tune the same schema-owned spatial-tendon attributes."""
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg, PhysxSpatialTendonPropertiesCfg

    from isaaclab.sim.schemas import apply_spatial_tendon_properties, modify_spatial_tendon_properties

    stage = _new_sim()
    legacy_prim = _make_spatial_tendon_prim(stage, "/World/legacySpatial", instance="s0")
    fragment_prim = _make_spatial_tendon_prim(stage, "/World/fragmentSpatial", instance="s0")

    modify_spatial_tendon_properties(
        "/World/legacySpatial",
        PhysxSpatialTendonPropertiesCfg(stiffness=6.0, damping=0.2, offset=0.1),
        stage,
    )
    apply_spatial_tendon_properties(
        "/World/fragmentSpatial",
        [PhysxSpatialTendonCfg(instance_names=None, stiffness=6.0, damping=0.2, offset=0.1)],
        stage,
    )

    for suffix in ("stiffness", "damping", "offset"):
        legacy_value = legacy_prim.GetAttribute(f"physxTendon:s0:{suffix}").Get()
        fragment_value = fragment_prim.GetAttribute(f"physxTendon:s0:{suffix}").Get()
        assert abs(fragment_value - legacy_value) < 1e-6
    assert not legacy_prim.HasAttribute("PhysxTendonAttachmentRootAPI:s0:stiffness")
    assert not fragment_prim.HasAttribute("PhysxTendonAttachmentRootAPI:s0:stiffness")


def test_legacy_spatial_tendon_skips_leaf_instances():
    """A leaf attachment declares none of this cfg's properties, so it is skipped, not written."""
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonPropertiesCfg

    from isaaclab.sim.schemas import modify_spatial_tendon_properties

    stage = _new_sim()
    # A single-attachment tendon carries both instances on one prim; a branching tendon has
    # leaf-only prims. Both reach the writer, and neither declares stiffness/damping/offset.
    both_prim = _make_spatial_tendon_prim(stage, "/World/rootAndLeaf", instance="s0")
    PhysxSchema.PhysxTendonAttachmentLeafAPI.Apply(both_prim, "s0")
    leaf_prim = _make_spatial_tendon_leaf_prim(stage, "/World/leafOnly", instance="s0")

    # a root under the leaf-only prim: reached only if the leaf-only prim reports "nothing written",
    # since ``apply_nested`` stops descending at the first prim the writer succeeds on
    nested_root = _make_spatial_tendon_prim(stage, "/World/leafOnly/root", instance="s0")

    cfg = PhysxSpatialTendonPropertiesCfg(stiffness=6.0, damping=0.2, offset=0.1)
    for path in ("/World/rootAndLeaf", "/World/leafOnly"):
        modify_spatial_tendon_properties(path, cfg, stage)

    for suffix, expected in (("stiffness", 6.0), ("damping", 0.2), ("offset", 0.1)):
        assert abs(both_prim.GetAttribute(f"physxTendon:s0:{suffix}").Get() - expected) < 1e-6
        assert abs(nested_root.GetAttribute(f"physxTendon:s0:{suffix}").Get() - expected) < 1e-6
        # the leaf-only prim declares no root property, so nothing is authored on it
        assert not leaf_prim.GetAttribute(f"physxTendon:s0:{suffix}").IsValid()


def test_spawn_from_file_with_empty_tendon_lists_is_noop(tmp_path):
    # a mapping entry with an empty fragment list is type-valid for the slot; the spawner must route
    # it through the fragment path (a no-op) rather than the legacy modify_*_tendon_properties writer.
    asset = tmp_path / "mini.usda"
    src = Usd.Stage.CreateNew(str(asset))
    UsdGeom.Xform.Define(src, "/Root")
    src.SetDefaultPrim(src.GetPrimAtPath("/Root"))
    src.GetRootLayer().Save()
    del src

    _new_sim()
    cfg = sim_utils.UsdFileCfg(
        usd_path=str(asset), fixed_tendons_props={"(/.*)?": []}, spatial_tendons_props={"(/.*)?": []}
    )
    cfg.func("/World/Asset", cfg)  # must not raise
    assert sim_utils.get_current_stage().GetPrimAtPath("/World/Asset").IsValid()
