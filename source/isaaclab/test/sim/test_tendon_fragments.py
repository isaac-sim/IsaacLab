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


def _tendon_attr_prefix(prim, schema_type):
    """Return the schema-declared property prefix for one applied tendon API."""
    for schema_name in prim.GetAppliedSchemas():
        applied_type, instance = Usd.SchemaRegistry.GetTypeNameAndInstance(str(schema_name))
        if applied_type == schema_type:
            return f"physxTendon:{instance}"
    raise AssertionError(f"no {schema_type!r} instance on {prim.GetPath()}")


# -------------------------------------------------------------------------------------
# Fixed-tendon marker + metadata defaults
# -------------------------------------------------------------------------------------


def test_fixed_tendon_fragment_metadata_defaults():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import FixedTendonFragment, SchemaFragment

    cfg = PhysxFixedTendonCfg(instance_names="index_finger", stiffness=1.0)
    assert isinstance(cfg, FixedTendonFragment) and isinstance(cfg, SchemaFragment)
    assert cfg.func == "isaaclab_physx.sim.schemas:apply_fixed_tendon"
    assert cfg.instance_names == "index_finger" and cfg.stiffness == 1.0 and cfg.damping is None


def test_fixed_tendon_axis_fragment_metadata_defaults():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonAxisCfg

    from isaaclab.sim.schemas import FixedTendonFragment, SchemaFragment

    cfg = PhysxFixedTendonAxisCfg(instance_names="index_finger", gearing=[-0.5])
    assert isinstance(cfg, FixedTendonFragment) and isinstance(cfg, SchemaFragment)
    assert cfg.func == "isaaclab_physx.sim.schemas:apply_fixed_tendon_axis"
    assert cfg.instance_names == "index_finger" and cfg.gearing == [-0.5]


def test_spatial_tendon_fragment_metadata_defaults():
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg

    from isaaclab.sim.schemas import SchemaFragment, SpatialTendonFragment

    cfg = PhysxSpatialTendonCfg(instance_names="cable", stiffness=2.0)
    assert isinstance(cfg, SpatialTendonFragment) and isinstance(cfg, SchemaFragment)
    assert cfg.func == "isaaclab_physx.sim.schemas:apply_spatial_tendon"
    assert cfg.instance_names == "cable" and cfg.stiffness == 2.0 and cfg.damping is None


def test_multiple_apply_mechanics_are_not_a_public_core_fragment_convention():
    import isaaclab.sim.schemas as schemas

    assert "MultiApplyFragment" not in schemas.__all__
    assert "apply_multi_apply" not in schemas.__all__


# -------------------------------------------------------------------------------------
# PhysxFixedTendonCfg writes the multi-instance namespace
# -------------------------------------------------------------------------------------


def test_physx_fixed_tendon_fragment_writes_instanced_namespace():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg, apply_fixed_tendon

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_fixed_tendon_prim(stage, "/World/FT", instance="t0")
    apply_fixed_tendon(
        PhysxFixedTendonCfg(instance_names="t0", stiffness=3.0, damping=0.5, lower_limit=-0.2, upper_limit=0.4),
        "/World/FT",
        stage,
    )
    prefix = _tendon_attr_prefix(prim, "PhysxTendonAxisRootAPI")
    assert abs(prim.GetAttribute(f"{prefix}:stiffness").Get() - 3.0) < 1e-6
    assert abs(prim.GetAttribute(f"{prefix}:damping").Get() - 0.5) < 1e-6
    assert prim.GetAttribute(f"{prefix}:lowerLimit").Get() == pytest.approx(-0.2)
    assert prim.GetAttribute(f"{prefix}:upperLimit").Get() == pytest.approx(0.4)
    assert not prim.HasAttribute("PhysxTendonAxisRootAPI:t0:stiffness")
    assert not prim.HasAttribute(f"{prefix}:instanceNames")
    assert not prim.HasAttribute(f"{prefix}:func")


# -------------------------------------------------------------------------------------
# PhysxSpatialTendonCfg writes the multi-instance namespace
# -------------------------------------------------------------------------------------


def test_apply_fixed_tendon_selects_one_instance():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg, apply_fixed_tendon

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_prim_with_schemas(stage, "/World/FTmulti", ["PhysxTendonAxisRootAPI:t0", "PhysxTendonAxisRootAPI:t1"])
    assert apply_fixed_tendon(PhysxFixedTendonCfg(instance_names="t0", stiffness=9.0), "/World/FTmulti", stage) is True
    assert prim.GetAttribute("physxTendon:t0:stiffness").Get() == pytest.approx(9.0)
    assert not prim.GetAttribute("physxTendon:t1:stiffness").HasAuthoredValue()


def test_apply_fixed_tendon_broadcasts_by_default():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg, apply_fixed_tendon

    stage = _new_sim()
    prim = _make_prim_with_schemas(
        stage, "/World/FTbroadcast", ["PhysxTendonAxisRootAPI:t0", "PhysxTendonAxisRootAPI:t1"]
    )
    assert apply_fixed_tendon(PhysxFixedTendonCfg(stiffness=9.0), "/World/FTbroadcast", stage)
    for inst in ("t0", "t1"):
        assert prim.GetAttribute(f"physxTendon:{inst}:stiffness").Get() == pytest.approx(9.0)


def test_apply_fixed_tendon_selects_a_list_of_instances():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg, apply_fixed_tendon

    stage = _new_sim()
    prim = _make_prim_with_schemas(
        stage,
        "/World/FTlist",
        ["PhysxTendonAxisRootAPI:t0", "PhysxTendonAxisRootAPI:t1", "PhysxTendonAxisRootAPI:t2"],
    )
    cfg = PhysxFixedTendonCfg(instance_names=["t0", "t2"], damping=0.25)
    assert apply_fixed_tendon(cfg, "/World/FTlist", stage)
    assert prim.GetAttribute("physxTendon:t0:damping").Get() == pytest.approx(0.25)
    assert not prim.GetAttribute("physxTendon:t1:damping").HasAuthoredValue()
    assert prim.GetAttribute("physxTendon:t2:damping").Get() == pytest.approx(0.25)


@pytest.mark.parametrize("instance_names", [[], ""])
def test_apply_fixed_tendon_rejects_empty_instance_selection(instance_names):
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg, apply_fixed_tendon

    stage = _new_sim()
    _make_fixed_tendon_prim(stage, "/World/FTempty", instance="t0")
    with pytest.raises(ValueError, match="instance_names"):
        apply_fixed_tendon(PhysxFixedTendonCfg(instance_names=instance_names, stiffness=1.0), "/World/FTempty", stage)


def test_fixed_tendon_axis_fragment_targets_root_and_child_axes():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonAxisCfg

    from isaaclab.sim.schemas import apply_fixed_tendon_properties

    stage = _new_sim()
    UsdGeom.Xform.Define(stage, "/World/Hand")
    root = _make_fixed_tendon_prim(stage, "/World/Hand/root", instance="index_finger")
    PhysxSchema.PhysxTendonAxisRootAPI.Apply(root, "shared_coupling")
    child = _make_xform(stage, "/World/Hand/child")
    PhysxSchema.PhysxTendonAxisAPI.Apply(child, "index_finger")

    cfg = PhysxFixedTendonAxisCfg(
        instance_names="index_finger", gearing=[-0.5], force_coefficient=[2.0], joint_axis=["rotX"]
    )
    assert apply_fixed_tendon_properties("/World/Hand(/.*)?", [cfg], stage)
    for prim in (root, child):
        prefix = "physxTendon:index_finger"
        assert list(prim.GetAttribute(f"{prefix}:gearing").Get()) == pytest.approx([-0.5])
        assert list(prim.GetAttribute(f"{prefix}:forceCoefficient").Get()) == pytest.approx([2.0])
        assert list(prim.GetAttribute(f"{prefix}:jointAxis").Get()) == ["rotX"]
    assert not root.GetAttribute("physxTendon:shared_coupling:gearing").HasAuthoredValue()


def test_tendon_fragments_match_schema_ownership():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonAxisCfg, PhysxFixedTendonCfg, PhysxSpatialTendonCfg

    from isaaclab.utils.string import to_camel_case

    address_fields = {"func", "instance_names"}
    root_fields = {field.name for field in dataclasses.fields(PhysxFixedTendonCfg)} - address_fields
    axis_fields = {field.name for field in dataclasses.fields(PhysxFixedTendonAxisCfg)} - address_fields
    spatial_fields = {field.name for field in dataclasses.fields(PhysxSpatialTendonCfg)} - address_fields
    assert root_fields == {
        "tendon_enabled",
        "stiffness",
        "damping",
        "limit_stiffness",
        "offset",
        "rest_length",
        "lower_limit",
        "upper_limit",
    }
    assert axis_fields == {"gearing", "force_coefficient", "joint_axis"}
    assert spatial_fields == {"tendon_enabled", "stiffness", "damping", "limit_stiffness", "offset"}
    assert root_fields.isdisjoint(axis_fields)

    registry = Usd.SchemaRegistry()
    for cfg_type, schema_type in (
        (PhysxFixedTendonCfg, "PhysxTendonAxisRootAPI"),
        (PhysxFixedTendonAxisCfg, "PhysxTendonAxisAPI"),
        (PhysxSpatialTendonCfg, "PhysxTendonAttachmentRootAPI"),
    ):
        definition = registry.FindAppliedAPIPrimDefinition(schema_type)
        assert definition is not None
        schema_properties = {
            str(Usd.SchemaRegistry.GetMultipleApplyNameTemplateBaseName(str(name)))
            for name in definition.GetPropertyNames()
            if "__INSTANCE_NAME__" in str(name)
        }
        cfg_properties = {
            to_camel_case(field.name, "cC")
            for field in dataclasses.fields(cfg_type)
            if field.name not in address_fields
        }
        assert cfg_properties <= schema_properties

    with pytest.raises(TypeError):
        PhysxFixedTendonCfg(instance_names="finger", gearing=[1.0])
    with pytest.raises(TypeError):
        PhysxFixedTendonAxisCfg(instance_names="finger", stiffness=1.0)


def test_apply_fixed_tendon_writer_descends_to_child_prims():
    # tendon schemas are authored on child joint prims, not the articulation root the spawner
    # targets. Targeting is owned by the core writer: its subtree expression descends to
    # every descendant carrying the schema, while the backend func is a per-prim tuner that
    # no-ops on a prim without the schema.
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg, apply_fixed_tendon

    from isaaclab.sim.schemas import apply_fixed_tendon_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    UsdGeom.Xform.Define(stage, "/World/Robot")  # root: no tendon schema
    child = _make_fixed_tendon_prim(stage, "/World/Robot/joint", instance="t0")  # child joint carries it
    # the backend func is per-prim: applied at the root it tunes nothing
    assert apply_fixed_tendon(PhysxFixedTendonCfg(stiffness=8.0), "/World/Robot", stage) is False
    # the core writer's subtree expression descends from the root to the joint
    assert apply_fixed_tendon_properties("/World/Robot(/.*)?", [PhysxFixedTendonCfg(stiffness=8.0)], stage) is True
    prefix = _tendon_attr_prefix(child, "PhysxTendonAxisRootAPI")
    assert abs(child.GetAttribute(f"{prefix}:stiffness").Get() - 8.0) < 1e-6


def test_apply_spatial_tendon_writer_descends_to_child_prims():
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg, apply_spatial_tendon

    from isaaclab.sim.schemas import apply_spatial_tendon_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    UsdGeom.Xform.Define(stage, "/World/Robot2")  # root: no tendon schema
    child = _make_prim_with_schemas(stage, "/World/Robot2/joint", ["PhysxTendonAttachmentRootAPI:s0"])
    # the backend func is per-prim: applied at the root it tunes nothing
    assert apply_spatial_tendon(PhysxSpatialTendonCfg(stiffness=5.0), "/World/Robot2", stage) is False
    # the core writer's subtree expression descends from the root to the joint
    assert apply_spatial_tendon_properties("/World/Robot2(/.*)?", [PhysxSpatialTendonCfg(stiffness=5.0)], stage) is True
    assert child.GetAttribute("physxTendon:s0:stiffness").Get() == pytest.approx(5.0)


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


def test_apply_spatial_tendon_selects_roots_and_skips_leaves():
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg, apply_spatial_tendon

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = _make_prim_with_schemas(
        stage,
        "/World/STmulti",
        [
            "PhysxTendonAttachmentRootAPI:r0",
            "PhysxTendonAttachmentRootAPI:r1",
            "PhysxTendonAttachmentLeafAPI:l0",
        ],
    )
    assert apply_spatial_tendon(PhysxSpatialTendonCfg(instance_names="r0", stiffness=4.0), "/World/STmulti", stage)
    assert prim.GetAttribute("physxTendon:r0:stiffness").Get() == pytest.approx(4.0)
    assert not prim.GetAttribute("physxTendon:r1:stiffness").HasAuthoredValue()
    assert not prim.GetAttribute("physxTendon:l0:stiffness").IsValid()


def test_legacy_spatial_tendon_writer_uses_root_property_namespace():
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonPropertiesCfg

    from isaaclab.sim.schemas import modify_spatial_tendon_properties

    stage = _new_sim()
    prim = _make_prim_with_schemas(
        stage,
        "/World/STlegacy",
        ["PhysxTendonAttachmentRootAPI:r0", "PhysxTendonAttachmentLeafAPI:l0"],
    )
    writer = modify_spatial_tendon_properties.__wrapped__
    assert writer("/World/STlegacy", PhysxSpatialTendonPropertiesCfg(stiffness=6.0), stage)
    assert prim.GetAttribute("physxTendon:r0:stiffness").Get() == pytest.approx(6.0)
    assert not prim.HasAttribute("PhysxTendonAttachmentRootAPI:r0:stiffness")
    assert not prim.GetAttribute("physxTendon:l0:stiffness").IsValid()


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
            PhysxFixedTendonCfg(stiffness=5.0),
            PhysxFixedTendonCfg(damping=0.75),
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
            PhysxSpatialTendonCfg(stiffness=6.0),
            PhysxSpatialTendonCfg(offset=0.1),
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
    from isaaclab_physx.sim.schemas import (  # noqa: F401
        PhysxFixedTendonAxisCfg,
        PhysxFixedTendonCfg,
        PhysxSpatialTendonCfg,
        apply_fixed_tendon,
        apply_fixed_tendon_axis,
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


def test_apply_fixed_tendon_warns_on_unmatched_path(caplog):
    from isaaclab_physx.sim.schemas import PhysxFixedTendonCfg

    from isaaclab.sim.schemas import apply_fixed_tendon_properties

    _new_sim()
    stage = sim_utils.get_current_stage()
    with caplog.at_level("WARNING"):
        result = apply_fixed_tendon_properties("/World/DoesNotExist", [PhysxFixedTendonCfg(stiffness=1.0)], stage)
    assert result is False
    assert "/World/DoesNotExist" in caplog.text


def test_apply_spatial_tendon_warns_on_unmatched_path(caplog):
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonCfg

    from isaaclab.sim.schemas import apply_spatial_tendon_properties

    _new_sim()
    stage = sim_utils.get_current_stage()
    with caplog.at_level("WARNING"):
        result = apply_spatial_tendon_properties("/World/DoesNotExist", [PhysxSpatialTendonCfg(stiffness=1.0)], stage)
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
    assert apply_fixed_tendon_properties("/World/Narrow/J0", [PhysxFixedTendonCfg(stiffness=50.0)], stage) is True
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
        result = apply_fixed_tendon_properties("/World/Parent", [PhysxFixedTendonCfg(stiffness=50.0)], stage)
    assert result is False
    for prim in (first, second):
        prefix = _tendon_attr_prefix(prim, "PhysxTendonAxisRootAPI")
        attr = prim.GetAttribute(f"{prefix}:stiffness")
        assert not (attr and attr.HasAuthoredValue())
    assert "/World/Parent" in caplog.text


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


def test_legacy_physx_tendon_cfg_does_not_leak_physx_only_fields_to_mujoco():
    from isaaclab_physx.sim.schemas import PhysxFixedTendonPropertiesCfg

    from isaaclab.sim.schemas import modify_fixed_tendon_properties

    stage = _new_sim()
    prim = stage.DefinePrim("/World/LegacyMjcTendon", "MjcTendon")
    cfg = PhysxFixedTendonPropertiesCfg(stiffness=2.0, damping=0.25, lower_limit=-1.0, upper_limit=1.0)
    assert modify_fixed_tendon_properties.__wrapped__(str(prim.GetPath()), cfg, stage)
    assert prim.GetAttribute("mjc:stiffness").Get() == pytest.approx(2.0)
    assert prim.GetAttribute("mjc:damping").Get() == pytest.approx(0.25)
    assert not prim.HasAttribute("mjc:lowerLimit")
    assert not prim.HasAttribute("mjc:upperLimit")


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
    apply_fixed_tendon_properties("/World/fragment(/.*)?", [PhysxFixedTendonCfg(limit_stiffness=30.0, damping=0.1)])

    def _collect(root):
        attrs = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            for schema_name in prim.GetAppliedSchemas():
                schema_type, instance = Usd.SchemaRegistry.GetTypeNameAndInstance(str(schema_name))
                if schema_type != "PhysxTendonAxisRootAPI":
                    continue
                for suffix in ("limitStiffness", "damping"):
                    attr_name = f"physxTendon:{instance}:{suffix}"
                    attr = prim.GetAttribute(attr_name)
                    if attr and attr.HasAuthoredValue():
                        rel = prim.GetPath().pathString[len(root) :]  # key relative to root so paths compare
                        attrs[f"{rel}|{instance}:{suffix}"] = attr.Get()
        return attrs

    legacy = _collect("/World/legacy")
    fragment = _collect("/World/fragment")

    assert legacy, "legacy writer authored no tendon attributes (test would be vacuous)"
    assert legacy.keys() == fragment.keys()
    for key, value in legacy.items():
        assert abs(fragment[key] - value) < 1e-6


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
