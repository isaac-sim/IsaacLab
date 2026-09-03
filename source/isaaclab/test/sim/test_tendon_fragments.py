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
from isaaclab_newton.sim.schemas import MujocoFixedTendonCfg, apply_mujoco_fixed_tendon
from isaaclab_physx.sim.schemas import (
    PhysxFixedTendonPropertiesCfg,
    PhysxSpatialTendonPropertiesCfg,
    PhysxTendonAttachmentRootCfg,
    PhysxTendonAxisCfg,
    PhysxTendonAxisRootCfg,
)

from pxr import PhysxSchema, Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.schemas import (
    apply_fixed_tendon_properties,
    apply_spatial_tendon_properties,
    modify_fixed_tendon_properties,
    modify_spatial_tendon_properties,
)
from isaaclab.utils.string import to_camel_case

pytestmark = pytest.mark.integration


def _new_sim():
    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    return sim_utils.get_current_stage()


def _make_prim_with_schemas(stage, path, schema_tokens):
    prim = _make_xform(stage, path)
    token_op = Sdf.TokenListOp()
    token_op.explicitItems = schema_tokens
    prim.SetMetadata("apiSchemas", token_op)
    return prim


def _make_xform(stage, path="/World/Tendon"):
    UsdGeom.Xform.Define(stage, path)
    return stage.GetPrimAtPath(path)


def _make_fixed_tendon_prim(stage, path, instance="default"):
    prim = _make_xform(stage, path)
    PhysxSchema.PhysxTendonAxisRootAPI.Apply(prim, instance)
    return prim


def test_tendon_axis_root_fragment_writes_instanced_namespace():
    stage = _new_sim()
    prim = _make_fixed_tendon_prim(stage, "/World/FT", instance="t0")
    assert apply_fixed_tendon_properties(
        "/World/FT",
        [PhysxTendonAxisRootCfg(instance_names="t0", stiffness=3.0, damping=0.5, lower_limit=-0.2, upper_limit=0.4)],
        stage,
    )
    prefix = "physxTendon:t0"
    assert abs(prim.GetAttribute(f"{prefix}:stiffness").Get() - 3.0) < 1e-6
    assert abs(prim.GetAttribute(f"{prefix}:damping").Get() - 0.5) < 1e-6
    assert prim.GetAttribute(f"{prefix}:lowerLimit").Get() == pytest.approx(-0.2)
    assert prim.GetAttribute(f"{prefix}:upperLimit").Get() == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("instance_names", "selected"),
    [("t0", {"t0"}), (None, {"t0", "t1", "t2"}), (["t0", "t2"], {"t0", "t2"})],
)
def test_fixed_tendon_instance_selection(instance_names, selected):
    stage = _new_sim()
    prim = _make_prim_with_schemas(
        stage,
        "/World/FTmulti",
        ["PhysxTendonAxisRootAPI:t0", "PhysxTendonAxisRootAPI:t1", "PhysxTendonAxisRootAPI:t2"],
    )
    cfg = PhysxTendonAxisRootCfg(instance_names=instance_names, stiffness=9.0)
    assert apply_fixed_tendon_properties("/World/FTmulti", [cfg], stage)
    for instance in ("t0", "t1", "t2"):
        attr = prim.GetAttribute(f"physxTendon:{instance}:stiffness")
        assert attr.HasAuthoredValue() is (instance in selected)
        if instance in selected:
            assert attr.Get() == pytest.approx(9.0)


def test_fixed_tendon_axis_fragment_targets_root_and_child_axes():
    stage = _new_sim()
    UsdGeom.Xform.Define(stage, "/World/Hand")
    root = _make_fixed_tendon_prim(stage, "/World/Hand/root", instance="index_finger")
    PhysxSchema.PhysxTendonAxisRootAPI.Apply(root, "shared_coupling")
    child = _make_xform(stage, "/World/Hand/child")
    PhysxSchema.PhysxTendonAxisAPI.Apply(child, "index_finger")

    cfg = PhysxTendonAxisCfg(
        instance_names="index_finger", gearing=[-0.5], force_coefficient=[2.0], joint_axis=["rotX"]
    )
    assert apply_fixed_tendon_properties("/World/Hand(/.*)?", [cfg], stage)
    for prim in (root, child):
        prefix = "physxTendon:index_finger"
        assert list(prim.GetAttribute(f"{prefix}:gearing").Get()) == pytest.approx([-0.5])
        assert list(prim.GetAttribute(f"{prefix}:forceCoefficient").Get()) == pytest.approx([2.0])
        assert list(prim.GetAttribute(f"{prefix}:jointAxis").Get()) == ["rotX"]
    assert not root.GetAttribute("physxTendon:shared_coupling:gearing").HasAuthoredValue()


@pytest.mark.parametrize(
    ("cfg_type", "schema_type"),
    [
        (PhysxTendonAxisRootCfg, "PhysxTendonAxisRootAPI"),
        (PhysxTendonAxisCfg, "PhysxTendonAxisAPI"),
        (PhysxTendonAttachmentRootCfg, "PhysxTendonAttachmentRootAPI"),
    ],
)
def test_tendon_fragment_fields_belong_to_schema(cfg_type, schema_type):
    address_fields = {"func", "instance_names"}
    definition = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(schema_type)
    schema_properties = {
        str(Usd.SchemaRegistry.GetMultipleApplyNameTemplateBaseName(str(name)))
        for name in definition.GetPropertyNames()
        if "__INSTANCE_NAME__" in str(name)
    }
    cfg_properties = {
        to_camel_case(field.name, "cC") for field in dataclasses.fields(cfg_type) if field.name not in address_fields
    }
    assert cfg_properties <= schema_properties


@pytest.mark.parametrize(
    ("writer", "cfg", "schema"),
    [
        (apply_fixed_tendon_properties, PhysxTendonAxisRootCfg(stiffness=5.0), "PhysxTendonAxisRootAPI:t0"),
        (
            apply_spatial_tendon_properties,
            PhysxTendonAttachmentRootCfg(stiffness=5.0),
            "PhysxTendonAttachmentRootAPI:t0",
        ),
    ],
)
def test_tendon_property_writers_descend_to_child_prims(writer, cfg, schema):
    stage = _new_sim()
    UsdGeom.Xform.Define(stage, "/World/Robot")
    child = _make_prim_with_schemas(stage, "/World/Robot/joint", [schema])
    assert writer("/World/Robot(/.*)?", [cfg], stage)
    assert child.GetAttribute("physxTendon:t0:stiffness").Get() == pytest.approx(5.0)


def test_spatial_tendon_selects_root_instance_and_skips_leaves():
    stage = _new_sim()
    prim = _make_prim_with_schemas(
        stage,
        "/World/STmulti",
        [
            "PhysxTendonAttachmentRootAPI:r0",
            "PhysxTendonAttachmentRootAPI:r1",
            "PhysxTendonAttachmentLeafAPI:l0",
        ],
    )
    assert apply_spatial_tendon_properties(
        "/World/STmulti",
        [PhysxTendonAttachmentRootCfg(instance_names="r0", stiffness=4.0, limit_stiffness=0.25)],
        stage,
    )
    assert prim.GetAttribute("physxTendon:r0:stiffness").Get() == pytest.approx(4.0)
    assert prim.GetAttribute("physxTendon:r0:limitStiffness").Get() == pytest.approx(0.25)
    assert not prim.GetAttribute("physxTendon:r1:stiffness").HasAuthoredValue()
    assert not prim.GetAttribute("physxTendon:l0:stiffness").IsValid()


def test_legacy_spatial_tendon_writer_uses_root_property_namespace():
    stage = _new_sim()
    prim = _make_prim_with_schemas(
        stage,
        "/World/STlegacy",
        ["PhysxTendonAttachmentRootAPI:r0", "PhysxTendonAttachmentLeafAPI:l0"],
    )
    writer = modify_spatial_tendon_properties.__wrapped__
    assert writer("/World/STlegacy", PhysxSpatialTendonPropertiesCfg(stiffness=6.0), stage)
    assert prim.GetAttribute("physxTendon:r0:stiffness").Get() == pytest.approx(6.0)
    assert not prim.GetAttribute("physxTendon:l0:stiffness").IsValid()


def test_tendon_writer_dispatches_multiple_fragments():
    stage = _new_sim()
    prim = _make_prim_with_schemas(stage, "/World/Tendon", ["PhysxTendonAxisRootAPI:t0"])
    fragments = [PhysxTendonAxisRootCfg(stiffness=5.0), PhysxTendonAxisRootCfg(damping=0.75)]
    assert apply_fixed_tendon_properties("/World/Tendon", fragments, stage)
    assert prim.GetAttribute("physxTendon:t0:stiffness").Get() == pytest.approx(5.0)
    assert prim.GetAttribute("physxTendon:t0:damping").Get() == pytest.approx(0.75)


def test_apply_mujoco_fixed_tendon_writes_mjc_namespace():
    stage = _new_sim()
    stage.DefinePrim("/World/MjcT", "MjcTendon")
    assert apply_mujoco_fixed_tendon(MujocoFixedTendonCfg(stiffness=2.0, damping=0.25), "/World/MjcT", stage) is True
    prim = stage.GetPrimAtPath("/World/MjcT")
    assert abs(prim.GetAttribute("mjc:stiffness").Get() - 2.0) < 1e-6
    assert abs(prim.GetAttribute("mjc:damping").Get() - 0.25) < 1e-6


def test_legacy_physx_tendon_cfg_does_not_leak_physx_only_fields_to_mujoco():
    stage = _new_sim()
    prim = stage.DefinePrim("/World/LegacyMjcTendon", "MjcTendon")
    cfg = PhysxFixedTendonPropertiesCfg(stiffness=2.0, damping=0.25, lower_limit=-1.0, upper_limit=1.0)
    assert modify_fixed_tendon_properties.__wrapped__(str(prim.GetPath()), cfg, stage)
    assert prim.GetAttribute("mjc:stiffness").Get() == pytest.approx(2.0)
    assert prim.GetAttribute("mjc:damping").Get() == pytest.approx(0.25)
    assert not prim.HasAttribute("mjc:lowerLimit")
    assert not prim.HasAttribute("mjc:upperLimit")


def test_apply_mujoco_fixed_tendon_returns_false_on_non_mjc_prim():
    stage = _new_sim()
    UsdGeom.Xform.Define(stage, "/World/NotMjc")
    assert apply_mujoco_fixed_tendon(MujocoFixedTendonCfg(stiffness=2.0), "/World/NotMjc", stage) is False
    prim = stage.GetPrimAtPath("/World/NotMjc")
    assert not prim.HasAttribute("mjc:stiffness")


def test_legacy_and_fragment_fixed_tendon_produce_identical_attrs():
    stage = _new_sim()

    for root in ("/World/legacy", "/World/fragment"):
        UsdGeom.Xform.Define(stage, root)
        _make_prim_with_schemas(stage, f"{root}/J0", ["PhysxTendonAxisRootAPI:t0", "PhysxTendonAxisRootAPI:t1"])
        _make_prim_with_schemas(stage, f"{root}/nested/J1", ["PhysxTendonAxisRootAPI:t0"])

    modify_fixed_tendon_properties("/World/legacy", PhysxFixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1))
    apply_fixed_tendon_properties("/World/fragment(/.*)?", [PhysxTendonAxisRootCfg(limit_stiffness=30.0, damping=0.1)])

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
                        rel = prim.GetPath().pathString[len(root) :]
                        attrs[f"{rel}|{instance}:{suffix}"] = attr.Get()
        return attrs

    legacy = _collect("/World/legacy")
    fragment = _collect("/World/fragment")

    assert legacy, "legacy writer authored no tendon attributes (test would be vacuous)"
    assert fragment == pytest.approx(legacy)


def test_spawn_from_file_with_empty_tendon_lists_is_noop(tmp_path):
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
