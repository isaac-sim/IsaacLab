# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for fixing a referenced articulation root link with pure USD.

These tests run on an in-memory USD stage and intentionally do NOT launch Isaac Sim /
Kit. Assets converted with ``MjcfConverter``/``UrdfConverter`` carry the articulation root and
its authored root properties (e.g. Newton's ``NewtonArticulationRootAPI`` attributes) in the
referenced asset layer, so the consuming stage's edit target cannot remove them. Fixing such a
root with the legacy ``fix_root_link=True`` must relocate it without raising.
"""

import pathlib

import pytest

newton_usd_schemas = pytest.importorskip("newton_usd_schemas", reason="newton-usd-schemas wheel not installed")

from pxr import Plug, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim.schemas import schemas, schemas_cfg

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def _ensure_newton_schema_registered() -> None:
    """Make the registry resolve ``NewtonArticulationRootAPI`` the way the Isaac Sim plugin does."""
    if Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("NewtonArticulationRootAPI") is not None:
        return
    Plug.Registry().RegisterPlugins(str(pathlib.Path(newton_usd_schemas.__file__).parent))
    assert Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("NewtonArticulationRootAPI") is not None


def _author_newton_root_asset(asset_path: str) -> None:
    """Author an asset whose root link carries the articulation root and an authored Newton property."""
    src = Usd.Stage.CreateNew(asset_path)
    src.SetDefaultPrim(UsdGeom.Xform.Define(src, "/Robot").GetPrim())
    base = UsdGeom.Xform.Define(src, "/Robot/base").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base)
    UsdPhysics.ArticulationRootAPI.Apply(base)
    base.AddAppliedSchema("NewtonArticulationRootAPI")
    base.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    src.Save()


@pytest.fixture()
def referenced_root_stage(tmp_path):
    """A memory stage referencing the asset, as the spawner authors it."""
    _ensure_newton_schema_registered()
    asset = str(tmp_path / "robot.usda")
    _author_newton_root_asset(asset)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    stage.DefinePrim("/World/Robot").GetReferences().AddReference(asset)
    return stage


def test_fix_root_link_relocates_referenced_newton_root(referenced_root_stage):
    """``fix_root_link=True`` must relocate a referenced Newton root link without raising.

    The Newton property is authored in the reference layer, so the edit target cannot remove
    it from the former root. The relocation still has to move it onto the parent, apply the
    schema there, author the world fixed joint, and strip the Newton and articulation-root
    schemas from the link in the composed stage.
    """
    stage = referenced_root_stage
    root = stage.GetPrimAtPath("/World/Robot/base")
    parent = stage.GetPrimAtPath("/World/Robot")

    # the writer returns None through the apply_nested shim; assert the authored effects
    schemas.modify_articulation_root_properties(
        "/World/Robot/base", schemas_cfg.ArticulationRootBaseCfg(fix_root_link=True), stage=stage
    )

    # the parent carries the relocated root: schema and the authored Newton property
    assert parent.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert "NewtonArticulationRootAPI" in parent.GetAppliedSchemas()
    assert parent.GetAttribute("newton:selfCollisionEnabled").Get() is False
    # the world fixed joint was authored under the root link
    fixed_joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.FixedJoint)]
    assert len(fixed_joints) == 1
    assert fixed_joints[0].GetPath().pathString.startswith("/World/Robot/base")
    # the former root link is no longer a root: the composed schemas drop the Newton API and
    # the articulation root API even though their opinions live in the reference layer
    assert not root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert "NewtonArticulationRootAPI" not in root.GetPrimTypeInfo().GetAppliedAPISchemas()


def test_fix_root_link_still_removes_local_newton_properties():
    """Without a reference (opinions in the edit target) the removal stays complete.

    Guards the best-effort fallback: when the properties are removable, relocating must not
    leave them authored on the former root link.
    """
    _ensure_newton_schema_registered()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    base = UsdGeom.Xform.Define(stage, "/World/Robot/base").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base)
    UsdPhysics.ArticulationRootAPI.Apply(base)
    base.AddAppliedSchema("NewtonArticulationRootAPI")
    base.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)

    # the writer returns None through the apply_nested shim; assert the authored effects
    schemas.modify_articulation_root_properties(
        "/World/Robot/base", schemas_cfg.ArticulationRootBaseCfg(fix_root_link=True), stage=stage
    )

    root = stage.GetPrimAtPath("/World/Robot/base")
    parent = stage.GetPrimAtPath("/World/Robot")
    assert parent.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert "NewtonArticulationRootAPI" in parent.GetAppliedSchemas()
    assert parent.GetAttribute("newton:selfCollisionEnabled").Get() is False
    assert not root.HasAttribute("newton:selfCollisionEnabled")
    assert not root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert "NewtonArticulationRootAPI" not in root.GetPrimTypeInfo().GetAppliedAPISchemas()
