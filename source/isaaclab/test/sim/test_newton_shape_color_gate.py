"""Regression tests for the Newton shape-color material-binding guard.

``replace_newton_builder_shape_colors`` resolves each shape's bound material via
``UsdShade.MaterialBindingAPI.ComputeBoundMaterial``. On assets whose USD authoring has
out-of-scope ``material:binding*`` targets (e.g. the ShadowHand payload), that traversal can
corrupt the process heap and abort (SIGABRT) during Newton environment cloning.
``_material_binding_targets_are_resolvable`` detects such shapes so they are skipped.
"""

from pxr import Sdf, Usd, UsdGeom

from isaaclab.sim.utils.newton_model_utils import _material_binding_targets_are_resolvable


def _stage_with_prims():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage, "/World/material")
    UsdGeom.Cube.Define(stage, "/World/geom_ok")
    UsdGeom.Cube.Define(stage, "/World/geom_bad")
    UsdGeom.Cube.Define(stage, "/World/geom_none")
    return stage


def test_resolvable_when_binding_target_exists():
    stage = _stage_with_prims()
    prim = stage.GetPrimAtPath("/World/geom_ok")
    prim.CreateRelationship("material:binding").SetTargets([Sdf.Path("/World/material")])
    assert _material_binding_targets_are_resolvable(prim) is True


def test_not_resolvable_when_target_out_of_scope():
    stage = _stage_with_prims()
    prim = stage.GetPrimAtPath("/World/geom_bad")
    prim.CreateRelationship("material:binding:physics").SetTargets([Sdf.Path("/missing/PhysicsMaterial")])
    assert _material_binding_targets_are_resolvable(prim) is False


def test_resolvable_when_no_binding():
    stage = _stage_with_prims()
    prim = stage.GetPrimAtPath("/World/geom_none")
    assert _material_binding_targets_are_resolvable(prim) is True
