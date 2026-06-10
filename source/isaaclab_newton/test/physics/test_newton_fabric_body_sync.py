# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton USD/Fabric body binding initialization."""

from __future__ import annotations

from isaaclab.app import AppLauncher

# Launch Isaac Sim before importing Newton modules so USD schema bindings are initialized.
simulation_app = AppLauncher(headless=True).app

from isaaclab_newton.physics import NewtonManager


class _FakeAttribute:
    def __init__(self, value_type, custom):
        self.value_type = value_type
        self.custom = custom
        self.value = None

    def Set(self, value):
        self.value = value


class _FakePrim:
    def __init__(self, valid=True):
        self.valid = valid
        self.attributes = {}
        self.applied_schemas = []
        self.created_world_matrix_attrs = 0
        self.set_world_xform_from_usd = 0

    def IsValid(self):
        return self.valid

    def CreateAttribute(self, name, value_type, custom=False):
        self.attributes[name] = _FakeAttribute(value_type, custom)
        return self.attributes[name]

    def GetAttribute(self, name):
        return self.attributes[name]

    def AddAppliedSchema(self, schema):
        self.applied_schemas.append(schema)


class _FakeStage:
    def __init__(self, prims=None):
        self.prims = prims or {}
        self.defined_prims = []

    def GetPrimAtPath(self, path):
        return self.prims.get(path, _FakePrim(valid=False))

    def DefinePrim(self, path, prim_type):
        prim = _FakePrim()
        self.prims[path] = prim
        self.defined_prims.append((path, prim_type))
        return prim


class _FakeXformable:
    def __init__(self, prim):
        self.prim = prim

    def SetWorldXformFromUsd(self):
        self.prim.set_world_xform_from_usd += 1

    def CreateFabricHierarchyWorldMatrixAttr(self):
        self.prim.created_world_matrix_attrs += 1


class _FakeFabricHierarchy:
    def __init__(self):
        self.update_world_xforms_count = 0

    def update_world_xforms(self):
        self.update_world_xforms_count += 1


class _FakeRt:
    Xformable = _FakeXformable


class _FakeValueTypeNames:
    UInt = "UInt"


class _FakeSdf:
    ValueTypeNames = _FakeValueTypeNames


class _FakeUsdrt:
    Rt = _FakeRt
    Sdf = _FakeSdf


def test_initialize_fabric_body_prims_uses_existing_fabric_prim():
    prim = _FakePrim()
    stage = _FakeStage({"/World/envs/env_0/Robot/base": prim})
    fabric_hierarchy = _FakeFabricHierarchy()

    NewtonManager._initialize_fabric_body_prims(
        stage, fabric_hierarchy, _FakeUsdrt, [("/World/envs/env_0/Robot/base", 3)]
    )

    assert stage.defined_prims == []
    assert prim.set_world_xform_from_usd == 1
    assert prim.created_world_matrix_attrs == 0
    assert prim.GetAttribute("newton:index").value_type == "UInt"
    assert prim.GetAttribute("newton:index").custom is True
    assert prim.GetAttribute("newton:index").value == 3
    assert prim.applied_schemas == ["PhysicsRigidBodyAPI"]
    assert fabric_hierarchy.update_world_xforms_count == 1


def test_initialize_fabric_body_prims_creates_missing_body_as_xform():
    stage = _FakeStage()
    fabric_hierarchy = _FakeFabricHierarchy()

    NewtonManager._initialize_fabric_body_prims(
        stage, fabric_hierarchy, _FakeUsdrt, [("/World/envs/env_1/Robot/joints/forearm", 7)]
    )

    prim = stage.prims["/World/envs/env_1/Robot/joints/forearm"]
    assert stage.defined_prims == [("/World/envs/env_1/Robot/joints/forearm", "Xform")]
    assert prim.set_world_xform_from_usd == 0
    assert prim.created_world_matrix_attrs == 1
    assert prim.GetAttribute("newton:index").value_type == "UInt"
    assert prim.GetAttribute("newton:index").custom is True
    assert prim.GetAttribute("newton:index").value == 7
    assert prim.applied_schemas == ["PhysicsRigidBodyAPI"]
    assert fabric_hierarchy.update_world_xforms_count == 1
