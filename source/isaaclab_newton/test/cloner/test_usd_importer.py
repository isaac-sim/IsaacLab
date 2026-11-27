# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the configurable Newton USD importer."""

import importlib

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonCfg
from newton import JointType, Model, ModelBuilder

from pxr import Sdf, UsdGeom, UsdPhysics

from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import make_clone_plan
from isaaclab.sim import SimulationCfg, SpawnerCfg, build_simulation_context

replicate_module = importlib.import_module("isaaclab_newton.cloner.replicate")

_PHYSICS_SCENE = "/physicsScene"
_SHARED = "/World/shared"
_ROBOT = "/World/env_0/Robot"


def _body(stage, path: str, marker: float):
    prim = UsdGeom.Cube.Define(stage, path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    prim.CreateAttribute("test:importMarker", Sdf.ValueTypeNames.Float, True).Set(marker)
    return prim


def _author_scene(stage) -> None:
    """Author one shared rigid body and one articulated robot source with a revolute joint."""
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, _PHYSICS_SCENE)
    _body(stage, _SHARED, 11.0)
    UsdPhysics.ArticulationRootAPI.Apply(_body(stage, _ROBOT, 17.0))
    _body(stage, f"{_ROBOT}/child", 19.0)
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{_ROBOT}/child/joint")
    joint.CreateAxisAttr().Set("Z")
    joint.CreateBody0Rel().SetTargets([_ROBOT])
    joint.CreateBody1Rel().SetTargets([f"{_ROBOT}/child"])


def _replicate(usd_importer):
    """Replicate the robot into two worlds beside the shared body and return the finalized model."""
    sim_cfg = SimulationCfg(device="cpu", physics=NewtonCfg(load_visual_shapes=False, usd_importer=usd_importer))
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        stage = sim.stage
        _author_scene(stage)
        assets = (
            AssetBaseCfg(prim_path=_SHARED),
            AssetBaseCfg(prim_path="/World/env_[^/]+/Robot", spawn=SpawnerCfg(spawn_path=_ROBOT)),
        )
        positions = np.asarray([[0, 0, 0], [2, 0, 0]], dtype=np.float32)
        placement = dict(env_template="/World/env_{}", positions=positions)
        plan = make_clone_plan(assets, ((1,),), 2, shared_assets=(0,), **placement)
        sim.set_clone_plan(plan)
        builder, _, _ = replicate_module._replicate_newton(
            stage, np.arange(2), sim, plan=plan, asset_prototype_ids=(0, 1), positions=positions
        )
        return builder, builder.finalize(device="cpu")


def test_custom_importer_receives_every_physics_import():
    """The configured importer runs for the stage and every source, with native options, and its result is kept."""
    calls = []

    def importer(builder, source, **native_options):
        calls.append(native_options)
        if not builder.has_custom_attribute("import_test:marker"):
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="marker",
                    namespace="import_test",
                    dtype=wp.float32,
                    frequency=Model.AttributeFrequency.BODY,
                    default=-1.0,
                    usd_attribute_name="test:importMarker",
                )
            )
        return builder.add_usd(source, **native_options)

    builder, model = _replicate(importer)

    assert {options["root_path"] for options in calls} == {_PHYSICS_SCENE, _SHARED, _ROBOT}
    assert all(options["schema_resolvers"] for options in calls)
    assert sorted(model.import_test.marker.numpy().tolist()) == [11.0, 17.0, 17.0, 19.0, 19.0]
    assert sum(joint_type == JointType.REVOLUTE for joint_type in builder.joint_type) == 2


def test_default_importer_is_native():
    """Without a configured importer, replication imports natively and registers no project attribute."""
    builder, model = _replicate(None)

    assert model.body_count == 5
    assert not builder.has_custom_attribute("import_test:marker")


def test_importer_exception_propagates():
    """An importer failure is raised unchanged instead of falling back to the native importer."""
    error = RuntimeError("custom USD import failed")

    def importer(_builder, _source, **_native_options):
        raise error

    with pytest.raises(RuntimeError) as exc_info:
        _replicate(importer)
    assert exc_info.value is error
