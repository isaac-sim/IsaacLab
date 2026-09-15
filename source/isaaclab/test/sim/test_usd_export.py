# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent USD preservation, fixed mapping and atomic-save contracts."""

import math
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.assets.physics_properties import (
    ACTUATOR_CONFIGURATION_SOURCES,
    ASSET_CONFIGURATION_SOURCES,
    JOINT_USD_PROPERTIES,
    validate_configuration_coverage,
)
from isaaclab.scene import InteractiveScene
from isaaclab.sim.usd_export import copy_scene_stage, save_stage, write_properties


@pytest.fixture
def scene():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    material = UsdShade.Material.Define(stage, "/Shared/Material")
    UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateStaticFrictionAttr().Set(0.73)
    for name, mass in (("Body", 2), ("Other", 4)):
        prim = UsdGeom.Cube.Define(stage, "/World/envs/env_0/" + name).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(mass)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    return SimpleNamespace(sim=SimpleNamespace(stage=stage), num_envs=1)


def test_copy_preserves_authored_content_and_source(scene, tmp_path):
    stage = scene.sim.stage
    before = stage.GetRootLayer().ExportToString()
    expected = {
        str(prim.GetPath()): {
            prop.GetName(): prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.Get()
            for prop in prim.GetAuthoredProperties()
        }
        for prim in stage.Traverse()
    }
    output = tmp_path / "complete.usda"
    save_stage(copy_scene_stage(stage), str(output))
    fresh = Usd.Stage.Open(str(output))
    assert {str(p.GetPath()) for p in fresh.Traverse()} == set(expected)
    for path, properties in expected.items():
        for name, value in properties.items():
            prop = fresh.GetPrimAtPath(path).GetProperty(name)
            assert (prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.Get()) == value
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize("angular", [False, True])
def test_declared_joint_mapping_authors_effective_values(angular):
    stage = Usd.Stage.CreateInMemory()
    prim = (UsdPhysics.RevoluteJoint if angular else UsdPhysics.PrismaticJoint).Define(stage, "/Joint").GetPrim()
    values = dict(
        joint_stiffness=[83.0],
        joint_damping=[4.5],
        joint_armature=[0.023],
        joint_friction_coeff=[0.3],
        joint_dynamic_friction_coeff=[0.2],
        joint_viscous_friction_coeff=[0.1],
        joint_effort_limits=[19.0],
        joint_vel_limits=[2.5],
        joint_pos_limits=[[-0.4, 0.8]],
        joint_pos=[0.21],
        joint_vel=[0.17],
    )
    write_properties(prim, {k: np.array(v) for k, v in values.items()}, 0, JOINT_USD_PROPERTIES)
    axis, angle = ("angular", 180 / math.pi) if angular else ("linear", 1)
    # Independent numeric oracle, not expected values derived from the mapping under test.
    expected = {
        f"drive:{axis}:physics:stiffness": 83 / angle,
        f"drive:{axis}:physics:damping": 4.5 / angle,
        f"drive:{axis}:physics:maxForce": 19,
        "physxJoint:armature": 0.023,
        f"physxJointAxis:{axis}:armature": 0.023,
        f"physxJointAxis:{axis}:staticFrictionEffort": 0.3,
        f"physxJointAxis:{axis}:dynamicFrictionEffort": 0.2,
        f"physxJointAxis:{axis}:viscousFrictionCoefficient": 0.1 / angle,
        "physxJoint:maxJointVelocity": 2.5 * angle,
        f"physxJointAxis:{axis}:maxJointVelocity": 2.5 * angle,
        "physics:lowerLimit": -0.4 * angle,
        "physics:upperLimit": 0.8 * angle,
        f"state:{axis}:physics:position": 0.21 * angle,
        f"state:{axis}:physics:velocity": 0.17 * angle,
    }
    for name, value in expected.items():
        assert prim.GetAttribute(name).Get() == pytest.approx(value), name
    with pytest.raises(NotImplementedError, match="Unsupported driven joint"):
        write_properties(
            UsdPhysics.SphericalJoint.Define(stage, "/Unsupported").GetPrim(), values, 0, JOINT_USD_PROPERTIES
        )


def test_fixed_export_rejects_replication_and_poststep(scene, tmp_path):
    scene.num_envs = 2
    with pytest.raises(ValueError, match="exactly one"):
        InteractiveScene.export_to_usd(scene, str(tmp_path / "scene.usda"))
    scene.num_envs = 1
    scene.sim.get_physics_step_count = lambda: 1
    with pytest.raises(ValueError, match="first physics step"):
        InteractiveScene.export_to_usd(scene, str(tmp_path / "scene.usda"))


def test_unregistered_physical_body_fails_before_save(scene, tmp_path):
    import torch

    path = "/World/envs/env_0/Body"
    asset = SimpleNamespace(
        cfg=RigidObjectCfg(prim_path=path),
        root_view=SimpleNamespace(prim_paths=[path]),
        data=SimpleNamespace(
            body_link_pose_w=SimpleNamespace(torch=torch.tensor([[0.0, 0, 0, 0, 0, 0, 1]])),
            body_com_vel_w=SimpleNamespace(torch=torch.zeros((1, 6))),
        ),
    )
    scene.articulations = {}
    scene.rigid_objects = {"body": asset}
    scene.rigid_object_collections = {}
    scene.deformable_objects = {}
    scene.cable_objects = {}
    scene.surface_grippers = {}
    scene.physics_scene_path = "/physicsScene"
    from isaaclab_physx.sim.usd_export import SceneAdapter

    from isaaclab.assets import BaseRigidObject

    asset.author_fixed_configuration = lambda stage, adapter: BaseRigidObject.author_fixed_configuration(
        asset, stage, adapter
    )
    scene.sim.physics_manager = SimpleNamespace(create_usd_export_adapter=lambda scene: SceneAdapter(scene))
    scene.sim.get_physics_step_count = lambda: 0
    scene.sim.get_physics_dt = lambda: 1 / 60
    output = tmp_path / "complete.usda"
    output.write_text("previous destination")
    before = scene.sim.stage.GetRootLayer().ExportToString()
    with pytest.raises(RuntimeError, match="Incomplete body export"):
        InteractiveScene.export_to_usd(scene, str(output))
    assert output.read_text() == "previous destination"
    assert scene.sim.stage.GetRootLayer().ExportToString() == before


def test_configuration_contract_covers_asset_and_actuator_fields():
    from isaaclab.actuators import ActuatorBaseCfg

    asset_fields = set().union(
        *(
            {field.name for field in fields(cfg)}
            for cfg in (AssetBaseCfg, ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg)
        )
    )
    assert asset_fields == set().union(*ASSET_CONFIGURATION_SOURCES.values())
    assert {field.name for field in fields(ActuatorBaseCfg)} == set().union(*ACTUATOR_CONFIGURATION_SOURCES.values())
    for cfg in (AssetBaseCfg, ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg):
        validate_configuration_coverage(cfg)
    from isaaclab.utils import configclass

    @configclass
    class NewPhysicalProperty(RigidObjectCfg):
        new_physical_property: float = 4.0

    with pytest.raises(NotImplementedError, match="new_physical_property"):
        validate_configuration_coverage(NewPhysicalProperty)


def test_atomic_save_rejects_missing_dependencies_and_preserves_destination(scene, tmp_path, monkeypatch):
    output = tmp_path / "scene.usda"
    output.write_text("existing destination")
    stage = copy_scene_stage(scene.sim.stage)
    body = stage.GetPrimAtPath("/World/envs/env_0/Body")
    for name, target in (
        ("physics:body0", "/Missing"),
        ("physics:filteredPairs", "/Missing"),
        ("collection:colliders:includes", "/Missing"),
        ("test:connection", "/Shared/Material.outputs:missing"),
    ):
        if name == "test:connection":
            body.CreateAttribute(name, Sdf.ValueTypeNames.Float).SetConnections([target])
        else:
            body.CreateRelationship(name).SetTargets([target])
        with pytest.raises(RuntimeError, match="Unresolved export dependency"):
            save_stage(stage, str(output))
        assert output.read_text() == "existing destination"
        body.RemoveProperty(name)
    shader = UsdShade.Shader.Define(stage, "/Shared/Texture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(tmp_path / "missing.png")))
    with pytest.raises(RuntimeError, match="asset dependencies"):
        save_stage(stage, str(output))
    assert output.read_text() == "existing destination"
    stage.RemovePrim("/Shared/Texture")

    def failed_replace(*args):
        raise OSError("injected save failure")

    monkeypatch.setattr("isaaclab.sim.usd_export.os.replace", failed_replace)
    with pytest.raises(OSError, match="injected save failure"):
        save_stage(stage, str(output))
    assert output.read_text() == "existing destination"
    assert list(tmp_path.iterdir()) == [output]
