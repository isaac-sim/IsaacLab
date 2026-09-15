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
    UsdAttribute,
    usd_field,
    usd_fields,
    validate_configuration_coverage,
)
from isaaclab.scene import InteractiveScene
from isaaclab.sim.usd_export import UsdWriter, copy_scene_stage


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
    UsdWriter(copy_scene_stage(stage)).save(str(output))
    fresh = Usd.Stage.Open(str(output))
    assert {str(p.GetPath()) for p in fresh.Traverse()} == set(expected)
    for path, properties in expected.items():
        for name, value in properties.items():
            prop = fresh.GetPrimAtPath(path).GetProperty(name)
            assert (prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.Get()) == value
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize("angular", [False, True])
def test_declared_joint_mapping_authors_effective_values(angular):
    from isaaclab.assets import BaseArticulationData

    assert BaseArticulationData.joint_stiffness.__isabstractmethod__
    assert BaseArticulationData.joint_stiffness.fget._leapp_semantics.const

    class Declared:
        joint_stiffness = BaseArticulationData.joint_stiffness
        joint_pos_limits = BaseArticulationData.joint_pos_limits

    class Data(Declared):
        reads = 0

        @property
        def joint_stiffness(self):
            self.reads += 1
            return np.array([[83.0, 41.0]])

        @property
        def joint_pos_limits(self):
            self.reads += 1
            return np.array([[[-0.4, 0.8], [-0.2, 0.3]]])

    data = Data()
    assert set(usd_fields(Data)) == {"joint_stiffness", "joint_pos_limits"}
    assert data.reads == 0  # Discovery traverses declarations, never backend getters.
    stage = Usd.Stage.CreateInMemory()
    writer = UsdWriter(stage)
    axis, scale = ("angular", 180 / math.pi) if angular else ("linear", 1.0)
    for row, (gain, lower, upper) in enumerate(((83.0, -0.4, 0.8), (41.0, -0.2, 0.3))):
        prim = (
            (UsdPhysics.RevoluteJoint if angular else UsdPhysics.PrismaticJoint).Define(stage, f"/Joint{row}").GetPrim()
        )
        writer.write_properties(str(prim.GetPath()), axis, data, row=row)
        assert prim.GetAttribute(f"drive:{axis}:physics:stiffness").Get() == pytest.approx(gain / scale)
        assert prim.GetAttribute("physics:lowerLimit").Get() == pytest.approx(lower * scale)
        assert prim.GetAttribute("physics:upperLimit").Get() == pytest.approx(upper * scale)
    assert data.reads == 2  # One read per source, including a two-target source.


def test_binding_override_extension_and_scalar_vector_schema_types():
    class Base:
        @property
        @usd_field(UsdAttribute("physics:mass", "PhysicsMassAPI"))
        def value(self):
            raise AssertionError("An overridden getter must not execute.")

    class Derived(Base):
        @property
        @usd_field(UsdAttribute("physics:density", "PhysicsMassAPI"), extend=True)
        def value(self):
            return np.array([[2.5]])

        @property
        @usd_field(UsdAttribute("physics:centerOfMass", "PhysicsMassAPI"))
        def center(self):
            return np.array([[[0.1, -0.2, 0.3]]])

    class Replaced(Derived):
        @property
        @usd_field(UsdAttribute("custom:replacement", type_name="double"))
        def value(self):
            return np.array([[7.0]])

    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/Body").GetPrim()
    writer = UsdWriter(stage)
    writer.write_properties("/Body", None, Derived(), row=0)
    assert body.GetAttribute("physics:mass").Get() == 2.5
    assert body.GetAttribute("physics:density").Get() == 2.5
    np.testing.assert_allclose(body.GetAttribute("physics:centerOfMass").Get(), [0.1, -0.2, 0.3])
    writer.write_properties("/Body", None, Replaced(), row=0)
    assert body.GetAttribute("physics:mass").Get() == 2.5
    assert body.GetAttribute("custom:replacement").Get() == 7.0
    # Registered typed schemas validate the prim type; they are never applied as APIs.
    writer.write_attribute("/Body", UsdAttribute("visibility", "Imageable"), "invisible")
    assert body.GetAttribute("visibility").Get() == "invisible"


@pytest.mark.parametrize(
    "target, error",
    [
        (UsdAttribute("physics:mass", "PhysicsMassAPI:angular"), ValueError),
        (UsdAttribute("drive:angular:physics:stiffness", "PhysicsDriveAPI"), ValueError),
        (UsdAttribute("physics:unknown", "PhysicsMassAPI"), NotImplementedError),
        (UsdAttribute("custom:value", "UnknownAPI"), NotImplementedError),
        (UsdAttribute("physics:mass", "PhysicsMassAPI", type_name="float3"), ValueError),
        (UsdAttribute("custom:value", type_name="not_a_type"), ValueError),
        (UsdAttribute("custom:array", type_name="float[]"), NotImplementedError),
        (UsdAttribute("custom:matrix", type_name="matrix4d"), NotImplementedError),
    ],
)
def test_unsupported_binding_fails_explicitly(target, error):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Body")
    with pytest.raises(error):
        UsdWriter(stage).write_attribute("/Body", target, 1.0, axis="angular")


def test_required_backend_binding_and_property_shadow_fail():
    class Base:
        @property
        @usd_field()
        def friction(self):
            raise AssertionError("Discovery must not read friction.")

    with pytest.raises(NotImplementedError, match="Missing backend"):
        usd_fields(Base)

    class Backend(Base):
        @property
        @usd_field(UsdAttribute("custom:friction", type_name="float"))
        def friction(self):
            return np.array([[0.2]])

    class Shadow(Backend):
        friction = None

    assert set(usd_fields(Backend)) == {"friction"}
    with pytest.raises(NotImplementedError, match="shadowed"):
        usd_fields(Shadow)


def test_articulation_rejects_unsupported_driven_joint():
    from isaaclab.assets import BaseArticulation
    from isaaclab.sim.usd_export import AssetPaths

    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.SphericalJoint.Define(stage, "/Joint")
    asset = SimpleNamespace(cfg=ArticulationCfg(prim_path="/Robot", actuators={}), num_joints=1, data=None)
    writer = SimpleNamespace(
        stage=stage,
        adapter=SimpleNamespace(paths=lambda _: AssetPaths([], [("/Joint", 0)])),
        write_bodies=lambda *_: None,
    )
    with pytest.raises(NotImplementedError, match="Unsupported driven joint"):
        BaseArticulation.author_fixed_configuration(asset, writer)


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

    asset.author_fixed_configuration = lambda writer: BaseRigidObject.author_fixed_configuration(asset, writer)
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
    from isaaclab.actuators.actuator_control import _JOINT_PROPERTY_KEYS

    asset_fields = set().union(
        *(
            {field.name for field in fields(cfg)}
            for cfg in (AssetBaseCfg, ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg)
        )
    )
    assert asset_fields == set().union(*ASSET_CONFIGURATION_SOURCES.values())
    assert {field.name for field in fields(ActuatorBaseCfg)} == set(_JOINT_PROPERTY_KEYS).union(
        *ACTUATOR_CONFIGURATION_SOURCES.values()
    )
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
            UsdWriter(stage).save(str(output))
        assert output.read_text() == "existing destination"
        body.RemoveProperty(name)
    shader = UsdShade.Shader.Define(stage, "/Shared/Texture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(tmp_path / "missing.png")))
    with pytest.raises(RuntimeError, match="asset dependencies"):
        UsdWriter(stage).save(str(output))
    assert output.read_text() == "existing destination"
    stage.RemovePrim("/Shared/Texture")

    def failed_replace(*args):
        raise OSError("injected save failure")

    monkeypatch.setattr("isaaclab.sim.usd_export.os.replace", failed_replace)
    with pytest.raises(OSError, match="injected save failure"):
        UsdWriter(stage).save(str(output))
    assert output.read_text() == "existing destination"
    assert list(tmp_path.iterdir()) == [output]
