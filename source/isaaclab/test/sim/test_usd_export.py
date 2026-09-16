# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent USD preservation, fixed mapping and atomic-save contracts."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.assets.physics_properties import (
    UsdAttribute,
    source_units,
    usd_field,
    usd_fields,
)
from isaaclab.assets.physx_contact_data import PhysxContactData
from isaaclab.scene import InteractiveScene
from isaaclab.sim.usd_export import UsdWriter


def _stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    return stage


class BodyDeclarations:
    from isaaclab.assets import BaseRigidObjectData as _Data

    body_mass = _Data.body_mass
    body_inertia = _Data.body_inertia
    body_com_pos_b = _Data.body_com_pos_b


class BodyData(BodyDeclarations):
    def __init__(self, **values):
        self.values = values
        self.body_link_pose_w = values["body_link_pose_w"]

    @property
    def body_mass(self):
        return self.values["body_mass"]

    @property
    def body_inertia(self):
        return self.values["body_inertia"]

    @property
    def body_com_quat_b(self):
        return self.values["body_com_pose_b"].torch[..., 3:].numpy()

    @property
    def body_com_pos_b(self):
        return self.values["body_com_pose_b"].torch[..., :3].numpy()


@pytest.fixture
def scene():
    stage = _stage()
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
    return SimpleNamespace(
        sim=SimpleNamespace(stage=stage), num_envs=1, clone_plan=None, env_prim_paths=["/World/envs/env_0"]
    )


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
    UsdWriter(UsdWriter.from_stage(stage).stage).save(str(output))
    fresh = Usd.Stage.Open(str(output))
    assert {str(p.GetPath()) for p in fresh.Traverse()} == set(expected)
    for path, properties in expected.items():
        for name, value in properties.items():
            prop = fresh.GetPrimAtPath(path).GetProperty(name)
            assert (prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.Get()) == value
    assert stage.GetRootLayer().ExportToString() == before


def test_body_contacts_respect_nested_body_ownership(scene):
    stage = scene.sim.stage
    parent = "/World/envs/env_0/Body"
    child = parent + "/Child"
    child_prim = UsdGeom.Cube.Define(stage, child).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(child_prim)
    for path in (parent, child):
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path))
    writer = UsdWriter.from_stage(stage)
    for path, friction in ((parent, 0.25), (child, 0.75)):
        PhysxContactData(False, [[friction, friction, 0]], [0.02], [0]).author_configuration(writer, path)
    for path, friction in ((parent, 0.25), (child, 0.75)):
        material, _ = UsdShade.MaterialBindingAPI(writer.stage.GetPrimAtPath(path)).ComputeBoundMaterial("physics")
        assert UsdPhysics.MaterialAPI(material.GetPrim()).GetDynamicFrictionAttr().Get() == friction


@pytest.mark.parametrize(
    "purpose, inherited_purpose",
    [("", None), ("physics", None), ("custom", None), ("", ""), ("custom", "custom"), ("physics", "physics")],
)
def test_copy_removes_only_ineffective_dangling_material_bindings(purpose, inherited_purpose, tmp_path):
    stage = _stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
    body = UsdGeom.Xform.Define(stage, "/Root/Body").GetPrim()
    child = UsdGeom.Cube.Define(stage, "/Root/Body/Visual").GetPrim()
    material = UsdShade.Material.Define(stage, "/Material")
    UsdShade.MaterialBindingAPI.Apply(child).Bind(material, materialPurpose=purpose)
    if inherited_purpose is not None:
        UsdShade.MaterialBindingAPI.Apply(root).Bind(material, materialPurpose=inherited_purpose)
    UsdShade.MaterialBindingAPI.Apply(body)
    name = "material:binding" + (":" + purpose if purpose else "")
    body.CreateRelationship(name).SetTargets(["/Missing"])
    before = stage.GetRootLayer().ExportToString()
    purposes = ("", "preview", "full", "physics", "custom")
    expected = {
        (str(prim.GetPath()), purpose): UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(purpose)[0].GetPath()
        for prim in (root, body, child)
        for purpose in purposes
    }

    writer = UsdWriter.from_stage(stage)
    writer.remove_unused_bindings()
    output = str(tmp_path / "scene.usda")
    if inherited_purpose is None:
        writer.save(output)
        fresh = Usd.Stage.Open(output)
        assert not fresh.GetPrimAtPath(body.GetPath()).GetRelationship(name).GetTargets()
        for (path, purpose), material_path in expected.items():
            actual = UsdShade.MaterialBindingAPI(fresh.GetPrimAtPath(path)).ComputeBoundMaterial(purpose)[0]
            assert actual.GetPath() == material_path
    else:
        with pytest.raises(RuntimeError, match="Unresolved export dependency"):
            writer.save(output)
    assert stage.GetRootLayer().ExportToString() == before


def test_selected_variant_preserves_transitive_resources(scene):
    from isaaclab.cloner.clone_plan import ClonePlan

    stage = scene.sim.stage
    roots = [f"/World/envs/env_{index}" for index in range(3)]
    variant = UsdGeom.Sphere.Define(stage, roots[1] + "/Body")
    variant.CreateRadiusAttr().Set(0.42)
    UsdPhysics.RigidBodyAPI.Apply(variant.GetPrim())
    material = UsdShade.Material.Define(stage, roots[0] + "/Material")
    shader = UsdShade.Shader.Define(stage, roots[0] + "/Shader")
    shader.CreateIdAttr().Set("UsdPreviewSurface")
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(variant.GetPrim()).Bind(material)
    UsdGeom.Xform.Define(stage, roots[2])
    plan = ClonePlan(
        sources=(roots[0] + "/Body", roots[1] + "/Body"),
        destinations=("/World/envs/env_{}/Body",) * 2,
        clone_mask=np.array([[True, False, False], [False, True, True]]),
        env_ids=np.arange(3),
    )
    before = stage.GetRootLayer().ExportToString()
    writer = UsdWriter.from_stage(stage)
    writer.select_environment(plan, 2, roots)
    selected = writer.stage.GetPrimAtPath(roots[2] + "/Body")
    assert UsdGeom.Sphere(selected).GetRadiusAttr().Get() == pytest.approx(0.42)
    bound, _ = UsdShade.MaterialBindingAPI(selected).ComputeBoundMaterial()
    source = bound.GetSurfaceOutput().GetConnectedSource()[0].GetPrim()
    assert UsdShade.Shader(source).GetIdAttr().Get() == "UsdPreviewSurface"
    assert not writer.stage.GetPrimAtPath(roots[0])
    assert not writer.stage.GetPrimAtPath(roots[1])
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
    stage = _stage()
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
        @source_units("kg")
        @usd_field(UsdAttribute("physics:mass", "PhysicsMassAPI"))
        def value(self):
            return np.array([[2.5]])

    class Derived(Base):
        @property
        @usd_field(UsdAttribute("physics:density", "PhysicsMassAPI"), extend=True)
        def value(self):
            return super().value

    assert len(usd_fields(Derived)["value"]) == 2
    stage = _stage()
    body = UsdGeom.Xform.Define(stage, "/Body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    writer = UsdWriter(stage)
    writer.write_properties("/Body", None, Base(), row=0)
    assert body.GetAttribute("physics:mass").Get() == 2.5
    # A mass value cannot be reused as a density merely because both are scalar floats.
    with pytest.raises(ValueError, match="Incompatible physical units"):
        writer.write_properties("/Body", None, Derived(), row=0)
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
    stage = _stage()
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


def test_articulation_rejects_unsupported_driven_joint(monkeypatch):
    from isaaclab.assets import BaseArticulation
    from isaaclab.sim.usd_export import AssetPaths

    stage = _stage()
    UsdPhysics.SphericalJoint.Define(stage, "/Joint")
    monkeypatch.setattr("isaaclab.sim.usd_export.UsdWriter.write_bodies", lambda *_: None)
    asset = SimpleNamespace(cfg=ArticulationCfg(prim_path="/Robot", actuators={}), num_joints=1, data=None)
    asset._usd_export_paths = lambda env_index=0: AssetPaths([], [("/Joint", 0)])
    writer = SimpleNamespace(
        stage=stage,
        write_bodies=lambda *_: None,
        env_index=0,
        resolve_paths=lambda paths: paths,
    )
    with pytest.raises(NotImplementedError, match="Unsupported driven joint"):
        BaseArticulation.author_fixed_configuration(asset, writer)


def test_fixed_export_rejects_invalid_selection_and_poststep(scene, tmp_path):
    with pytest.raises(ValueError, match="outside"):
        InteractiveScene.export_to_usd(scene, str(tmp_path / "scene.usda"), env_id=1)
    scene.sim.get_physics_step_count = lambda: 1
    with pytest.raises(ValueError, match="first physics step"):
        InteractiveScene.export_to_usd(scene, str(tmp_path / "scene.usda"))


def test_unregistered_physical_body_fails_before_save(scene, tmp_path):
    import torch

    path = "/World/envs/env_0/Body"
    asset = SimpleNamespace(
        cfg=RigidObjectCfg(prim_path=path),
        root_view=SimpleNamespace(prim_paths=[path]),
        data=BodyData(
            body_link_pose_w=SimpleNamespace(torch=torch.tensor([[0.0, 0, 0, 0, 0, 0, 1]])),
            body_com_vel_w=SimpleNamespace(torch=torch.zeros((1, 6))),
            body_mass=SimpleNamespace(torch=torch.ones((1, 1))),
            body_inertia=SimpleNamespace(torch=torch.eye(3).reshape(1, 1, 9)),
            body_com_pose_b=SimpleNamespace(torch=torch.tensor([[[0.0, 0, 0, 0, 0, 0, 1]]])),
        ),
    )
    scene.articulations = {}
    scene.rigid_objects = {"body": asset}
    scene.rigid_object_collections = {}
    scene.deformable_objects = {}
    scene.cable_objects = {}
    scene.surface_grippers = {}
    scene.physics_scene_path = "/physicsScene"
    from isaaclab.assets import BaseRigidObject
    from isaaclab.sim.usd_export import AssetPaths

    asset.author_fixed_configuration = lambda writer: BaseRigidObject.author_fixed_configuration(asset, writer)
    from isaaclab.physics import PhysicsManager

    asset._usd_export_paths = lambda env_index=0: AssetPaths([(path, 0)], [])
    scene.sim.physics_manager = SimpleNamespace(author_fixed_configuration=PhysicsManager.author_fixed_configuration)
    scene.sim.get_physics_step_count = lambda: 0
    scene.sim.get_physics_dt = lambda: 1 / 60
    output = tmp_path / "complete.usda"
    output.write_text("previous destination")
    before = scene.sim.stage.GetRootLayer().ExportToString()
    with pytest.raises(RuntimeError, match="Incomplete body export"):
        InteractiveScene.export_to_usd(scene, str(output))
    assert output.read_text() == "previous destination"
    assert scene.sim.stage.GetRootLayer().ExportToString() == before


def test_atomic_save_rejects_missing_dependencies_and_preserves_destination(scene, tmp_path, monkeypatch):
    output = tmp_path / "scene.usda"
    output.write_text("existing destination")
    stage = UsdWriter.from_stage(scene.sim.stage).stage
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


@pytest.mark.parametrize("available", [True, False])
def test_dependency_validation_uses_authored_uri(available, monkeypatch):
    """A localization-only URI normalization must not reject a reachable resource."""
    from pxr import Sdf, UsdUtils

    from isaaclab.sim.usd_export import UsdWriter
    from isaaclab.utils import assets

    stage = _stage()
    uri = "https://assets.example.test/sky.hdr"
    stage.DefinePrim("/Light").CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(uri))
    monkeypatch.setattr(UsdUtils, "ComputeAllDependencies", lambda _: ([], [], ["https:/assets.example.test/sky.hdr"]))
    checked = []

    def check(path):
        checked.append(path)
        return 2 if available else 0

    monkeypatch.setattr(assets, "check_file_path", check)
    if available:
        UsdWriter(stage).validate_dependencies()
    else:
        with pytest.raises(RuntimeError, match="Unresolved export asset"):
            UsdWriter(stage).validate_dependencies()
    assert checked == [uri]
    assert stage.GetPrimAtPath("/Light").GetAttribute("inputs:texture:file").Get().path == uri


@pytest.mark.parametrize("length", [1.0, 0.01])
def test_multi_axis_joint_values_do_not_overwrite_other_axes(length):
    """Independent Cartesian DOFs convert compound units and retain axis identity."""
    from isaaclab.assets import BaseArticulationData

    class Declarations:
        joint_pos_limits = BaseArticulationData.joint_pos_limits
        joint_stiffness = BaseArticulationData.joint_stiffness

    class Data(Declarations):
        @property
        def joint_pos_limits(self):
            return np.array([[[-0.2, 0.2], [-0.7, 0.7], [-0.3, 0.4]]])

        @property
        def joint_stiffness(self):
            return np.array([[13.0, 29.0, 47.0]])

    stage = _stage()
    UsdGeom.SetStageMetersPerUnit(stage, length)
    joint = UsdPhysics.Joint.Define(stage, "/Joint")
    writer = UsdWriter(stage)
    for row, axis in enumerate(("rotX", "rotZ", "transY")):
        writer.write_properties("/Joint", axis, Data(), row=row)
    for axis, angle, stiffness in (("rotX", 0.2, 13.0), ("rotZ", 0.7, 29.0)):
        limit = UsdPhysics.LimitAPI(joint.GetPrim(), axis)
        assert limit.GetLowAttr().Get() == pytest.approx(-np.degrees(angle))
        assert limit.GetHighAttr().Get() == pytest.approx(np.degrees(angle))
        assert UsdPhysics.DriveAPI(joint.GetPrim(), axis).GetStiffnessAttr().Get() == pytest.approx(
            stiffness * np.pi / 180 / length**2
        )
    assert UsdPhysics.LimitAPI(joint.GetPrim(), "transY").GetLowAttr().Get() == pytest.approx(-0.3 / length)
    assert UsdPhysics.DriveAPI(joint.GetPrim(), "transY").GetStiffnessAttr().Get() == 47.0
    assert not joint.GetPrim().HasAttribute("physics:lowerLimit")


@pytest.mark.parametrize("length,mass_unit", [(1.0, 1.0), (0.01, 0.001)])
def test_fixed_properties_write_placement_and_zero_initial_velocities(scene, length, mass_unit):
    import torch

    from pxr import Gf

    from isaaclab.assets import BaseArticulationData

    UsdGeom.SetStageMetersPerUnit(scene.sim.stage, length)
    UsdPhysics.SetStageKilogramsPerUnit(scene.sim.stage, mass_unit)
    path = "/World/envs/env_0/Body"
    prim = scene.sim.stage.GetPrimAtPath(path)
    UsdGeom.XformCommonAPI(prim).SetTranslate((1, 2, 3))
    body = UsdPhysics.RigidBodyAPI(prim)
    body.CreateVelocityAttr().Set((4, 5, 6))
    body.CreateAngularVelocityAttr().Set((7, 8, 9))
    body.GetVelocityAttr().Set((10, 11, 12), 1)
    joint = UsdPhysics.RevoluteJoint.Define(scene.sim.stage, "/Joint").GetPrim()
    joint.CreateAttribute("state:angular:physics:position", Sdf.ValueTypeNames.Float).Set(12)
    joint.CreateAttribute("state:angular:physics:velocity", Sdf.ValueTypeNames.Float).Set(34)
    joint.CreateAttribute("newton:angular:velocity", Sdf.ValueTypeNames.Float).Set(56)
    points = UsdGeom.Mesh.Define(scene.sim.stage, "/Cloth")
    points.CreatePointsAttr().Set([(0, 0, 0), (1, 0, 0)])
    points.CreateVelocitiesAttr().Set([(1, 2, 3), (4, 5, 6)])
    points.GetVelocitiesAttr().Set([(2, 3, 4), (5, 6, 7)], 1)
    before = scene.sim.stage.GetRootLayer().ExportToString()
    data = BodyData(
        body_link_pose_w=SimpleNamespace(torch=torch.tensor([[[3.0, 2, 1, 0, 0, 0, 1]]])),
        body_mass=SimpleNamespace(torch=torch.tensor([[2.0]])),
        body_inertia=SimpleNamespace(torch=torch.eye(3).reshape(1, 1, 9)),
        body_com_pose_b=SimpleNamespace(torch=torch.tensor([[[0.0, 0, 0, 0, 0, 0, 1]]])),
    )
    writer = UsdWriter.from_stage(scene.sim.stage)
    writer.write_bodies(data, [(path, 0)])
    writer.clear_initial_velocities()
    actual = writer.stage.GetPrimAtPath(path)
    properties = UsdPhysics.MassAPI(actual)
    assert properties.GetMassAttr().Get() == pytest.approx(2.0 / mass_unit)
    np.testing.assert_allclose(properties.GetDiagonalInertiaAttr().Get(), np.ones(3) / mass_unit / length**2)
    velocities = UsdGeom.PointBased(writer.stage.GetPrimAtPath("/Cloth")).GetVelocitiesAttr()
    np.testing.assert_array_equal(np.asarray(velocities.Get()), np.zeros((2, 3)))
    assert velocities.GetTimeSamples() == []
    assert UsdGeom.XformCache().GetLocalToWorldTransform(actual).ExtractTranslation() == Gf.Vec3d(
        3 / length, 2 / length, 1 / length
    )
    for name in ("physics:velocity", "physics:angularVelocity"):
        assert actual.GetAttribute(name).Get() == Gf.Vec3f(0)
        assert actual.GetAttribute(name).GetTimeSamples() == []
    assert writer.stage.GetPrimAtPath("/Joint").GetAttribute("state:angular:physics:position").Get() == 12
    for name in ("state:angular:physics:velocity", "newton:angular:velocity"):
        assert writer.stage.GetPrimAtPath("/Joint").GetAttribute(name).Get() == 0

    class StateOnly:
        joint_pos = BaseArticulationData.joint_pos
        joint_vel = BaseArticulationData.joint_vel

    assert usd_fields(StateOnly) == {}
    assert scene.sim.stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize("preserve", [False, True])
def test_source_contacts_preserve_distinct_materials_and_automatic_offsets(scene, preserve):
    stage = scene.sim.stage
    body = "/World/envs/env_0/Body"
    for index, friction in enumerate((0.5, 1.0)):
        collider = UsdGeom.Sphere.Define(stage, f"{body}/Shape{index}")
        collider.CreateRadiusAttr().Set(0.1 * (index + 1))
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        material = UsdShade.Material.Define(stage, f"/Material{index}")
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr().Set(friction)
        physics.CreateDynamicFrictionAttr().Set(friction)
        UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(material, materialPurpose="physics")
    writer = UsdWriter.from_stage(stage)
    writer.preserve_source_contacts = preserve
    args = (body, True, [[0.5, 0.5, 0], [1, 1, 0]], [0.004, 0.008], [0, 0])
    if not preserve:
        with pytest.raises(NotImplementedError, match="Distinct per-shape"):
            PhysxContactData(*args[1:]).author_configuration(writer, args[0])
        return
    PhysxContactData(*args[1:]).author_configuration(writer, args[0])
    assert writer.stage.GetPrimAtPath(body).GetAttribute("physxRigidBody:disableGravity").Get()
    for index, friction in enumerate((0.5, 1.0)):
        collider = writer.stage.GetPrimAtPath(f"{body}/Shape{index}")
        assert not collider.GetAttribute("physxCollision:contactOffset").HasAuthoredValueOpinion()
        material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
        assert material.GetPath() == Sdf.Path(f"/Material{index}")
        assert UsdPhysics.MaterialAPI(material.GetPrim()).GetDynamicFrictionAttr().Get() == friction
