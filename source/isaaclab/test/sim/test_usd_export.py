# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-independent scene export contracts; no simulation application is required."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.assets.physics_properties import (
    ACTUATOR_CONFIGURATION_SOURCES,
    ASSET_CONFIGURATION_SOURCES,
    BodyPhysicsProperties,
    validate_configuration_coverage,
)
from isaaclab.cloner import ClonePlan
from isaaclab.sim.usd_export import (
    ArticulationExporter,
    ArticulationPrimPaths,
    SceneExporter,
    check_body_coverage,
    retain_stage_objects,
    save_environment_snapshot,
    write_body_properties,
)


class RegisteredScene(SimpleNamespace):
    def keys(self):
        return self.registry.keys()

    def __getitem__(self, name):
        return self.registry[name]


@pytest.fixture
def scene():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    material = UsdShade.Material.Define(stage, "/Shared/Material")
    UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateStaticFrictionAttr().Set(0.73)
    stage.DefinePrim("/Shared/Light", "DomeLight").CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(
        1500
    )
    for env, size in ((0, 0.2), (1, 0.7)):
        path = f"/World/envs/env_{env}"
        for name in ("Body", "SecondBody"):
            prim = UsdGeom.Cube.Define(stage, f"{path}/{name}").GetPrim()
            prim.GetAttribute("size").Set(size)
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(2 + env)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
        static = UsdGeom.Cube.Define(stage, f"{path}/Table")
        UsdPhysics.CollisionAPI.Apply(static.GetPrim())
    cfgs = {name: AssetBaseCfg(prim_path=f"/World/envs/env_.*/{name}") for name in ("Body", "SecondBody", "Table")}
    return RegisteredScene(
        sim=SimpleNamespace(stage=stage, get_physics_step_count=lambda: 0),
        num_envs=2,
        clone_plan=ClonePlan(
            sources=tuple(f"/World/envs/env_0/{name}" for name in cfgs),
            destinations=tuple("/World/envs/env_{}/" + name for name in cfgs),
            clone_mask=np.ones((3, 2), dtype=bool),
        ),
        env_prim_paths=["/World/envs/env_0", "/World/envs/env_1"],
        sensors={},
        visual_materials={},
        articulations={},
        rigid_objects={},
        rigid_object_collections={},
        deformable_objects={},
        cable_objects={},
        surface_grippers={},
        registry=cfgs,
    )


def test_scene_selection_preserves_all_authored_properties_and_shared_resources(scene):
    before = scene.sim.stage.GetRootLayer().ExportToString()
    expected = {
        str(prim.GetPath()): {
            str(prop.GetName()): prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.Get()
            for prop in prim.GetAuthoredProperties()
        }
        for prim in scene.sim.stage.Traverse()
        if not str(prim.GetPath()).startswith("/World/envs/env_0")
    }
    snapshot = SceneExporter(scene).create_snapshot(1)
    assert not snapshot.GetPrimAtPath("/World/envs/env_0")
    for path, properties in expected.items():
        prim = snapshot.GetPrimAtPath(path)
        assert prim, path
        for name, value in properties.items():
            prop = prim.GetProperty(name)
            actual = prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.Get()
            assert actual == value, (path, name)
    assert scene.sim.stage.GetRootLayer().ExportToString() == before
    check_body_coverage(snapshot, {"/World/envs/env_1/Body", "/World/envs/env_1/SecondBody"})
    with pytest.raises(RuntimeError, match="Incomplete body export"):
        check_body_coverage(snapshot, {"/World/envs/env_1/Body"})


def test_fixed_snapshot_rejects_multiple_environments_and_post_step_state(scene):
    exporter = SceneExporter(scene, fixed_configuration=True)
    with pytest.raises(ValueError, match="exactly one"):
        exporter.create_snapshot()
    scene.num_envs = 1
    scene.sim.get_physics_step_count = lambda: 1
    with pytest.raises(ValueError, match="first physics step"):
        exporter.create_snapshot()


def test_asset_scope_keeps_transitive_dependencies_without_other_bodies(scene, tmp_path, monkeypatch):
    stage = scene.sim.stage
    body = stage.GetPrimAtPath("/World/envs/env_0/Body")
    body.CreateRelationship("physics:filteredPairs").SetTargets(["/World/envs/env_1/Body"])
    retain_stage_objects(stage, [Sdf.Path("/World/envs/env_0/Body")])
    assert body.GetRelationship("physics:filteredPairs").GetTargets() == []
    assert stage.GetPrimAtPath("/Shared/Material")
    assert stage.GetPrimAtPath("/physicsScene")
    assert not stage.GetPrimAtPath("/World/envs/env_0/SecondBody")
    assert not stage.GetPrimAtPath("/World/envs/env_1")
    # A single-link articulation's common body prefix does not include sibling controller prims.
    root = "/World/Robot"
    prim = stage.DefinePrim(f"{root}/Base", "Xform")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    stage.DefinePrim(f"{root}/Controller", "Scope")
    asset = SimpleNamespace(stage=stage, cfg=ArticulationCfg(prim_path=root))
    exporter = ArticulationExporter(asset, lambda *_: ArticulationPrimPaths([f"{root}/Base"], []))
    monkeypatch.setattr(exporter, "write_to_stage", lambda *args, **kwargs: None)
    exported = Usd.Stage.Open(exporter.export(str(tmp_path / "robot.usda")))
    assert exported.GetPrimAtPath(f"{root}/Controller")
    assert not exported.GetPrimAtPath("/World/envs")


def test_body_contract_joins_by_identity_and_converts_inertia_units(scene):
    stage = scene.sim.stage
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 0.001)
    tensors = {
        "body_mass": [[[2.0], [7.0]]],
        "body_inertia": [[np.diag([0.1, 0.2, 0.3]).reshape(-1).tolist(), [[0.8, 0.1, 0], [0.1, 0.9, 0], [0, 0, 1.0]]]],
        "body_com_pose_b": [[[0.01, 0.02, 0.03, 0, 0, 0, 1], [0.04, 0.05, 0.06, 0, 0, 0, 1]]],
        "body_link_pose_w": [[[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 0, 1]]],
        "body_com_vel_w": [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1, 2, 3, 4, 5, 6]]],
    }
    tensors["body_inertia"][0][1] = np.asarray(tensors["body_inertia"][0][1]).reshape(-1).tolist()
    data = SimpleNamespace(**{name: SimpleNamespace(torch=torch.tensor(value)) for name, value in tensors.items()})
    properties = BodyPhysicsProperties.from_data(data, 0)
    paths = ["/World/envs/env_0/Body", "/World/envs/env_0/SecondBody"]
    write_body_properties(stage, paths, properties, [1, 0])
    for path, row in zip(paths, [1, 0]):
        api = UsdPhysics.MassAPI(stage.GetPrimAtPath(path))
        assert api.GetMassAttr().Get() == pytest.approx(properties.mass[row] / 0.001)
        axes = Gf.Matrix3d(api.GetPrincipalAxesAttr().Get())
        rotation = np.asarray(axes).T
        reconstructed = rotation @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ rotation.T * 1e-7
        np.testing.assert_allclose(reconstructed, properties.inertia[row], rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(api.GetCenterOfMassAttr().Get(), properties.com_pose[row, :3] / 0.01)
    # Fixed configuration retains even mass properties that have no special-case exporter field.
    mass_before = UsdPhysics.MassAPI(stage.GetPrimAtPath(paths[0])).GetMassAttr().Get()
    write_body_properties(stage, paths, properties, preserve_authored_mass=True)
    assert UsdPhysics.MassAPI(stage.GetPrimAtPath(paths[0])).GetMassAttr().Get() == mass_before


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
    stage = SceneExporter(scene).create_snapshot(1)
    body = stage.GetPrimAtPath("/World/envs/env_1/Body")
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
            save_environment_snapshot(stage, str(output))
        assert output.read_text() == "existing destination"
        body.RemoveProperty(name)
    shader = UsdShade.Shader.Define(stage, "/Shared/Texture")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(tmp_path / "missing.png")))
    with pytest.raises(RuntimeError, match="asset dependencies"):
        save_environment_snapshot(stage, str(output))
    assert output.read_text() == "existing destination"
    stage.RemovePrim("/Shared/Texture")

    def failed_replace(*args):
        raise OSError("injected save failure")

    monkeypatch.setattr("isaaclab.sim.usd_export.os.replace", failed_replace)
    with pytest.raises(OSError, match="injected save failure"):
        save_environment_snapshot(stage, str(output))
    assert output.read_text() == "existing destination"
    assert list(tmp_path.iterdir()) == [output]
