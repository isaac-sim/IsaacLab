# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less tests of how the USD export maps engine values and actuators to prims."""

import contextlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.sim.usd_export.base_usd_exporter import (
    _DEFINE_PRIM,
    JOINT_DATA_FIELDS,
    BaseUsdExporter,
    PrimDefinition,
    UsdEdit,
)
from isaaclab.sim.usd_export.physx_usd_exporter import PhysxUsdExporter
from isaaclab.utils import assets as assets_utils

pytestmark = pytest.mark.kitless


class _Exporter(PhysxUsdExporter):
    """Reads the live collider values of rigid objects from ``colliders``, keyed by prim path."""

    colliders: dict[str, dict[str, np.ndarray]] = {}

    @staticmethod
    def _resolve_engine_path(prim_path, articulation_cfg=None):
        return prim_path

    @contextlib.contextmanager
    def _open_engine_view(self, path_expr, articulation):
        yield path_expr

    def _read_collider_values(self, view):
        return {name: np.asarray(value)[None] for name, value in self.colliders[view].items()}


def _engine(**values):
    """Engine values of one environment, as the scene keeps them from startup."""
    return [{name: np.asarray(value, dtype=float)[None] for name, value in values.items()}]


def _articulation_engine(articulation):
    """What the engine loaded for an articulation: zeros, so every nonzero live value is Isaac Lab's."""
    data = articulation.data
    dofs = data.joint_stiffness.torch.shape[1]
    zeros = {name: np.zeros(dofs) for name in JOINT_DATA_FIELDS}
    bodies = data.body_mass.torch.shape[1]
    return _engine(
        **zeros,
        joint_pos_limits_lower=np.zeros(dofs),
        joint_pos_limits_upper=np.zeros(dofs),
        body_mass=np.zeros(bodies),
        body_com_pos_b=np.zeros((bodies, 3)),
        body_inertia=np.zeros((bodies, 9)),
    )


def _articulation(joint_names, body_names):
    """An articulation of one environment whose data names joints and bodies without their prim paths."""

    def rows(*shape):
        return SimpleNamespace(torch=torch.zeros(1, *shape))

    dofs, bodies = len(joint_names), len(body_names)
    data = SimpleNamespace(
        joint_pos_limits=rows(dofs, 2),
        **{
            name: rows(dofs)
            for name in (
                "joint_stiffness",
                "joint_damping",
                "joint_effort_limits",
                "joint_armature",
                "joint_vel_limits",
                "joint_friction_coeff",
                "joint_dynamic_friction_coeff",
                "joint_viscous_friction_coeff",
                "default_joint_pos",
                "default_joint_vel",
            )
        },
        body_mass=rows(bodies),
        body_com_pose_b=rows(bodies, 7),
        body_inertia=rows(bodies, 9),
    )
    cfg = SimpleNamespace(prim_path="/Robot")
    names = dict(joint_names=joint_names, body_names=body_names)
    return SimpleNamespace(cfg=cfg, **names, backend_joint_names=joint_names, backend_body_names=body_names, data=data)


def _scene(base=None, **assets):
    """A scene of one environment, before its first physics step; articulations load as zeros unless ``base``."""
    assets = {
        "articulations": {},
        "rigid_objects": {},
        "rigid_object_collections": {},
        "deformable_objects": {},
        **assets,
    }
    base = {name: _articulation_engine(a) for name, a in assets["articulations"].items()} | (base or {})
    sim = SimpleNamespace(
        get_physics_step_count=lambda: 0, backend_manager=SimpleNamespace(get_gravity=lambda: (0.0, 0.0, -9.81))
    )
    return SimpleNamespace(num_envs=1, sim=sim, physics_scene_path="/physicsScene", _usd_export_base=base, **assets)


def _stage():
    """An in-memory metric stage whose physics scene states Earth gravity down the Z axis."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    physics_scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    return stage


def _rigid_object(path, mass, com=(0.0, 0.0, 0.0), inertia=np.eye(3)):
    """A rigid object whose data holds one row per mass in ``mass``, or one row for a single mass."""
    masses = torch.tensor(mass, dtype=torch.float32).reshape(-1, 1)
    rows = len(masses)
    data = SimpleNamespace(
        body_mass=SimpleNamespace(torch=masses),
        body_com_pose_b=SimpleNamespace(torch=torch.tensor([*com, 0.0, 0.0, 0.0, 1.0]).repeat(rows, 1, 1)),
        body_inertia=SimpleNamespace(
            torch=torch.tensor(inertia, dtype=torch.float32).reshape(1, 1, 9).repeat(rows, 1, 1)
        ),
    )
    return SimpleNamespace(cfg=SimpleNamespace(prim_path=path), data=data)


@pytest.mark.parametrize(
    "joint_names,body_names,name",
    [([], ["Base", "Foot"], "Foot"), (["Ankle"], ["Base"], "Ankle")],
    ids=["link name matches two prims", "joint name matches no prim"],
)
def test_names_must_match_one_prim(joint_names, body_names, name):
    """A name that matches no prim or several prims fails the export instead of writing another prim's values."""
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage, "/Robot").GetPrim())
    for link in ("/Robot/Base", "/Robot/Left/Foot", "/Robot/Right/Foot"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, link).GetPrim())
    scene = _scene(articulations={"robot": _articulation(joint_names, body_names)})

    with pytest.raises(RuntimeError, match=f"one prim named '{name}'"):
        _Exporter(scene).compute_diff(stage)


def test_a_pd_actuator_of_a_multi_axis_joint_stays_an_actuator_prim():
    """A USD drive is per axis, so a PD actuator of a joint other than a revolute or prismatic one stays its prim."""
    stage = Usd.Stage.CreateInMemory()
    joint = UsdPhysics.SphericalJoint.Define(stage, "/Robot/Joint").GetPrim()
    actuator = stage.DefinePrim("/Robot/Actuator", "NewtonActuator")
    actuator.AddAppliedSchema("NewtonPDControlAPI")
    actuator.CreateRelationship("newton:targets").SetTargets([joint.GetPath()])
    actuator.CreateAttribute("newton:kp", Sdf.ValueTypeNames.Float).Set(80.0)

    assert _Exporter(_scene())._compute_pd_drive_edits(stage) == []


def test_reordered_joints_compare_with_their_own_engine_values():
    """Isaac Lab may order joints differently from the engine; each joint is compared with its own loaded value."""
    stage = _stage()
    for name in ("Base", "Link"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/Robot/{name}").GetPrim())
    for name in ("A", "B"):
        UsdPhysics.DriveAPI.Apply(UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/{name}").GetPrim(), "angular")
    robot = _articulation(["A", "B"], ["Base", "Link"])
    robot.backend_joint_names = ["B", "A"]
    robot.data.joint_stiffness.torch[0] = torch.tensor([5.0, 7.0])  # A as loaded, B changed
    base = _articulation_engine(robot)
    base[0]["joint_stiffness"][0] = [0.0, 5.0]  # engine order: B, A
    scene = _scene({"robot": base}, articulations={"robot": robot})

    edits = {edit.path: edit.value for edit in _Exporter(scene).compute_diff(stage) if "stiffness" in edit.attribute}

    assert edits == {"/Robot/B": pytest.approx(7.0 * np.pi / 180.0)}


def test_values_authored_without_their_api_schema_are_written_live_with_it():
    """PhysX ignores an attribute whose API schema the prim lacks, so the engine loaded the schema's default.

    An edit that applies the schema puts every authored attribute of it in effect, so the export states those live
    (joint ``Ignored``, collider of ``/Box``); an authored value equal to the live one is written with its schema
    (joint ``Configured``).
    """
    stage = _stage()
    for name in ("Base", "Link"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/Robot/{name}").GetPrim())
    for name in ("Ignored", "Configured"):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/{name}").GetPrim()
        joint.CreateAttribute("physxJointAxis:angular:armature", Sdf.ValueTypeNames.Float).Set(0.5)
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Box").GetPrim())
    collider = UsdGeom.Cube.Define(stage, "/Box/Collider").GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider)
    collider.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(5.0)
    robot = _articulation(["Ignored", "Configured"], ["Base", "Link"])
    robot.data.joint_vel_limits.torch[0] = torch.tensor([2.0, 0.0])  # loaded as 0, so written with the schema
    robot.data.joint_armature.torch[0] = torch.tensor([0.0, 0.5])  # loaded as 0 (ignored)
    live = {"material": np.array([[0.5, 0.5, 0.0]]), "contact_offset": [0.02], "rest_offset": [0.004]}
    mass = {"body_mass": [1.0], "body_com_pos_b": [np.zeros(3)], "body_inertia": [np.eye(3).ravel()]}
    scene = _scene(
        {"box": _engine(**mass, **live | {"rest_offset": [0.0]})},
        articulations={"robot": robot},
        rigid_objects={"box": _rigid_object("/Box", 1.0)},
    )
    exporter = _Exporter(scene)
    exporter.colliders = {"/Box": live}

    edits = exporter.compute_diff(stage)

    def written(path):
        return {edit.attribute: edit.value for edit in edits if edit.path == path}

    assert written("/Robot/Ignored")["physxJointAxis:angular:armature"] == pytest.approx(0.0)
    assert "physxJointAxis:angular:maxJointVelocity" in written("/Robot/Ignored")
    configured = [edit for edit in edits if edit.path == "/Robot/Configured" and "armature" in edit.attribute]
    assert len(configured) == 1 and configured[0].schema == "PhysxJointAxisAPI:angular"
    assert written("/Box/Collider") == {
        "physxCollision:restOffset": pytest.approx(0.004),
        "physxCollision:contactOffset": pytest.approx(0.02),
    }


def test_axes_of_a_rotational_d6_joint_map_to_its_usd_axes():
    """PhysX names the DOFs of a multi-axis joint "<joint>:<n>" after the axis index (0 rotX, 1 rotY, 2 rotZ)."""
    stage = _stage()
    for name in ("Base", "Link"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/Robot/{name}").GetPrim())
    joint = UsdPhysics.Joint.Define(stage, "/Robot/Waist").GetPrim()
    limits = {"transX": (1, -1), "transY": (1, -1), "transZ": (1, -1), "rotX": (-45, 45), "rotY": (1, -1)}
    for axis, (low, high) in limits.items():  # rotZ has no limit: free
        limit = UsdPhysics.LimitAPI.Apply(joint, axis)
        limit.CreateLowAttr(low)
        limit.CreateHighAttr(high)
    for axis in ("rotX", "rotZ"):
        UsdPhysics.DriveAPI.Apply(joint, axis)
    robot = _articulation(["Waist:0", "Waist:2"], ["Base", "Link"])
    robot.data.joint_pos_limits.torch[0] = torch.deg2rad(torch.tensor([[-45.0, 45.0], [-60.0, 30.0]]))
    robot.data.joint_effort_limits.torch.fill_(150.0)
    robot.data.joint_armature.torch[0] = torch.tensor([0.01, 0.02])
    scene = _scene(articulations={"robot": robot})

    edits = {edit.attribute: edit.value for edit in _Exporter(scene).compute_diff(stage) if edit.path == "/Robot/Waist"}

    assert "limit:rotX:physics:low" not in edits  # the USD states DOF 0's limits
    assert edits["limit:rotZ:physics:low"] == pytest.approx(-60.0)
    assert edits["drive:rotX:physics:maxForce"] == pytest.approx(150.0)
    assert edits["drive:rotZ:physics:maxForce"] == pytest.approx(150.0)
    assert edits["physxJointAxis:rotX:armature"] == pytest.approx(0.01)
    assert edits["physxJointAxis:rotZ:armature"] == pytest.approx(0.02)


def test_a_changed_joint_of_a_type_not_exported_is_refused():
    """A joint other than revolute, prismatic and rotational D6 keeps its USD values, so changing it is refused."""
    stage = Usd.Stage.CreateInMemory()
    for name in ("Base", "Link"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/Robot/{name}").GetPrim())
    UsdPhysics.SphericalJoint.Define(stage, "/Robot/Ball")
    robot = _articulation(["Ball:0", "Ball:1", "Ball:2"], ["Base", "Link"])
    scene = _scene(articulations={"robot": robot})

    assert not [edit for edit in _Exporter(scene).compute_diff(stage) if edit.path == "/Robot/Ball"]
    robot.data.joint_armature.torch.fill_(0.01)
    with pytest.raises(NotImplementedError, match="Joint '/Robot/Ball' changed"):
        _Exporter(scene).compute_diff(stage)


@pytest.mark.parametrize(
    "meters_per_unit,stated,live,written",
    [
        (1.0, None, (0.0, 0.0, -9.81), False),
        (0.01, None, (0.0, 0.0, -9.81), False),
        (1.0, ((0.0, 0.0, -1.0), 9.81), (0.0, 0.0, -9.81), False),
        (1.0, None, (0.2, -0.1, -4.0), True),
    ],
    ids=[
        "unset resolves to the live gravity",
        "unset on a centimeter stage",
        "stated equals the live gravity",
        "live gravity differs",
    ],
)
def test_gravity_is_written_only_where_the_scene_resolves_to_another(meters_per_unit, stated, live, written):
    """An unset scene gravity is Earth gravity down the up axis, as importers read it, so it is not restated."""
    from pxr import Gf

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    if stated is not None:
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(*stated[0]))
        physics_scene.CreateGravityMagnitudeAttr(stated[1])
    edits = _Exporter(_scene())._compute_gravity_edits(stage, live)

    assert bool(edits) == written


def test_only_values_that_differ_from_what_the_engine_loaded_are_written():
    """The engine loads what the stage states and derives the rest, as a PhysX import of the file does.

    A value Isaac Lab did not change is therefore not written, whether the USD states it or the engine derives it.
    """
    stage = _stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    for name in ("Unchanged", "Changed"):
        body = UsdGeom.Xform.Define(stage, f"/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, f"/{name}/Collider").GetPrim())
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/Changed")).CreateMassAttr().Set(1.0)
    stage.GetPrimAtPath("/Changed/Collider").CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(
        0.0
    )

    def rigid_object(path):
        return _rigid_object(path, 3.0, com=(0.1, 0.0, 0.0), inertia=2.0 * np.eye(3))

    live = {"material": np.array([[0.5, 0.5, 0.0]]), "contact_offset": [0.02], "rest_offset": [0.001]}
    loaded = {
        "body_mass": [3.0],
        "body_com_pos_b": [[0.1, 0.0, 0.0]],
        "body_inertia": [(2.0 * np.eye(3)).ravel()],
        **live,
    }
    base = {"unchanged": _engine(**loaded), "changed": _engine(**loaded | {"body_mass": [1.0], "rest_offset": [0.0]})}
    scene = _scene(base, rigid_objects={"unchanged": rigid_object("/Unchanged"), "changed": rigid_object("/Changed")})
    exporter = _Exporter(scene)
    exporter.colliders = {"/Unchanged": live, "/Changed": live}

    edits = {(edit.path, edit.attribute): edit.value for edit in exporter.compute_diff(stage)}

    # The changed mass also states the inertia, which an import would otherwise derive from it.
    assert edits.keys() == {
        ("/Changed", "physics:mass"),
        ("/Changed", "physics:diagonalInertia"),
        ("/Changed", "physics:principalAxes"),
        ("/Changed/Collider", "physxCollision:restOffset"),
    }
    assert edits["/Changed", "physics:mass"] == pytest.approx(3.0)


def test_collider_values_its_shapes_disagree_on_are_written_only_if_unchanged():
    """Shapes that report different values of a collider property keep their USD values; a change of such values
    is refused, since the engine reports them without their prims.

    PhysX derives an unset contact offset per collider, so a body with several colliders reports several.
    """
    stage = _stage()
    table = UsdGeom.Xform.Define(stage, "/Table").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(table)
    UsdPhysics.MassAPI.Apply(table).CreateMassAttr().Set(1.0)
    for name in ("Top", "Legs"):
        collider = UsdGeom.Cube.Define(stage, f"/Table/{name}").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider)
        for attribute in ("physxCollision:contactOffset", "physxCollision:restOffset"):
            collider.CreateAttribute(attribute, Sdf.ValueTypeNames.Float).Set(0.0)
    material = np.full((3, 3), 0.5)
    live = {"material": material, "contact_offset": [0.002, 0.016, 0.016], "rest_offset": np.full(3, 0.001)}
    loaded = {"material": material, "contact_offset": [0.002, 0.016, 0.016], "rest_offset": np.zeros(3)}
    base = {"table": _engine(body_mass=[1.0], body_com_pos_b=[np.zeros(3)], body_inertia=[np.eye(3).ravel()], **loaded)}
    scene = _scene(base, rigid_objects={"table": _rigid_object("/Table", 3.0)})
    exporter = _Exporter(scene)
    exporter.colliders = {"/Table": live}

    edits = {(edit.path, edit.attribute): edit.value for edit in exporter.compute_diff(stage)}

    assert edits["/Table", "physics:mass"] == pytest.approx(3.0)
    for collider in ("/Table/Top", "/Table/Legs"):
        assert (collider, "physxCollision:contactOffset") not in edits
        assert edits[collider, "physxCollision:restOffset"] == pytest.approx(0.001)
    live["contact_offset"] = [0.003, 0.016, 0.016]
    with pytest.raises(NotImplementedError, match="contact_offset of body '/Table' changed"):
        exporter.compute_diff(stage)


def test_a_collider_without_a_material_gets_one_stating_its_values():
    """Material values of a collider without a physics material go on a new material bound to it, at an unused path;
    colliders with other values get another new material. A collider bound only to a render material, which an
    importer does not read for physics, has none.

    A body without colliders, which reports empty per-shape arrays, gets no collider values.
    """
    stage = _stage()
    stage.DefinePrim("/PhysicsMaterials/Material_0", "Material")
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Box").GetPrim())
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, "/Box/Collider").GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Other").GetPrim())
    other_collider = UsdGeom.Cube.Define(stage, "/Other/Collider").GetPrim()
    UsdPhysics.CollisionAPI.Apply(other_collider)
    UsdShade.MaterialBindingAPI.Apply(other_collider).Bind(UsdShade.Material.Define(stage, "/Looks/Render"))
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Bare").GetPrim())
    live = {"material": np.array([[0.8, 0.6, 0.1]]), "contact_offset": [0.02], "rest_offset": [0.0]}
    loaded = {"body_mass": [1.0], "body_com_pos_b": [np.zeros(3)], "body_inertia": [np.eye(3).ravel()], **live}
    bare = {"material": np.zeros((0, 3)), "contact_offset": np.zeros(0), "rest_offset": np.zeros(0)}
    base = {
        "box": _engine(**loaded | {"material": np.array([[0.5, 0.5, 0.0]])}),
        "other": _engine(**loaded | {"material": np.array([[0.5, 0.5, 0.0]])}),
        "bare": _engine(**loaded | bare | {"body_mass": [2.0]}),
    }
    rigid_objects = {
        name: _rigid_object(path, 1.0) for name, path in (("box", "/Box"), ("other", "/Other"), ("bare", "/Bare"))
    }
    scene = _scene(base, rigid_objects=rigid_objects)
    exporter = _Exporter(scene)
    other = live | {"material": np.array([[0.9, 0.9, 0.0]])}
    exporter.colliders = {"/Box": live, "/Other": other, "/Bare": bare}

    edits = exporter.compute_diff(stage)
    for edit in edits:
        edit.apply(stage)

    material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/Box/Collider")).ComputeBoundMaterial("physics")
    assert material.GetPath() == "/PhysicsMaterials/Material_1"
    assert not stage.GetPrimAtPath("/PhysicsMaterials/Material_0").GetAuthoredAttributes()
    assert {edit.path for edit in edits if edit.path.startswith("/Bare")} == {"/Bare"}  # its changed mass only
    values = {
        name: material.GetPrim().GetAttribute(name).Get()
        for name in ("physics:staticFriction", "physics:dynamicFriction", "physics:restitution")
    }
    assert values == pytest.approx(
        {"physics:staticFriction": 0.8, "physics:dynamicFriction": 0.6, "physics:restitution": 0.1}
    )
    other, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/Other/Collider")).ComputeBoundMaterial("physics")
    assert other.GetPath() == "/PhysicsMaterials/Material_2"
    assert other.GetPrim().GetAttribute("physics:dynamicFriction").Get() == pytest.approx(0.9)


def test_values_of_a_prim_an_edit_defines_are_kept():
    """A prim an edit defines holds that edit's values, though another edit writes an attribute of the same name, as
    the gains of a velocity actuator the Newton export adds next to the asset's own MuJoCo actuators."""
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Robot/AssetActuator", "MjcActuator")
    gain = [80.0] + [0.0] * 9
    actuator = PrimDefinition(
        "MjcActuator", {"mjc:target": ["/Robot/Joint"]}, {"mjc:gainPrm": (Sdf.ValueTypeNames.DoubleArray, gain)}
    )
    edits = [
        UsdEdit("/Robot/AssetActuator", "mjc:gainPrm", [50.0] + [0.0] * 9, "double[]"),
        UsdEdit("/Robot/VelocityActuator", _DEFINE_PRIM, actuator),
    ]

    BaseUsdExporter._apply_edits(stage, edits)

    assert list(stage.GetPrimAtPath("/Robot/VelocityActuator").GetAttribute("mjc:gainPrm").Get()) == gain


def test_relative_references_of_a_scene_opened_from_a_file_resolve_in_another_directory(tmp_path):
    """The export copies the scene's root layer, so its relative asset paths are anchored to the scene's file."""
    (tmp_path / "scene").mkdir()
    robot = Usd.Stage.CreateNew(str(tmp_path / "scene" / "robot.usda"))
    robot.SetDefaultPrim(robot.DefinePrim("/Robot", "Xform"))
    robot.Save()
    stage = Usd.Stage.CreateNew(str(tmp_path / "scene" / "scene.usda"))
    stage.DefinePrim("/World/Robot").GetReferences().AddReference("./robot.usda")
    exporter = _Exporter(_scene(stage=stage, cable_objects={}))
    exporter._reduce_stage_to_environment = lambda stage: None
    exporter.compute_diff = lambda stage: []

    exported = Usd.Stage.Open(exporter.export(str(tmp_path / "export.usda")))

    assert exported.GetPrimAtPath("/World/Robot").GetTypeName() == "Xform"


def test_a_cached_remote_asset_is_referenced_by_its_url_without_opening_it(tmp_path, capfd, caplog):
    """The export names a cached remote asset by its URL in a copy no stage composes, so nothing opens the URL.

    An import resolves the URL only with a resolver that fetches it, so the export warns about it.
    """
    url = "https://example.com/Robots/robot.usda"
    cached = Path(assets_utils._mirror_path(url, str(tmp_path / "cache")))
    cached.parent.mkdir(parents=True)
    robot = Usd.Stage.CreateNew(str(cached))
    robot.SetDefaultPrim(robot.DefinePrim("/Robot", "Xform"))
    robot.Save()
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Robot").GetReferences().AddReference(str(cached))
    exporter = _Exporter(_scene(stage=stage, cable_objects={}))
    exporter._reduce_stage_to_environment = lambda stage: None
    held = []  # an engine import may keep the stage it read alive past the diff
    exporter.compute_diff = lambda stage: held.append(stage) or []

    exported = Path(exporter.export(str(tmp_path / "export.usda")))

    assert f"@{url}@" in exported.read_text()
    assert "Could not open asset" not in capfd.readouterr().err
    assert url in caplog.text


def test_assets_are_read_from_the_exported_environments_row_or_their_one_row():
    """An asset in the environments has one engine row per environment, of which the exported one's is read; an asset
    outside the environment namespace has one row, whichever environment is exported. Other row counts are refused."""
    stage = _stage()
    for path in ("/Shared", "/World/envs/env_2/Box"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, path).GetPrim())
        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(path)).CreateMassAttr().Set(1.0)
    live = {"material": np.zeros((0, 3)), "contact_offset": np.zeros(0), "rest_offset": np.zeros(0)}
    loaded = {"body_mass": [1.0], "body_com_pos_b": [np.zeros(3)], "body_inertia": [np.eye(3).ravel()], **live}
    box = "/World/envs/env_.*/Box"
    rigid_objects = {"shared": _rigid_object("/Shared", 3.0), "box": _rigid_object(box, [1.0, 2.0, 4.0])}
    scene = _scene({"shared": _engine(**loaded), "box": _engine(**loaded)}, rigid_objects=rigid_objects)
    scene.num_envs = 3
    exporter = _Exporter(scene, env_id=2)
    exporter.colliders = {"/Shared": live, box: live}

    edits = {(edit.path, edit.attribute): edit.value for edit in exporter.compute_diff(stage)}

    assert edits["/Shared", "physics:mass"] == pytest.approx(3.0)
    assert edits["/World/envs/env_2/Box", "physics:mass"] == pytest.approx(4.0)
    rigid_objects["box"] = _rigid_object(box, [1.0, 2.0])
    with pytest.raises(NotImplementedError, match="2 instances for 3 environments"):
        exporter.compute_diff(stage)


def test_an_environment_cloned_in_physics_only_is_exported_from_its_prototype():
    """An environment the cloner created in physics only has no prim; its export defines it and places it."""
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.cloner import make_clone_plan
    from isaaclab.sim import SpawnerCfg

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage, "/World/envs/env_0/Box")
    box = AssetBaseCfg(prim_path="/World/envs/env_.*/Box", spawn=SpawnerCfg(spawn_path="/World/envs/env_0/Box"))
    scene = _scene()
    scene.num_envs = 2
    scene.env_prim_paths = ["/World/envs/env_0", "/World/envs/env_1"]
    scene.clone_plan = make_clone_plan([box], [[0]], 2, positions=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))

    _Exporter(scene, env_id=1)._reduce_stage_to_environment(stage)

    exported = stage.GetPrimAtPath("/World/envs/env_1/Box")
    assert exported.IsDefined() and exported.GetTypeName() == "Cube"
    assert not stage.GetPrimAtPath("/World/envs/env_0")
    assert UsdGeom.XformCache().GetLocalToWorldTransform(exported).ExtractTranslation()[0] == pytest.approx(3.0)


def test_environments_a_weaker_layer_defines_are_left_out():
    """Only the exported environment remains, also where a sublayer defines the other environments."""
    weaker = Usd.Stage.CreateInMemory()
    for index in range(2):
        weaker.DefinePrim(f"/World/envs/env_{index}", "Xform")
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths.append(weaker.GetRootLayer().identifier)
    scene = _scene()
    scene.num_envs, scene.clone_plan = 2, None
    scene.env_prim_paths = ["/World/envs/env_0", "/World/envs/env_1"]

    _Exporter(scene)._reduce_stage_to_environment(stage)

    assert [str(prim.GetPath()) for prim in stage.Traverse()] == ["/World", "/World/envs", "/World/envs/env_0"]


def test_temporary_files_of_the_same_name_keep_their_own_copies(tmp_path):
    """Files the run wrote to the temporary directory are copied next to the export, each under a name of its own."""
    sources = []
    for folder, contents in (("a", "first"), ("b", "second")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "network.pt").write_text(contents)
        sources.append(str(tmp_path / folder / "network.pt"))
    export_path = str(tmp_path / "export" / "exported.usda")

    copies = [BaseUsdExporter._copy_temporary_file(source, export_path) for source in sources]

    assert [(tmp_path / "export" / copy).read_text() for copy in copies] == ["first", "second"]


def test_scenes_with_deformable_or_cable_objects_are_refused():
    """Deformable and cable values are not exported, so the export raises instead of writing a file that differs."""
    for assets in ({"deformable_objects": {"cloth": object()}}, {"cable_objects": {"cable": object()}}):
        scene = _scene(**assets)
        scene.deformable_objects = assets.get("deformable_objects", {})
        scene.cable_objects = assets.get("cable_objects", {})
        with pytest.raises(NotImplementedError, match="deformable or cable"):
            _Exporter(scene).export("unused.usda")


def test_converted_usd_in_the_temporary_directory_is_refused(tmp_path, monkeypatch):
    """A layer an asset converter wrote to its default temporary folder would be lost; other USD stays referenced."""
    import tempfile

    from isaaclab.sim.usd_export.base_usd_exporter import BaseUsdExporter

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    converted = tmp_path / "IsaacLab" / "usd_0" / "robot.usda"
    rewritten = tmp_path / "isaaclab_usd_0123" / "assets" / "robot.usda"
    authored = tmp_path / "scene" / "table.usda"
    for layer in (converted, rewritten, authored):
        layer.parent.mkdir(parents=True)
        Sdf.Layer.CreateNew(str(layer)).Save()
    export = str(tmp_path / "export" / "env.usda")

    for layer in (converted, rewritten):
        with pytest.raises(ValueError, match="temporary directory"):
            BaseUsdExporter._copy_temporary_file(str(layer), export)
    assert BaseUsdExporter._copy_temporary_file(str(authored), export) == str(authored)


def test_material_shared_with_a_collider_not_exported_is_not_changed():
    """Writing a material changes every collider bound to it, so one also used by an unexported collider raises."""
    from pxr import UsdShade

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/Looks/Rubber")
    UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    for name in ("Exported", "Other"):
        prim = UsdGeom.Cube.Define(stage, f"/{name}").GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")

    with pytest.raises(NotImplementedError, match="shared with colliders not exported"):
        PhysxUsdExporter._compute_material_edits(stage, {"/Exported": {"physics:dynamicFriction": 0.7}})


_SCALED_COM = {"scale": 2.0, "physics:centerOfMass": (0.01, -0.02, 0.03)}


@pytest.mark.parametrize(
    "authored,engine,expected",
    [
        # The engine's center of mass includes the body's scale, which importers apply to the USD value again.
        pytest.param(_SCALED_COM, {"com": (0.02, -0.04, 0.06)}, {"physics:centerOfMass": None}, id="scaled-com-kept"),
        pytest.param(
            _SCALED_COM, {"com": (0.04, -0.04, 0.06)}, {"physics:centerOfMass": (0.02, -0.02, 0.03)}, id="scaled-com"
        ),
        # USD states an unset mass as zero, from which an import derives one, so a zero engine mass is reported.
        pytest.param(
            {"physics:mass": 2.0},
            {"mass": 0.0, "inertia": np.zeros((3, 3)), "changed": {"body_mass", "body_inertia"}},
            {},
            id="zero-mass",
        ),
        # An importer derives an unset inertia from the mass, so a changed mass also states it; a stated one is kept.
        pytest.param(
            {"physics:mass": 2.0, "physics:diagonalInertia": (1.0, 1.0, 1.0)},
            {"mass": 4.0, "changed": {"body_mass"}},
            {"physics:mass": 4.0, "physics:diagonalInertia": None},
            id="stated-inertia",
        ),
        # A zero diagonal inertia is unspecified, as an unset one is.
        pytest.param(
            {"physics:mass": 1.0, "physics:diagonalInertia": (0.0, 0.0, 0.0)},
            {"mass": 2.0, "inertia": 0.5 * np.eye(3), "changed": {"body_mass"}},
            {"physics:mass": 2.0, "physics:diagonalInertia": (0.5, 0.5, 0.5)},
            id="zero-inertia",
        ),
        # The edits apply the mass schema, which turns on a center of mass authored without it; that is restated.
        pytest.param(
            {"PhysicsMassAPI": False, "physics:centerOfMass": (0.4, 0.0, 0.0)},
            {"mass": 2.0, "changed": {"body_mass"}},
            {"physics:mass": 2.0, "physics:centerOfMass": (0.0, 0.0, 0.0)},
            id="com-without-schema",
        ),
        # Newton reads newton:inertia ahead of the diagonal inertia, so a changed inertia updates it too.
        pytest.param(
            {"newton:inertia": [1.0, 2.0, 2.0, 0.0, 0.0, 0.0]},
            {"inertia": np.diag([2.0, 3.0, 3.0])},
            {"newton:inertia": [2.0, 3.0, 3.0, 0.0, 0.0, 0.0]},
            id="newton-compact-inertia",
        ),
    ],
)
def test_mass_edits_state_engine_values_as_an_import_reads_them(caplog, authored, engine, expected):
    """A body's mass edits for the engine's values; ``None`` marks an attribute left unwritten, and no expected edit an
    engine value USD cannot state, which is reported. ``authored`` may leave the mass schema unapplied."""
    from pxr import Gf

    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/Body")
    applied = authored.get("PhysicsMassAPI", True)
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim()) if applied else UsdPhysics.MassAPI(body.GetPrim())
    create = {
        "physics:mass": mass.CreateMassAttr,
        "physics:centerOfMass": mass.CreateCenterOfMassAttr,
        "physics:diagonalInertia": mass.CreateDiagonalInertiaAttr,
    }
    for name, value in authored.items():
        if name == "PhysicsMassAPI":
            continue
        if name == "scale":
            body.AddScaleOp().Set(Gf.Vec3f(value, value, value))
        elif name == "newton:inertia":
            body.GetPrim().AddAppliedSchema("NewtonMassAPI")
            body.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.DoubleArray).Set(value)
        else:
            create[name]().Set(Gf.Vec3f(*value) if isinstance(value, tuple) else value)
    engine = {"mass": 1.0, "com": (0.0, 0.0, 0.0), "inertia": np.eye(3)} | engine
    changed = {"changed": engine.pop("changed")} if "changed" in engine else {}

    edits = PhysxUsdExporter._compute_mass_edits(
        body.GetPrim(), engine["mass"], np.asarray(engine["com"]), engine["inertia"], **changed
    )

    written = {edit.attribute: edit.value for edit in edits}
    if not expected:
        assert written == {} and "cannot state the zero mass of '/Body'" in caplog.text
    for name, value in expected.items():
        if value is None:
            assert name not in written, name
        else:
            np.testing.assert_allclose(np.asarray(written[name], dtype=float), value, rtol=1e-6, err_msg=name)
