# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less tests of how the USD export maps engine values and actuators to prims."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.sim.usd_export.physx_usd_exporter import PhysxUsdExporter


class _Exporter(PhysxUsdExporter):
    """Reads the live collider values of rigid objects from ``colliders``, keyed by prim path."""

    colliders: dict[str, dict[str, np.ndarray]] = {}

    def _read_engine(self, path_expr, articulation):
        raise NotImplementedError

    def _read_object(self, prim_path, articulation_cfg=None):
        return {name: np.asarray(value)[None] for name, value in self.colliders[prim_path].items()}


def _engine(**values):
    """Engine values of one environment, as the scene keeps them from startup."""
    return [{name: np.asarray(value, dtype=float)[None] for name, value in values.items()}]


def _articulation_engine(articulation):
    """What the engine loaded for an articulation: zeros, so every nonzero live value is Isaac Lab's."""
    data = articulation.data
    dofs = data.joint_stiffness.torch.shape[1]
    zeros = {name: np.zeros(dofs) for name in ("stiffness", "damping", "max_force", "armature", "max_velocity")}
    zeros |= {name: np.zeros(dofs) for name in ("static_friction", "dynamic_friction", "viscous_friction")}
    bodies = data.body_mass.torch.shape[1]
    return _engine(
        **zeros,
        lower_limit=np.zeros(dofs),
        upper_limit=np.zeros(dofs),
        mass=np.zeros(bodies),
        com=np.zeros((bodies, 3)),
        inertia=np.zeros((bodies, 9)),
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
    assets = {"articulations": {}, "rigid_objects": {}, "rigid_object_collections": {}, **assets}
    base = {name: _articulation_engine(a) for name, a in assets["articulations"].items()} | (base or {})
    sim = SimpleNamespace(get_physics_step_count=lambda: 0)
    return SimpleNamespace(num_envs=1, sim=sim, _usd_export_base=base, **assets)


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


@pytest.mark.parametrize(
    "joint_type,converted",
    [(UsdPhysics.RevoluteJoint, True), (UsdPhysics.SphericalJoint, False)],
    ids=["revolute joint", "spherical joint"],
)
def test_pd_actuator_becomes_a_drive_of_single_axis_joints_only(joint_type, converted):
    """A PD actuator becomes a revolute or prismatic joint's drive; on other joints it stays an actuator prim."""
    stage = Usd.Stage.CreateInMemory()
    joint = joint_type.Define(stage, "/Robot/Joint").GetPrim()
    actuator = stage.DefinePrim("/Robot/Actuator", "NewtonActuator")
    actuator.AddAppliedSchema("NewtonPDControlAPI")
    actuator.CreateRelationship("newton:targets").SetTargets([joint.GetPath()])
    actuator.CreateAttribute("newton:kp", Sdf.ValueTypeNames.Float).Set(80.0)

    _Exporter(_scene())._write_pd_actuators_as_drives(stage)

    assert actuator.IsActive() != converted
    assert bool([schema for schema in joint.GetAppliedSchemas() if schema.startswith("PhysicsDriveAPI")]) == converted


def test_gains_of_a_joint_without_a_usd_drive_add_no_drive():
    """PhysX drives only joints whose USD states a drive, so gains the engine holds elsewhere are not exported."""
    stage = Usd.Stage.CreateInMemory()
    physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    physics_scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    for name in ("Base", "Link"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/Robot/{name}").GetPrim())
    UsdPhysics.RevoluteJoint.Define(stage, "/Robot/Hinge")
    robot = _articulation(["Hinge"], ["Base", "Link"])
    robot.data.joint_damping.torch.fill_(0.1)
    scene = _scene(articulations={"robot": robot})
    scene.physics_scene_path = "/physicsScene"
    scene.sim.backend_manager = SimpleNamespace(get_gravity=lambda: (0.0, 0.0, -9.81))

    edits = _Exporter(scene).compute_diff(stage)

    assert not [edit for edit in edits if edit.path == "/Robot/Hinge" and edit.attribute.startswith("drive:")]


def test_reordered_joints_compare_with_their_own_engine_values():
    """Isaac Lab may order joints differently from the engine; each joint is compared with its own loaded value."""
    stage = Usd.Stage.CreateInMemory()
    physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    physics_scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    for name in ("Base", "Link"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/Robot/{name}").GetPrim())
    for name in ("A", "B"):
        UsdPhysics.DriveAPI.Apply(UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/{name}").GetPrim(), "angular")
    robot = _articulation(["A", "B"], ["Base", "Link"])
    robot.backend_joint_names = ["B", "A"]
    robot.data.joint_stiffness.torch[0] = torch.tensor([5.0, 7.0])  # A as loaded, B changed
    base = _articulation_engine(robot)
    base[0]["stiffness"][0] = [0.0, 5.0]  # engine order: B, A
    scene = _scene({"robot": base}, articulations={"robot": robot})
    scene.physics_scene_path = "/physicsScene"
    scene.sim.backend_manager = SimpleNamespace(get_gravity=lambda: (0.0, 0.0, -9.81))

    edits = {edit.path: edit.value for edit in _Exporter(scene).compute_diff(stage) if "stiffness" in edit.attribute}

    assert edits == {"/Robot/B": pytest.approx(7.0 * np.pi / 180.0)}


def test_attributes_a_new_api_schema_enables_are_written():
    """PhysX ignores an attribute whose API schema the prim lacks; once an edit applies the schema, it states live."""
    stage = Usd.Stage.CreateInMemory()
    physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    physics_scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    for name in ("Base", "Link"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/Robot/{name}").GetPrim())
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/Hinge").GetPrim()
    hinge.CreateAttribute("physxJointAxis:angular:armature", Sdf.ValueTypeNames.Float).Set(0.5)  # without its schema
    robot = _articulation(["Hinge"], ["Base", "Link"])
    robot.data.joint_vel_limits.torch.fill_(2.0)  # loaded as 0: Isaac Lab's, written with PhysxJointAxisAPI
    scene = _scene(articulations={"robot": robot})
    scene.physics_scene_path = "/physicsScene"
    scene.sim.backend_manager = SimpleNamespace(get_gravity=lambda: (0.0, 0.0, -9.81))

    edits = {edit.attribute: edit.value for edit in _Exporter(scene).compute_diff(stage) if edit.path == "/Robot/Hinge"}

    assert edits["physxJointAxis:angular:armature"] == pytest.approx(0.0)  # live, as the engine ignored the 0.5
    assert "physxJointAxis:angular:maxJointVelocity" in edits


def test_a_value_authored_without_its_schema_is_written_with_it():
    """An authored value equal to the live one states nothing without its API schema, so the export applies it."""
    stage = Usd.Stage.CreateInMemory()
    physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    physics_scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    for name in ("Base", "Link"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/Robot/{name}").GetPrim())
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/Hinge").GetPrim()
    hinge.CreateAttribute("physxJointAxis:angular:armature", Sdf.ValueTypeNames.Float).Set(0.5)  # without its schema
    robot = _articulation(["Hinge"], ["Base", "Link"])
    robot.data.joint_armature.torch.fill_(0.5)  # loaded as 0 (ignored), configured to the authored value
    scene = _scene(articulations={"robot": robot})
    scene.physics_scene_path = "/physicsScene"
    scene.sim.backend_manager = SimpleNamespace(get_gravity=lambda: (0.0, 0.0, -9.81))

    edits = [
        edit for edit in _Exporter(scene).compute_diff(stage) if edit.attribute == "physxJointAxis:angular:armature"
    ]

    assert len(edits) == 1 and edits[0].schema == "PhysxJointAxisAPI:angular"


def test_axes_of_a_rotational_d6_joint_map_to_its_usd_axes():
    """PhysX names the DOFs of a multi-axis joint "<joint>:<n>" after the axis index (0 rotX, 1 rotY, 2 rotZ)."""
    stage = Usd.Stage.CreateInMemory()
    physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    physics_scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
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
    scene.physics_scene_path = "/physicsScene"
    scene.sim.backend_manager = SimpleNamespace(get_gravity=lambda: (0.0, 0.0, -9.81))

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
    scene.physics_scene_path = "/physicsScene"
    scene.sim.backend_manager = SimpleNamespace(get_gravity=lambda: (0.0, 0.0, -9.81))

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
    scene = _scene()
    scene.physics_scene_path = "/physicsScene"

    edits = _Exporter(scene)._gravity_edits(stage, live)

    assert bool(edits) == written


def test_only_values_that_differ_from_what_the_engine_loaded_are_written():
    """The engine loads what the stage states and derives the rest, as a PhysX import of the file does.

    A value Isaac Lab did not change is therefore not written, whether the USD states it or the engine derives it.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    physics_scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    for name in ("Unchanged", "Changed"):
        body = UsdGeom.Xform.Define(stage, f"/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, f"/{name}/Collider").GetPrim())
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/Changed")).CreateMassAttr().Set(1.0)
    stage.GetPrimAtPath("/Changed/Collider").CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(
        0.0
    )

    def rigid_object(path):
        data = SimpleNamespace(
            body_mass=SimpleNamespace(torch=torch.full((1, 1), 3.0)),
            body_com_pose_b=SimpleNamespace(torch=torch.tensor([[[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]])),
            body_inertia=SimpleNamespace(torch=(2.0 * torch.eye(3)).reshape(1, 1, 9)),
        )
        return SimpleNamespace(cfg=SimpleNamespace(prim_path=path), data=data)

    live = {"material": np.array([[0.5, 0.5, 0.0]]), "contact_offset": [0.02], "rest_offset": [0.001]}
    loaded = {"mass": [3.0], "com": [[0.1, 0.0, 0.0]], "inertia": [(2.0 * np.eye(3)).ravel()], **live}
    base = {"unchanged": _engine(**loaded), "changed": _engine(**loaded | {"mass": [1.0], "rest_offset": [0.0]})}
    scene = _scene(base, rigid_objects={"unchanged": rigid_object("/Unchanged"), "changed": rigid_object("/Changed")})
    scene.physics_scene_path = "/physicsScene"
    scene.sim.backend_manager = SimpleNamespace(get_gravity=lambda: (0.0, 0.0, -9.81))
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
    stage = Usd.Stage.CreateInMemory()
    physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    physics_scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    table = UsdGeom.Xform.Define(stage, "/Table").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(table)
    UsdPhysics.MassAPI.Apply(table).CreateMassAttr().Set(1.0)
    for name in ("Top", "Legs"):
        collider = UsdGeom.Cube.Define(stage, f"/Table/{name}").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider)
        for attribute in ("physxCollision:contactOffset", "physxCollision:restOffset"):
            collider.CreateAttribute(attribute, Sdf.ValueTypeNames.Float).Set(0.0)

    data = SimpleNamespace(
        body_mass=SimpleNamespace(torch=torch.full((1, 1), 3.0)),
        body_com_pose_b=SimpleNamespace(torch=torch.zeros(1, 1, 7)),
        body_inertia=SimpleNamespace(torch=torch.eye(3).reshape(1, 1, 9)),
    )
    material = np.full((3, 3), 0.5)
    live = {"material": material, "contact_offset": [0.002, 0.016, 0.016], "rest_offset": np.full(3, 0.001)}
    loaded = {"material": material, "contact_offset": [0.002, 0.016, 0.016], "rest_offset": np.zeros(3)}
    base = {"table": _engine(mass=[1.0], com=[np.zeros(3)], inertia=[np.eye(3).ravel()], **loaded)}
    scene = _scene(base, rigid_objects={"table": SimpleNamespace(cfg=SimpleNamespace(prim_path="/Table"), data=data)})
    scene.physics_scene_path = "/physicsScene"
    scene.sim.backend_manager = SimpleNamespace(get_gravity=lambda: (0.0, 0.0, -9.81))
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
    """Material values of a collider without a physics material go on a new material bound to it, at an unused path.

    A body without colliders, which reports empty per-shape arrays, gets no collider values.
    """
    stage = Usd.Stage.CreateInMemory()
    physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    physics_scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    stage.DefinePrim("/PhysicsMaterials/Material_0", "Material")
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Box").GetPrim())
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, "/Box/Collider").GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Bare").GetPrim())
    data = SimpleNamespace(
        body_mass=SimpleNamespace(torch=torch.full((1, 1), 1.0)),
        body_com_pose_b=SimpleNamespace(torch=torch.zeros(1, 1, 7)),
        body_inertia=SimpleNamespace(torch=torch.eye(3).reshape(1, 1, 9)),
    )
    live = {"material": np.array([[0.8, 0.6, 0.1]]), "contact_offset": [0.02], "rest_offset": [0.0]}
    loaded = {"mass": [1.0], "com": [np.zeros(3)], "inertia": [np.eye(3).ravel()], **live}
    bare = {"material": np.zeros((0, 3)), "contact_offset": np.zeros(0), "rest_offset": np.zeros(0)}
    base = {
        "box": _engine(**loaded | {"material": np.array([[0.5, 0.5, 0.0]])}),
        "bare": _engine(**loaded | bare | {"mass": [2.0]}),
    }
    rigid_objects = {
        name: SimpleNamespace(cfg=SimpleNamespace(prim_path=path), data=data)
        for name, path in (("box", "/Box"), ("bare", "/Bare"))
    }
    scene = _scene(base, rigid_objects=rigid_objects)
    scene.physics_scene_path = "/physicsScene"
    scene.sim.backend_manager = SimpleNamespace(get_gravity=lambda: (0.0, 0.0, -9.81))
    exporter = _Exporter(scene)
    exporter.colliders = {"/Box": live, "/Bare": bare}

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


def test_a_collision_schema_the_export_applies_states_the_other_offsets():
    """Writing one offset applies PhysxCollisionAPI, which enables the collider's other authored offset."""
    stage = Usd.Stage.CreateInMemory()
    physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    physics_scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Box").GetPrim())
    collider = UsdGeom.Cube.Define(stage, "/Box/Collider").GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider)
    collider.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(5.0)  # ignored
    data = SimpleNamespace(
        body_mass=SimpleNamespace(torch=torch.full((1, 1), 1.0)),
        body_com_pose_b=SimpleNamespace(torch=torch.zeros(1, 1, 7)),
        body_inertia=SimpleNamespace(torch=torch.eye(3).reshape(1, 1, 9)),
    )
    live = {"material": np.array([[0.5, 0.5, 0.0]]), "contact_offset": [0.02], "rest_offset": [0.004]}
    loaded = {"mass": [1.0], "com": [np.zeros(3)], "inertia": [np.eye(3).ravel()], **live, "rest_offset": [0.0]}
    scene = _scene(
        {"box": _engine(**loaded)},
        rigid_objects={"box": SimpleNamespace(cfg=SimpleNamespace(prim_path="/Box"), data=data)},
    )
    scene.physics_scene_path = "/physicsScene"
    scene.sim.backend_manager = SimpleNamespace(get_gravity=lambda: (0.0, 0.0, -9.81))
    exporter = _Exporter(scene)
    exporter.colliders = {"/Box": live}

    edits = {edit.attribute: edit.value for edit in exporter.compute_diff(stage) if edit.path == "/Box/Collider"}

    assert edits == {
        "physxCollision:restOffset": pytest.approx(0.004),
        "physxCollision:contactOffset": pytest.approx(0.02),
    }


def test_an_asset_outside_the_environments_is_read_from_its_one_row():
    """A rigid object outside the environment namespace has one engine row, whichever environment is exported."""
    stage = Usd.Stage.CreateInMemory()
    physics_scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    physics_scene.CreateGravityDirectionAttr((0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Shared").GetPrim())
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/Shared")).CreateMassAttr().Set(1.0)
    data = SimpleNamespace(
        body_mass=SimpleNamespace(torch=torch.full((1, 1), 3.0)),
        body_com_pose_b=SimpleNamespace(torch=torch.zeros(1, 1, 7)),
        body_inertia=SimpleNamespace(torch=torch.eye(3).reshape(1, 1, 9)),
    )
    live = {"material": np.zeros((0, 3)), "contact_offset": np.zeros(0), "rest_offset": np.zeros(0)}
    loaded = {"mass": [1.0], "com": [np.zeros(3)], "inertia": [np.eye(3).ravel()], **live}
    scene = _scene(
        {"shared": _engine(**loaded)},
        rigid_objects={"shared": SimpleNamespace(cfg=SimpleNamespace(prim_path="/Shared"), data=data)},
    )
    scene.num_envs = 2
    scene.physics_scene_path = "/physicsScene"
    scene.sim.backend_manager = SimpleNamespace(get_gravity=lambda: (0.0, 0.0, -9.81))
    exporter = _Exporter(scene, env_id=1)
    exporter.colliders = {"/Shared": live}

    edits = {edit.attribute: edit.value for edit in exporter.compute_diff(stage) if edit.path == "/Shared"}

    assert edits["physics:mass"] == pytest.approx(3.0)


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

    _Exporter(scene, env_id=1)._select_environment(stage)

    exported = stage.GetPrimAtPath("/World/envs/env_1/Box")
    assert exported.IsDefined() and exported.GetTypeName() == "Cube"
    assert not stage.GetPrimAtPath("/World/envs/env_0")
    assert UsdGeom.XformCache().GetLocalToWorldTransform(exported).ExtractTranslation()[0] == pytest.approx(3.0)


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
            BaseUsdExporter._keep_temporary_file(str(layer), export)
    assert BaseUsdExporter._keep_temporary_file(str(authored), export) == str(authored)


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
        PhysxUsdExporter._material_edits(stage, {"/Exported": {"physics:dynamicFriction": 0.7}})


def test_center_of_mass_of_a_scaled_body_is_written_unscaled():
    """The engine's center of mass includes the body's scale, which importers apply to the USD value again."""
    from pxr import Gf

    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/Body")
    body.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 2.0))
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.01, -0.02, 0.03))

    edits = PhysxUsdExporter._mass_edits(body.GetPrim(), 1.0, np.array([0.02, -0.04, 0.06]), np.eye(3))
    assert "physics:centerOfMass" not in {edit.attribute for edit in edits}
    edits = PhysxUsdExporter._mass_edits(body.GetPrim(), 1.0, np.array([0.04, -0.04, 0.06]), np.eye(3))
    com = next(edit.value for edit in edits if edit.attribute == "physics:centerOfMass")
    np.testing.assert_allclose(np.asarray(com), [0.02, -0.02, 0.03], rtol=1e-6)


@pytest.mark.parametrize("stated", [False, True], ids=["inertia derived", "inertia stated"])
def test_a_changed_mass_writes_an_inertia_the_import_would_derive_from_it(stated):
    """An importer derives an unset inertia from the mass, so a changed mass also states the inertia."""
    from pxr import Gf

    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/Body").GetPrim()
    mass = UsdPhysics.MassAPI.Apply(body)
    mass.CreateMassAttr().Set(2.0)
    if stated:
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

    edits = PhysxUsdExporter._mass_edits(body, 4.0, np.zeros(3), np.eye(3), changed={"mass"})

    written = {edit.attribute for edit in edits}
    assert "physics:mass" in written
    assert ("physics:diagonalInertia" in written) is not stated


def test_newton_compact_inertia_is_updated_with_the_inertia():
    """Newton reads ``newton:inertia`` ahead of the diagonal inertia, so a changed inertia updates it too."""
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/Body").GetPrim()
    UsdPhysics.MassAPI.Apply(body)
    body.AddAppliedSchema("NewtonMassAPI")
    body.CreateAttribute("newton:inertia", Sdf.ValueTypeNames.DoubleArray).Set([1.0, 2.0, 2.0, 0.0, 0.0, 0.0])

    edits = PhysxUsdExporter._mass_edits(body, 1.0, np.zeros(3), np.diag([2.0, 3.0, 3.0]))
    compact = next(edit.value for edit in edits if edit.attribute == "newton:inertia")
    np.testing.assert_allclose(compact, [2.0, 3.0, 3.0, 0.0, 0.0, 0.0])
