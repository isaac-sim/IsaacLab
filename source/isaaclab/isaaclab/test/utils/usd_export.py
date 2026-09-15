# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent USD parser checks supplementing fresh-backend export tests."""

import numpy as np

from pxr import Gf, Sdf, Usd, UsdPhysics, UsdShade


def capture_physics_structure(stage: Usd.Stage) -> dict:
    """Capture parsed entity coverage, topology, collision geometry and filtering by prim identity.

    Runtime body poses, drives, limits and material coefficients are compared through backend
    views by the caller. Everything else exposed by the USD physics descriptors is retained here,
    including geometry dimensions, shape-to-body associations and joint attachment frames.
    """
    result = {}
    buffered = {
        "position",
        "rotation",
        "linearVelocity",
        "angularVelocity",
        "materials",
        "drive",
        "limit",
        "jointDrives",
        "jointLimits",
    }
    parsed = UsdPhysics.LoadUsdPhysicsFromRange(stage, ["/"])
    for kind, (paths, descriptions) in parsed.items():
        if kind in (UsdPhysics.ObjectType.Scene, UsdPhysics.ObjectType.RigidBodyMaterial):
            continue
        for path, description in zip(paths, descriptions):
            assert description.isValid, path
            prim = stage.GetPrimAtPath(path)
            # D6 descriptors bundle per-axis buffered values; retain axis identities separately.
            for field in ("jointDrives", "jointLimits"):
                if hasattr(description, field):
                    result[str(path), field + "Axes"] = tuple(str(item.first) for item in getattr(description, field))
            for field in dir(description):
                if field.startswith("_") or field in buffered:
                    continue
                value = getattr(description, field)
                if kind == UsdPhysics.ObjectType.CollisionGroup and field == "invertFilteredGroups":
                    # Some parser builds leave this descriptor field unset for an unauthored
                    # attribute. Read the USD value explicitly, including its false default.
                    value = bool(UsdPhysics.CollisionGroup(prim).GetInvertFilteredGroupsAttr().Get())
                if not callable(value):
                    result[str(path), field] = _value(value)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                if material:
                    for attribute in material.GetPrim().GetAttributes():
                        if "CombineMode" in attribute.GetName():
                            result[str(path), attribute.GetName()] = _value(attribute.Get())
    return result


def assert_physics_structure_equal(expected: dict, actual: dict) -> None:
    """Compare discrete structure exactly and geometric floating-point values with tolerances."""
    assert actual.keys() == expected.keys(), (actual.keys() - expected.keys(), expected.keys() - actual.keys())
    for key, value in expected.items():
        other = actual[key]
        if isinstance(value, np.ndarray):
            if value.dtype.kind == "f":
                np.testing.assert_allclose(other, value, rtol=1e-5, atol=1e-6, err_msg=str(key))
            else:
                np.testing.assert_array_equal(other, value, err_msg=str(key))
        elif isinstance(value, float):
            np.testing.assert_allclose(other, value, rtol=1e-5, atol=1e-6, err_msg=str(key))
        else:
            assert other == value, (key, other, value)


def _value(value):
    if isinstance(value, Sdf.Path):
        return str(value)
    if isinstance(value, (str, bool, int, float, type(None))):
        return value
    if isinstance(value, (Gf.Quatf, Gf.Quatd)):
        return np.array(Gf.Matrix3d(Gf.Quatd(value)))
    try:
        values = list(value)
    except TypeError:
        # USD enum values have stable symbolic names; descriptor subobjects expose properties.
        fields = {
            name: _value(getattr(value, name))
            for name in dir(value)
            if not name.startswith("_") and not callable(getattr(value, name))
        }
        return fields if fields else str(value)
    if all(isinstance(v, (float, int)) for v in values):
        return np.asarray(values)
    return tuple(_value(v) for v in values)


def make_fixed_scene_cfg(directory):
    """Create a portable deployment fixture using locally authored assets and non-default cfg values."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
    from isaaclab.scene import InteractiveSceneCfg

    source = directory / "fixed_robot.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    for index, name in enumerate(("Base", "Link")):
        body = UsdGeom.Xform.Define(stage, f"/Robot/{name}")
        body.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5 + index * 0.5))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        body.GetPrim().AddAppliedSchema("PhysxRigidBodyAPI")
        body.GetPrim().CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(index == 1)
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr().Set(3.0 - index)
        mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.01, -0.02, 0.03))
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(0.1, 0.2, 0.25))
        mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(0.9238795, Gf.Vec3f(0, 0, 0.3826834)))
        shape = UsdGeom.Cube.Define(stage, f"/Robot/{name}/Collision")
        shape.CreateSizeAttr().Set(0.2)
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        material = UsdShade.Material.Define(stage, f"/Robot/Materials/{name}")
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr().Set(0.5 + index * 0.2)
        physics_material.CreateDynamicFrictionAttr().Set(0.4 + index * 0.1)
        physics_material.CreateRestitutionAttr().Set(0.1 + index * 0.1)
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material, materialPurpose="physics")
        shape.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
        shape.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(
            0.02 + 0.01 * index
        )
        shape.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.001 * index)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Robot/FixedRoot")
    fixed.CreateBody1Rel().SetTargets(["/Robot/Base"])
    fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0.5))
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/Hinge")
    joint.CreateBody0Rel().SetTargets(["/Robot/Base"])
    joint.CreateBody1Rel().SetTargets(["/Robot/Link"])
    joint.CreateAxisAttr().Set("Y")
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0.25))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, -0.25))
    joint.CreateLowerLimitAttr().Set(-60)
    joint.CreateUpperLimitAttr().Set(75)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateStiffnessAttr().Set(1)
    drive.CreateDampingAttr().Set(1)
    drive.CreateMaxForceAttr().Set(100)
    UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/Robot/Base")).CreateFilteredPairsRel().SetTargets(
        ["/Robot/Link"]
    )
    stage.GetRootLayer().Save()

    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0)
    cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(source)),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={"Hinge": 0.21}, joint_vel={"Hinge": 0.17}),
        actuators={
            "hinge": ImplicitActuatorCfg(
                joint_names_expr=["Hinge"],
                stiffness=83.0,
                damping=4.5,
                armature=0.023,
                joint_effort_limit=19.0,
                joint_velocity_limit=2.5,
            )
        },
    )

    def rigid(name, mass, position):
        return RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.3, 0.4),
                rigid_props=sim_utils.RigidBodyBaseCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionBaseCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.61, dynamic_friction=0.43, restitution=0.2
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=position, lin_vel=(0.12, -0.03, 0.02), ang_vel=(0.1, 0.2, 0.3)
            ),
        )

    cfg.box = rigid("Box", 2.5, (1.0, 0, 1.0))
    cfg.collection = RigidObjectCollectionCfg(
        rigid_objects={
            "first": rigid("CollectedFirst", 1.5, (2, 0, 1)),
            "second": rigid("CollectedSecond", 3.5, (3, 0, 1)),
        }
    )
    cfg.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(size=(1.0, 1.0, 0.1), collision_props=sim_utils.CollisionBaseCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 2, 0.5)),
    )
    cfg.ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.CuboidCfg(size=(10.0, 10.0, 0.1), collision_props=sim_utils.CollisionBaseCfg()),
    )
    cfg.light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1200))
    return cfg


# Importable fixtures let the isolated training worker construct the same Direct task.
from isaaclab.envs import DirectRLEnv


class FixedExportProbeEnv(DirectRLEnv):
    """Direct fixture with extra setup assets and observable event/random-state effects."""

    def _setup_scene(self):
        from pxr import UsdGeom

        prim = UsdGeom.Cube.Define(self.sim.stage, "/World/SetupOnly").GetPrim()
        prim.GetAttribute("size").Set(0.05)
        UsdPhysics.CollisionAPI.Apply(prim)

    def _pre_physics_step(self, actions):
        pass

    def _apply_action(self):
        pass

    def _get_observations(self):
        return {"policy": self.scene.rigid_objects["box"].data.root_pose_w.torch.clone()}

    def _get_rewards(self):
        import torch

        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        import torch

        value = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        return value, value


def fixed_export_prestartup(env, env_ids):
    """Visible one-time authored randomization retained in the deployment artifact."""
    import torch

    prims = [p for p in env.sim.stage.Traverse() if p.GetName() == "Box" and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert prims
    for prim in prims:
        UsdPhysics.MassAPI(prim).GetMassAttr().Set(20.0 + float(torch.rand(())))


def fixed_export_startup(env, env_ids):
    """Visible backend randomization and RNG use in ordinary task startup."""
    import torch

    asset = env.scene.rigid_objects["box"]
    velocity = torch.rand((env.num_envs, 6), device=env.device)
    asset.write_root_velocity_to_sim_index(root_velocity=velocity)
    factors = 1 + torch.rand((env.num_envs, 1), device=env.device)
    asset.set_masses_index(masses=asset.data.body_mass.torch.clone() * factors)
    asset.set_inertias_index(inertias=asset.data.body_inertia.torch.clone() * factors[..., None])
