# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD export test utilities: a scene fixture whose configuration differs from its assets, and prim discovery."""

from __future__ import annotations

from pathlib import Path


def make_export_scene_cfg(directory: Path):
    """Create a scene whose configuration overrides values of its own USD assets.

    The robot USD authors a revolute drive (stiffness 1, damping 1 per degree, max force 100) that
    the actuator configuration replaces. The scene has one environment.

    Args:
        directory: Directory for the generated robot USD file.

    Returns:
        The scene configuration.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
    from isaaclab.scene import InteractiveSceneCfg

    source = Path(directory) / "robot.usda"
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
    stage.GetRootLayer().Save()

    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0)
    cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(source)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2), joint_pos={"Hinge": 0.21}),
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        )

    cfg.box = rigid("Box", 2.5, (1.0, 0.0, 1.0))
    cfg.collection = RigidObjectCollectionCfg(
        rigid_objects={
            "first": rigid("CollectedFirst", 1.5, (2.0, 0.0, 1.0)),
            "second": rigid("CollectedSecond", 3.5, (3.0, 0.0, 1.0)),
        }
    )
    cfg.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(size=(1.0, 1.0, 0.1), collision_props=sim_utils.CollisionBaseCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 2.0, 0.5)),
    )
    cfg.ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.CuboidCfg(size=(10.0, 10.0, 0.1), collision_props=sim_utils.CollisionBaseCfg()),
    )
    return cfg


def physics_prims(stage) -> tuple[list[str], list[str]]:
    """Articulation roots, and rigid bodies that belong to no articulation."""
    from pxr import UsdPhysics

    prims = list(stage.Traverse())
    articulations = [str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    scopes = [articulation_scope(stage, root).GetPath() for root in articulations]
    bodies = [
        str(prim.GetPath())
        for prim in prims
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not any(prim.GetPath().HasPrefix(scope) for scope in scopes)
    ]
    return articulations, bodies


def articulation_scope(stage, root: str):
    """The prim containing an articulation's links: its root, or the root's parent if the root is a link or joint."""
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(root)
    return prim.GetParent() if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint) else prim
