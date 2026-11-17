# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD export test utilities: a scene fixture whose configuration differs from its assets, registered tasks exported
at initialization, prim discovery, and the comparison of a live engine with an import of the export."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


def make_export_scene_cfg(directory: Path):
    """Create a scene whose configuration overrides values of its own USD assets.

    The robot USD authors a revolute drive (stiffness 1, damping 1 per degree, max force 100) that
    the actuator configuration replaces. The scene has one environment and exports to ``directory``.

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

    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0, export_usd_path=str(Path(directory) / "exported.usda"))
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


def write_default_joint_state(scene):
    """Write each articulation's default joint state to the engine, as Isaac Lab's reset does.

    The export states the initial joint state, which the engine holds only from that reset on.
    """
    for robot in scene.articulations.values():
        robot.write_joint_state_to_sim_index(
            position=robot.data.default_joint_pos.torch, velocity=robot.data.default_joint_vel.torch
        )


def read_live_engine(scene, path: str, read: Callable[[Any, list[str], list[str]], dict]) -> tuple:
    """Read a PhysX-family engine's values of the physics objects of an export, after
    :func:`write_default_joint_state`.

    Args:
        scene: The exported scene.
        path: USD file the scene exported to.
        read: Reads the engine's values of the given articulation roots and rigid bodies outside articulations from
            the exported stage, keyed by prim and axis.

    Returns:
        The exported stage, its articulation roots, its rigid bodies outside articulations, and the live readings.
    """
    from isaaclab.sim.usd_export.base_usd_exporter import BaseUsdExporter

    write_default_joint_state(scene)
    stage = BaseUsdExporter.open_stage(path)
    articulations, bodies = physics_prims(stage)
    return stage, articulations, bodies, read(stage, articulations, bodies)


def export_task(env_cfg, task: str, path: str, read_live: Callable[[Any], Any]) -> Any:
    """Build a registered task with one environment, which exports it to ``path`` at initialization, and read the
    live engine there.

    Args:
        env_cfg: The task's environment configuration, with its physics preset resolved.
        task: Registered task name.
        path: USD file the environment exports to.
        read_live: Called with the exported scene right after the file is written, before startup events
            (randomization) change the engine; its result is returned.

    Returns:
        What ``read_live`` returned.
    """
    import gymnasium as gym

    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext
    from isaaclab.sim.usd_export.base_usd_exporter import BaseUsdExporter

    env_cfg.scene.num_envs = 1
    env_cfg.scene.export_usd_path = path
    env_cfg.seed = 0  # the same terrain and reset state every run
    export, live = BaseUsdExporter.export, []

    def export_and_read_live(exporter, destination):
        written = export(exporter, destination)
        live.append(read_live(exporter.scene))
        return written

    sim_utils.create_new_stage()
    env = None
    BaseUsdExporter.export = export_and_read_live
    try:
        env = gym.make(task, cfg=env_cfg)
    finally:
        BaseUsdExporter.export = export
        if env is not None:
            env.close()
        SimulationContext.clear_instance()
    assert live, f"{task} did not export at initialization."
    return live[0]


def reading_pairs(readings: dict) -> set[tuple[str, str]]:
    """The ``(reading, entity)`` pairs of readings keyed by prim and axis, as :func:`find_reading_differences` names
    them."""
    return {(name, _entity(key)) for key, values in readings.items() for name in values}


def find_reading_differences(live: dict, loaded: dict) -> set[tuple[str, str]]:
    """Find the readings of a live engine and an import that differ, both keyed by prim and axis.

    Infinite and NaN components, e.g. the limits of an unlimited joint, must be equal; finite ones within a tolerance
    scaled by the reading's largest finite magnitude.

    Args:
        live: Per ``(prim path, axis or None)``, the live engine's readings by name.
        loaded: The same for the import.

    Returns:
        The ``(reading, entity)`` pairs that differ, the entity being ``<prim path>`` or ``<prim path>:<axis>``;
        an entity only one side has differs in its reading ``"missing"``.
    """
    import numpy as np

    differences = set()
    for key in live.keys() | loaded.keys():
        if key not in live or key not in loaded:
            differences.add(("missing", _entity(key)))
            continue
        for name, value in live[key].items():
            a, b = np.asarray(value, dtype=float), np.asarray(loaded[key][name], dtype=float)
            if a.shape != b.shape:
                differences.add((name, _entity(key)))
                continue
            finite = np.isfinite(a) & np.isfinite(b)
            scale = max(np.abs(a[finite]).max(initial=0.0), np.abs(b[finite]).max(initial=0.0))
            same = np.array_equal(a[~finite], b[~finite], equal_nan=True) and np.allclose(
                a[finite], b[finite], rtol=1e-4, atol=1e-6 + 1e-5 * scale
            )
            if not same:
                differences.add((name, _entity(key)))
    return differences


def assert_differences_match(
    differences: set[tuple[str, str]], expected: dict[str, str], compared: Iterable[tuple[str, str]] = ()
):
    """Assert that the values a live engine and an import of its export hold differently are exactly the expected ones.

    Every compared entity an expected pattern matches must differ, and no other: a difference the table does not
    list fails, and so does an expected one that no longer occurs, so that the table follows the export's known gaps.

    Args:
        differences: The differing ``(value name, entity)`` pairs.
        expected: Per value name, a regular expression matching exactly the entities whose value differs.
        compared: Every ``(value name, entity)`` pair compared, which the expected patterns select from. Defaults
            to none, for an empty ``expected``.
    """
    wanted = {(name, entity) for name, entity in compared if name in expected and re.fullmatch(expected[name], entity)}
    unexpected, gone = sorted(differences - wanted), sorted(wanted - differences)
    assert not unexpected and not gone, f"Unexpected differences: {unexpected[:20]}; expected ones gone: {gone[:20]}"


def _entity(key: tuple[str, str | None]) -> str:
    """``<prim path>`` or ``<prim path>:<axis>`` of a reading key."""
    return key[0] if key[1] is None else f"{key[0]}:{key[1]}"
