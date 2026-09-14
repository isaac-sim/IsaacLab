# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Complete fixed single-environment USD export in the PhysX deployment dialect.

Authored USD is preserved; only fixed initialization values absent from USD are
written onto an isolated copy. This is not a runtime or multi-environment snapshot.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

from isaaclab.assets.physics_properties import (
    JOINT_PROPERTY_SOURCES,
    JOINT_USD_PROPERTIES,
    BodyInitialState,
    UsdProperty,
    validate_configuration_coverage,
)

if TYPE_CHECKING:
    from isaaclab.assets import BaseArticulation, BaseRigidObject, BaseRigidObjectCollection
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg


@dataclass(frozen=True)
class AssetPaths:
    """Prim identities paired with public body/DOF row indices in one environment."""

    bodies: list[tuple[str, int]]
    joints: list[tuple[str, int]]


class SceneExportAdapter(Protocol):
    """Source-backend provenance and fixed values absent from public asset data.

    Standard writes target UsdPhysics plus PhysX extensions. A source adapter must
    reject physical semantics it cannot preserve; it does not select a target backend.
    """

    def paths(self, asset: BaseArticulation | BaseRigidObject | BaseRigidObjectCollection) -> AssetPaths:
        """Resolve all bodies and supported DOFs to prims, in public data order."""
        ...

    def write_extensions(self, stage: Usd.Stage) -> None:
        """Preserve source-specific initialized semantics or fail before saving."""
        ...


def write_properties(prim: Usd.Prim, values: dict[str, np.ndarray], row: int, rules: dict[str, UsdProperty]) -> None:
    """Execute a declared joint-property mapping on an isolated SI stage."""
    axis = {"PhysicsRevoluteJoint": "angular", "PhysicsPrismaticJoint": "linear"}.get(prim.GetTypeName())
    if axis is None:
        raise NotImplementedError(f"Unsupported driven joint {prim.GetPath()} ({prim.GetTypeName()}).")
    for rule in rules.values():
        value = values[rule.source][row]
        if rule.component is not None:
            value = value[rule.component]
        scale = (180 / math.pi) ** rule.angular_power if axis == "angular" else 1.0
        for schema, name in rule.targets:
            if schema:
                if not prim.AddAppliedSchema(schema.format(axis=axis)):
                    raise RuntimeError(f"Could not apply {schema} at {prim.GetPath()}.")
            attr = prim.CreateAttribute(name.format(axis=axis), Sdf.ValueTypeNames.Float, custom=False)
            if not attr.Set(float(value) * scale):
                raise RuntimeError(f"Could not author {attr.GetPath()}.")


class SceneExporter:
    """Export a complete initialized fixed scene before events or application steps.

    Use :meth:`export_from_cfg` for a scene config, or the opt-in training CLI for
    a task whose scene also contains assets created by ``_setup_scene``.
    The scene must have exactly one environment. Runtime overrides are unsupported.
    """

    def __init__(self, scene: InteractiveScene) -> None:
        self.scene = scene
        self.timings: dict[str, float] = {}

    @classmethod
    def export_from_cfg(cls, scene_cfg: InteractiveSceneCfg, sim_cfg: SimulationCfg, usd_path: str) -> str:
        """Build and export one fixed scene without task events; inputs are copied.

        Args:
            scene_cfg: Fixed scene configuration, with ``num_envs=1``.
            sim_cfg: Simulation configuration, timestep [s] and gravity [m/s^2].
            usd_path: Destination USD file. External assets must remain accessible.
        """
        from copy import deepcopy

        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext, build_simulation_context

        if scene_cfg.num_envs != 1:
            raise ValueError("Fixed export requires exactly one environment.")
        if SimulationContext.instance() is not None:
            raise RuntimeError("export_from_cfg requires a fresh simulation context.")
        with build_simulation_context(sim_cfg=deepcopy(sim_cfg)) as sim:
            scene = InteractiveScene(deepcopy(scene_cfg))
            sim.reset()
            scene.reset_to_default()
            sim.forward()
            scene.update(0.0)
            return cls(scene).export(usd_path)

    @classmethod
    def export_task(cls, factory: Callable[[], object], usd_path: str) -> dict[str, float]:
        """Construct an actual task up to its complete scene, then export before events.

        ``factory`` must construct a fresh task with one environment. Run this in an
        isolated process: task construction may seed RNGs or create native resources.
        Assets created by Direct tasks' ``_setup_scene`` must be registered in the scene;
        unregistered physical bodies fail coverage checks. Controller/observation setup
        after the scene boundary is intentionally outside the deployment USD.
        """
        from isaaclab.envs.utils.scene_export import _scene_export_callback
        from isaaclab.sim import SimulationContext
        from isaaclab.sim.utils import use_stage

        if SimulationContext.instance() is not None:
            raise RuntimeError("Task export requires a fresh simulation context.")
        timings = {}
        started = perf_counter()

        class ExportComplete(Exception):
            pass

        def capture(env):
            timings["construction"] = perf_counter() - started
            initialized = perf_counter()
            with use_stage(env.sim.stage):
                env.sim.reset()
                env.scene.reset_to_default()
                env.sim.forward()
                env.scene.update(0.0)
            timings["initialize"] = perf_counter() - initialized
            exporter = cls(env.scene)
            exporter.export(usd_path)
            timings.update(exporter.timings)
            raise ExportComplete

        token = _scene_export_callback.set(capture)
        try:
            factory()
            raise RuntimeError("Task did not expose the pre-event scene construction boundary.")
        except ExportComplete:
            return timings
        finally:
            _scene_export_callback.reset(token)
            sim = SimulationContext.instance()
            if sim is not None:
                sim.clear_instance()

    def export(self, usd_path: str) -> str:
        """Export fixed initialization, preserving authored content and the live scene.

        Backend initialization/warmup must be complete and configured default state
        applied. Call before any task events or application physics steps. The training
        integration enforces this point in an isolated process.
        """
        from importlib import import_module

        from isaaclab.actuators.actuator_base_cfg import _is_implicit_actuator_cfg

        scene = self.scene
        if scene.num_envs != 1 or scene.sim.get_physics_step_count() != 0:
            raise ValueError("Fixed export requires exactly one environment before its first physics step.")
        for name in ("deformable_objects", "cable_objects", "surface_grippers"):
            if getattr(scene, name):
                raise NotImplementedError(f"Fixed export does not support required {name}.")
        if set(JOINT_PROPERTY_SOURCES) - JOINT_USD_PROPERTIES.keys():
            raise NotImplementedError("Actuator initialization properties lack USD mappings.")
        package = scene.sim.physics_manager.__module__.split(".")[0]
        if package not in {"isaaclab_physx", "isaaclab_ov", "isaaclab_newton"}:
            raise NotImplementedError(f"No fixed scene adapter for {package}.")
        start = perf_counter()
        adapter: SceneExportAdapter = import_module(f"{package}.sim.usd_export").SceneAdapter(scene)
        records = []
        for assets in (scene.articulations, scene.rigid_objects, scene.rigid_object_collections):
            for asset in assets.values():
                validate_configuration_coverage(asset.cfg)
                for cfg in getattr(asset.cfg, "rigid_objects", {}).values():
                    validate_configuration_coverage(cfg)
                for name, cfg in getattr(asset.cfg, "actuators", {}).items():
                    validate_configuration_coverage(cfg, actuator=True)
                    if not _is_implicit_actuator_cfg(cfg) and name not in asset.actuators.usd_actuator_groups:
                        raise NotImplementedError(f"Controller {name!r} has no native USD representation.")
                paths = adapter.paths(asset)
                body = BodyInitialState.from_data(asset.data)
                values = {}
                for rule in JOINT_USD_PROPERTIES.values() if paths.joints else ():
                    if rule.source in values:
                        continue
                    value = getattr(asset.data, rule.source, None)
                    if value is None:
                        if rule.absent_value is None:
                            raise NotImplementedError(f"Missing required initialized property {rule.source}.")
                        values[rule.source] = np.full(asset.num_joints, rule.absent_value)
                    else:
                        values[rule.source] = value.torch[0].detach().cpu().numpy().copy()
                if sorted(row for _, row in paths.bodies) != list(range(len(body.pose))):
                    raise RuntimeError(f"Incomplete body identities for {asset.cfg.prim_path}.")
                if asset in scene.articulations.values() and sorted(row for _, row in paths.joints) != list(
                    range(asset.num_joints)
                ):
                    raise RuntimeError(f"Incomplete DOF identities for {asset.cfg.prim_path}.")
                records.append((paths, body, values))
        self.timings["read"] = perf_counter() - start
        start = perf_counter()
        stage = copy_scene_stage(scene.sim.stage)
        self.timings["flatten"] = perf_counter() - start
        if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdPhysics.GetStageKilogramsPerUnit(stage) != 1:
            raise NotImplementedError("Fixed export requires SI stage units.")
        start = perf_counter()
        written = set()
        for paths, body, values in records:
            for path, row in sorted(paths.bodies, key=lambda item: Sdf.Path(item[0]).pathElementCount):
                prim = stage.GetPrimAtPath(path)
                if path in written or not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise RuntimeError(f"Missing or multiply owned body {path}.")
                author_body_state(prim, body.pose[row], body.velocity[row])
                written.add(path)
            for path, row in paths.joints:
                prim = stage.GetPrimAtPath(path)
                if not prim:
                    raise RuntimeError(f"Missing joint {path}.")
                write_properties(prim, values, row, JOINT_USD_PROPERTIES)
        author_fixed_root_frames(stage, written)
        adapter.write_extensions(stage)
        frequency = 1 / scene.sim.get_physics_dt()
        if not math.isclose(frequency, round(frequency), rel_tol=1e-6):
            raise NotImplementedError("PhysX USD timeStepsPerSecond cannot represent this timestep.")
        physics = stage.GetPrimAtPath(scene.physics_scene_path)
        physics.AddAppliedSchema("PhysxSceneAPI")
        physics.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(round(frequency))
        stage.GetRootLayer().customLayerData = {
            **stage.GetRootLayer().customLayerData,
            "isaaclab:configuration": "fixed initialization before task events; controller/sensor runtime excluded",
        }
        self.timings["author"] = perf_counter() - start
        start = perf_counter()
        check_body_coverage(stage, written)
        validate_dependencies(stage)
        self.timings["validate"] = perf_counter() - start
        start = perf_counter()
        result = save_stage(stage, usd_path, validate=False)
        self.timings["save"] = perf_counter() - start
        return result


def copy_scene_stage(stage: Usd.Stage) -> Usd.Stage:
    """Flatten composition and make instances writable without touching the source stage."""
    snapshot = Usd.Stage.Open(stage.Flatten())
    while True:
        instances = [prim for prim in snapshot.Traverse() if prim.IsInstance()]
        if not instances:
            return Usd.Stage.Open(snapshot.Flatten())
        for prim in instances:
            prim.SetInstanceable(False)


def validate_dependencies(stage: Usd.Stage) -> None:
    """Reject dangling prim/property connections and unresolved external resources."""
    for prim in stage.Traverse():
        for prop in prim.GetProperties():
            targets = prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.GetConnections()
            for path in targets:
                target = stage.GetPropertyAtPath(path) if path.IsPropertyPath() else stage.GetPrimAtPath(path)
                if not target:
                    raise RuntimeError(f"Unresolved export dependency {prop.GetPath()}: {path}")
    _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(stage.GetRootLayer().identifier))
    unresolved = [p for p in unresolved if p not in {"OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl"}]
    if unresolved:
        raise RuntimeError(f"Unresolved export asset dependencies: {unresolved}")


def save_stage(stage: Usd.Stage, usd_path: str, *, validate: bool = True) -> str:
    """Save a complete snapshot atomically, leaving an existing destination intact on failure."""
    if validate:
        validate_dependencies(stage)
    destination = os.path.abspath(usd_path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    suffix = os.path.splitext(destination)[1]
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(destination), suffix=suffix, delete=False) as stream:
        temporary = stream.name
    try:
        if not stage.Export(temporary):
            raise RuntimeError(f"Could not export USD to {destination}.")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(usd_path)


def author_world_transform(prim: Usd.Prim, transform: Gf.Matrix4d) -> None:
    """Author a world transform in stage units while preserving the existing parent hierarchy."""
    parent = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
    local = transform * parent.GetInverse()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp(opSuffix="export").Set(local)


def check_body_coverage(stage: Usd.Stage, written: set[str]) -> None:
    """Reject physical bodies retained from USD without a corresponding initialized backend data."""
    authored = {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
    }
    if authored != written:
        raise RuntimeError(
            f"Incomplete body export: missing={sorted(authored - written)}, extra={sorted(written - authored)}"
        )


def author_fixed_root_frames(stage: Usd.Stage, body_paths: set[str]) -> None:
    """Update world anchors moved by fixed-base default root-pose initialization."""
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if not body0 and len(body1) == 1 and str(body1[0]) in body_paths:
            other, world = 1, 0
            body = body1[0]
        elif not body1 and len(body0) == 1 and str(body0[0]) in body_paths:
            other, world = 0, 1
            body = body0[0]
        else:
            continue
        position = prim.GetAttribute(f"physics:localPos{other}").Get()
        rotation = prim.GetAttribute(f"physics:localRot{other}").Get()
        local = Gf.Matrix4d(1.0).SetRotate(Gf.Quatd(rotation))
        local.SetTranslateOnly(Gf.Vec3d(position))
        anchor = local * cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body))
        prim.GetAttribute(f"physics:localPos{world}").Set(Gf.Vec3f(anchor.ExtractTranslation()))
        prim.GetAttribute(f"physics:localRot{world}").Set(Gf.Quatf(anchor.ExtractRotationQuat()))


def author_body_state(prim: Usd.Prim, pose: np.ndarray, velocity: np.ndarray) -> None:
    """Write body-link world pose [m, xyzw] and COM world velocity [m/s, rad/s]."""
    length = UsdGeom.GetStageMetersPerUnit(prim.GetStage())
    previous = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    scale = Gf.Transform(previous).GetScale()
    transform = Gf.Matrix4d(1.0).SetRotate(Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float, pose[3:6]))))
    transform.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3] / length)))
    author_world_transform(prim, Gf.Matrix4d().SetScale(scale) * transform)
    body = UsdPhysics.RigidBodyAPI(prim)
    body.CreateVelocityAttr().Set(Gf.Vec3f(*map(float, velocity[:3] / length)))
    body.CreateAngularVelocityAttr().Set(Gf.Vec3f(*map(float, np.rad2deg(velocity[3:]))))
