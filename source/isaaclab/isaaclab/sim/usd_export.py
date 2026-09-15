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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

from isaaclab.assets.physics_properties import (
    JOINT_PROPERTY_SOURCES,
    JOINT_USD_PROPERTIES,
    UsdProperty,
)

if TYPE_CHECKING:
    from isaaclab.assets import (
        BaseArticulation,
        BaseArticulationData,
        BaseRigidObject,
        BaseRigidObjectCollection,
        BaseRigidObjectCollectionData,
        BaseRigidObjectData,
    )
    from isaaclab.scene import InteractiveScene


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


def author_bodies(
    stage: Usd.Stage,
    data: BaseArticulationData | BaseRigidObjectData | BaseRigidObjectCollectionData,
    paths: list[tuple[str, int]],
) -> set[str]:
    """Write initialized public body state into its existing prims and return their identities."""
    poses = data.body_link_pose_w.torch[0].detach().cpu().numpy().reshape(-1, 7).copy()
    velocities = data.body_com_vel_w.torch[0].detach().cpu().numpy().reshape(-1, 6).copy()
    if sorted(row for _, row in paths) != list(range(len(poses))):
        raise RuntimeError("Incomplete body identities for fixed configuration.")
    written = set()
    for path, row in sorted(paths, key=lambda item: Sdf.Path(item[0]).pathElementCount):
        prim = stage.GetPrimAtPath(path)
        if path in written or not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Missing or multiply owned body {path}.")
        author_body_state(prim, poses[row], velocities[row])
        written.add(path)
    return written


def author_joints(stage: Usd.Stage, data: BaseArticulationData, paths: list[tuple[str, int]]) -> None:
    """Write public fixed joint properties using the shared semantic mapping."""
    if set(JOINT_PROPERTY_SOURCES) - JOINT_USD_PROPERTIES.keys():
        raise NotImplementedError("Actuator initialization properties lack USD mappings.")
    count = data.joint_pos.shape[1]
    if sorted(row for _, row in paths) != list(range(count)):
        raise RuntimeError("Incomplete DOF identities for fixed configuration.")
    values = {}
    for rule in JOINT_USD_PROPERTIES.values():
        if rule.source in values:
            continue
        value = getattr(data, rule.source, None)
        if value is None:
            if rule.absent_value is None:
                raise NotImplementedError(f"Missing required initialized property {rule.source}.")
            values[rule.source] = np.full(count, rule.absent_value)
        else:
            values[rule.source] = value.torch[0].detach().cpu().numpy().copy()
    for path, row in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError(f"Missing joint {path}.")
        write_properties(prim, values, row, JOINT_USD_PROPERTIES)


def author_scene_settings(stage: Usd.Stage, scene: InteractiveScene) -> None:
    """Preserve fixed simulation timing and identify the configuration boundary."""
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


def copy_scene_stage(stage: Usd.Stage) -> Usd.Stage:
    """Flatten composition and make instances writable without touching the source stage."""
    if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdPhysics.GetStageKilogramsPerUnit(stage) != 1:
        raise NotImplementedError("Fixed export requires SI stage units.")
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
