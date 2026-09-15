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
    UsdAttribute,
    usd_fields,
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

    def write_extensions(self, writer: UsdWriter) -> None:
        """Preserve source-specific initialized semantics or fail before saving."""
        ...


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


class UsdWriter:
    """Write declared fixed properties onto one preserved, isolated SI stage.

    Registered schemas provide exact target names and types, not source fields or
    units. Unregistered extensions need explicit declarations. Data properties use
    [environment, row, ...] layout and are read once per writer; getter discovery
    never evaluates them. Tasks, spawning and training are outside this object.
    """

    def __init__(self, stage: Usd.Stage, adapter: SceneExportAdapter | None = None):
        self.stage = stage
        self.adapter = adapter
        self.body_paths: set[str] = set()
        # Retain owners with their arrays, so ids cannot be reused during an export.
        self._values: dict[int, tuple[object, dict[str, np.ndarray]]] = {}

    def write_properties(self, path: str, axis: str | None, data: object, *, row: int) -> None:
        """Write inherited property declarations from environment zero at one data row."""
        declarations = usd_fields(type(data))
        if not declarations:
            raise NotImplementedError(f"No USD declarations on {type(data).__name__}.")
        values = self._values.setdefault(id(data), (data, {}))[1]
        for source, targets in declarations.items():
            if source not in values:
                value = getattr(data, source)
                if hasattr(value, "torch"):
                    value = value.torch.detach().cpu().numpy()
                array = np.asarray(value)
                if array.ndim < 2 or array.shape[0] != 1:
                    raise ValueError(f"Expected one environment and row dimension for {source}.")
                values[source] = array[0].copy()
            for target in targets:
                value = values[source][row]
                if target.component is not None:
                    value = value[target.component]
                self.write_attribute(path, target, value, axis=axis)

    def write_attribute(self, path: str, target: UsdAttribute, value, *, axis: str | None = None) -> None:
        """Write a scalar/vector target; schema names, instance kind and types must agree."""
        if (target.angular_power or "{axis}" in target.attribute or "{axis}" in (target.schema or "")) and axis not in {
            "angular",
            "linear",
        }:
            raise ValueError(f"Missing supported axis for {target.attribute}.")
        attr = self._attribute(path, target, axis)
        converted = np.asarray(value)
        if axis == "angular" and target.angular_power:
            converted = converted * (180 / math.pi) ** target.angular_power
        value_type = attr.GetTypeName()
        default = value_type.defaultValue
        dimension = getattr(default, "dimension", None)
        if value_type.isArray or (dimension is not None and not isinstance(dimension, int)):
            raise NotImplementedError(f"Use a dedicated writer for {value_type} at {attr.GetPath()}.")
        if dimension is not None:
            if converted.shape != (dimension,):
                raise ValueError(f"Expected {dimension} vector components for {attr.GetPath()}.")
            value = type(default)(*converted.tolist())
        elif converted.ndim == 0:
            value = converted.item()
        else:
            raise NotImplementedError(f"No scalar/vector conversion for {value_type} at {attr.GetPath()}.")
        if not attr.Set(value):
            raise RuntimeError(f"Could not author {attr.GetPath()}.")

    def _attribute(self, path: str, target: UsdAttribute, axis: str | None) -> Usd.Attribute:
        """Resolve the exact registered schema attribute or an explicitly typed extension."""
        prim = self.stage.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError(f"Missing export prim {path}.")
        name = target.attribute.format(axis=axis)
        if target.schema:
            schema, _, instance = target.schema.format(axis=axis).partition(":")
            schema_type = Usd.SchemaRegistry.GetTypeFromSchemaTypeName(schema)
            api = schema_type.pythonClass
            if api is not None:
                multiple = Usd.SchemaRegistry.IsMultipleApplyAPISchema(schema_type)
                if bool(instance) != multiple:
                    raise ValueError(f"Incorrect schema instance for {target.schema}.")
                names = api.GetSchemaAttributeNames(True, instance) if multiple else api.GetSchemaAttributeNames(True)
                if name not in names:
                    raise NotImplementedError(f"{name} is not an attribute of {target.schema}.")
                if Usd.SchemaRegistry.IsAppliedAPISchema(schema_type):
                    applied = api.Apply(prim, instance) if multiple else api.Apply(prim)
                    if not applied:
                        raise RuntimeError(f"Could not apply {target.schema} at {path}.")
                elif not prim.IsA(api):
                    raise ValueError(f"{path} is not a {schema} prim.")
            elif target.type_name is None:
                raise NotImplementedError(f"Unregistered schema {schema} requires an explicit target type.")
            elif not prim.AddAppliedSchema(target.schema.format(axis=axis)):
                raise RuntimeError(f"Could not apply extension {target.schema} at {path}.")
        attr = prim.GetAttribute(name)
        if target.type_name:
            value_type = Sdf.ValueTypeNames.Find(target.type_name)
            if not value_type or (attr and attr.GetTypeName() != value_type):
                raise ValueError(f"Invalid or conflicting type {target.type_name} for {path}.{name}.")
            if not attr:
                attr = prim.CreateAttribute(name, value_type, custom=False)
        if not attr:
            raise NotImplementedError(f"No declared USD attribute/type for {path}.{name}.")
        return attr

    def write_bodies(
        self,
        data: BaseArticulationData | BaseRigidObjectData | BaseRigidObjectCollectionData,
        paths: list[tuple[str, int]],
    ) -> None:
        """Write initialized public body state into its existing prims and track their identities."""
        poses = data.body_link_pose_w.torch[0].detach().cpu().numpy().reshape(-1, 7).copy()
        velocities = data.body_com_vel_w.torch[0].detach().cpu().numpy().reshape(-1, 6).copy()
        if sorted(row for _, row in paths) != list(range(len(poses))):
            raise RuntimeError("Incomplete body identities for fixed configuration.")
        for path, row in sorted(paths, key=lambda item: Sdf.Path(item[0]).pathElementCount):
            prim = self.stage.GetPrimAtPath(path)
            if path in self.body_paths or not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Missing or multiply owned body {path}.")
            self._write_body_state(prim, poses[row], velocities[row])
            self.body_paths.add(path)

    def write_scene_settings(self, scene: InteractiveScene) -> None:
        """Preserve fixed simulation timing and identify the configuration boundary."""
        frequency = 1 / scene.sim.get_physics_dt()
        if not math.isclose(frequency, round(frequency), rel_tol=1e-6):
            raise NotImplementedError("PhysX USD timeStepsPerSecond cannot represent this timestep.")
        physics = self.stage.GetPrimAtPath(scene.physics_scene_path)
        physics.AddAppliedSchema("PhysxSceneAPI")
        physics.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(round(frequency))
        self.stage.GetRootLayer().customLayerData = {
            **self.stage.GetRootLayer().customLayerData,
            "isaaclab:configuration": "fixed initialization before task events; controller/sensor runtime excluded",
        }

    def validate(self) -> None:
        """Reject physical bodies retained from USD without a corresponding initialized backend data."""
        authored = {
            str(prim.GetPath())
            for prim in self.stage.Traverse()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
        }
        if authored != self.body_paths:
            raise RuntimeError(
                f"Incomplete body export: missing={sorted(authored - self.body_paths)}, "
                f"extra={sorted(self.body_paths - authored)}"
            )
        self.validate_dependencies()

    def validate_dependencies(self) -> None:
        """Reject dangling prim/property connections and unresolved external resources."""
        for prim in self.stage.Traverse():
            for prop in prim.GetProperties():
                targets = prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.GetConnections()
                for path in targets:
                    target = (
                        self.stage.GetPropertyAtPath(path) if path.IsPropertyPath() else self.stage.GetPrimAtPath(path)
                    )
                    if not target:
                        raise RuntimeError(f"Unresolved export dependency {prop.GetPath()}: {path}")
        _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(self.stage.GetRootLayer().identifier))
        unresolved = [p for p in unresolved if p not in {"OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl"}]
        if unresolved:
            raise RuntimeError(f"Unresolved export asset dependencies: {unresolved}")

    def save(self, usd_path: str, *, validate: bool = True) -> str:
        """Save a complete snapshot atomically, leaving an existing destination intact on failure."""
        if validate:
            self.validate_dependencies()
        destination = os.path.abspath(usd_path)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        suffix = os.path.splitext(destination)[1]
        with tempfile.NamedTemporaryFile(dir=os.path.dirname(destination), suffix=suffix, delete=False) as stream:
            temporary = stream.name
        try:
            if not self.stage.Export(temporary):
                raise RuntimeError(f"Could not export USD to {destination}.")
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return str(usd_path)

    def write_fixed_root_frames(self) -> None:
        """Update world anchors moved by fixed-base default root-pose initialization."""
        cache = UsdGeom.XformCache()
        for prim in self.stage.Traverse():
            if not prim.IsA(UsdPhysics.FixedJoint):
                continue
            joint = UsdPhysics.Joint(prim)
            body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not body0 and len(body1) == 1 and str(body1[0]) in self.body_paths:
                other, world = 1, 0
                body = body1[0]
            elif not body1 and len(body0) == 1 and str(body0[0]) in self.body_paths:
                other, world = 0, 1
                body = body0[0]
            else:
                continue
            position = prim.GetAttribute(f"physics:localPos{other}").Get()
            rotation = prim.GetAttribute(f"physics:localRot{other}").Get()
            local = Gf.Matrix4d(1.0).SetRotate(Gf.Quatd(rotation))
            local.SetTranslateOnly(Gf.Vec3d(position))
            anchor = local * cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(body))
            prim.GetAttribute(f"physics:localPos{world}").Set(Gf.Vec3f(anchor.ExtractTranslation()))
            prim.GetAttribute(f"physics:localRot{world}").Set(Gf.Quatf(anchor.ExtractRotationQuat()))

    def _write_body_state(self, prim: Usd.Prim, pose: np.ndarray, velocity: np.ndarray) -> None:
        """Write body-link world pose [m, xyzw] and COM world velocity [m/s, rad/s]."""
        length = UsdGeom.GetStageMetersPerUnit(prim.GetStage())
        previous = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        scale = Gf.Transform(previous).GetScale()
        transform = Gf.Matrix4d(1.0).SetRotate(Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float, pose[3:6]))))
        transform.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3] / length)))
        parent = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
        local = Gf.Matrix4d().SetScale(scale) * transform * parent.GetInverse()
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp(opSuffix="export").Set(local)
        body = UsdPhysics.RigidBodyAPI(prim)
        body.CreateVelocityAttr().Set(Gf.Vec3f(*map(float, velocity[:3] / length)))
        body.CreateAngularVelocityAttr().Set(Gf.Vec3f(*map(float, np.rad2deg(velocity[3:]))))
