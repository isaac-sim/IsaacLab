# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Complete fixed single-environment USD export in the PhysX deployment dialect.

Authored USD and one-time initialization results are preserved in an isolated copy
of the selected environment. Subsequent training changes are outside this contract.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Set
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt

from isaaclab.assets.physics_properties import (
    UsdAttribute,
    property_metadata,
    usd_fields,
)

if TYPE_CHECKING:
    from isaaclab.cloner.clone_plan import ClonePlan


def validate_stage_units(stage: Usd.Stage) -> None:
    """Require the meter/kilogram stage convention used by initialized Isaac Lab scenes."""
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdPhysics.GetStageKilogramsPerUnit(stage) != 1.0:
        raise ValueError("Deployment USD requires metersPerUnit=1 and kilogramsPerUnit=1.")


@dataclass(frozen=True)
class AssetPaths:
    """Prim identities paired with public body/DOF row indices in one environment."""

    bodies: list[tuple[str, int]]
    joints: list[tuple[str, int]]
    joint_axes: dict[int, str] = field(default_factory=dict)

    @staticmethod
    def rows(paths, native_names: list[str], public_names: list[str]) -> list[tuple[str, int]]:
        """Align concrete native identities with the public data order."""
        rows = {name: row for row, name in enumerate(public_names)}
        if len(rows) != len(public_names) or len(paths) != len(native_names):
            raise RuntimeError("Ambiguous or incomplete native physical identities.")
        return [(str(path), rows[name]) for path, name in zip(paths, native_names)]

    @staticmethod
    def scalar_joint_axis(prim: Usd.Prim) -> str | None:
        """Resolve standard scalar USD joints; native adapters decode multi-axis joints."""
        if prim.IsA(UsdPhysics.RevoluteJoint):
            return "angular"
        if prim.IsA(UsdPhysics.PrismaticJoint):
            return "linear"
        return None


class UsdWriter:
    """Write declared fixed properties onto one preserved, isolated stage.

    Registered schemas provide exact target names and types, not source fields or
    units. Unregistered extensions need explicit declarations. Data properties use
    [environment, row, ...] layout and are read once per writer; getter discovery
    never evaluates them. Tasks, spawning and training are outside this object.
    """

    ##
    # Stage preparation and identities.
    ##

    def __init__(self, stage: Usd.Stage):
        validate_stage_units(stage)
        self.stage = stage
        self.env_index = 0
        self.preserve_source_contacts = False
        self.include_solver_settings = False
        self.env_id = 0
        self.clone_plan: ClonePlan | None = None
        self.body_paths: set[str] = set()
        self._moved_body_paths: set[str] = set()
        self.represented_collider_paths: set[str] = set()
        self._joint_shared_values: dict[tuple[str, str], float] = {}
        # Retain owners with their arrays, so ids cannot be reused during an export.
        self._values: dict[int, tuple[object, dict[str, np.ndarray]]] = {}
        self._source_missing_bindings: set[tuple[str, Sdf.Path]] = set()

    @classmethod
    def from_stage(cls, stage: Usd.Stage) -> UsdWriter:
        """Preserve composition in an isolated writable stage without changing the source."""
        writer = cls(Usd.Stage.Open(stage.Flatten()))
        # Preserve ineffective source bindings rather than rewriting unrelated scene content.
        for prim in writer.stage.Traverse(Usd.TraverseInstanceProxies()):
            for rel in prim.GetRelationships():
                if rel.GetName().startswith("material:binding"):
                    writer._source_missing_bindings.update(
                        (str(rel.GetPath()), target)
                        for target in rel.GetTargets()
                        if target.IsPrimPath() and not writer.stage.GetPrimAtPath(target)
                    )
        return writer

    def writable_prim(self, path: str | Sdf.Path) -> Usd.Prim:
        """Expand only instance ancestors of a prim that actually needs an edit."""
        prim = self.stage.GetPrimAtPath(path)
        while prim and prim.IsInstanceProxy():
            parent = prim.GetParent()
            while parent.IsInstanceProxy():
                parent = parent.GetParent()
            parent.SetInstanceable(False)
            prim = self.stage.GetPrimAtPath(path)
        return prim

    def select_environment(self, plan: ClonePlan | None, env_id: int, env_paths: list[str]) -> None:
        """Materialize the selected USD variant and retain its shared resources.

        Physics-only replication may omit clone prims from USD. Copy their authored
        prototype before writing effective buffers. The plan supplies layout only.
        """
        from isaaclab.cloner.query import path_to_source

        self.clone_plan, self.env_id = plan, env_id
        layer = self.stage.GetRootLayer()
        if plan is not None:
            for template in dict.fromkeys(plan.destinations):
                destination = template.format(env_id)
                source = path_to_source(plan, destination, env_id)
                if source is None:
                    continue
                source_path = source[0] + source[2]
                source_prim = self.stage.GetPrimAtPath(source_path)
                if not source_prim:
                    raise RuntimeError(f"Missing source variant {source_path}.")
                for prim in list(Usd.PrimRange(source_prim)):
                    target = prim.GetPath().ReplacePrefix(Sdf.Path(source_path), Sdf.Path(destination))
                    # Keep authored clone overrides; copy only missing prototype content.
                    if self.stage.GetPrimAtPath(target):
                        continue
                    Sdf.CreatePrimInLayer(layer, target.GetParentPath())
                    if not Sdf.CopySpec(layer, prim.GetPath(), layer, target):
                        raise RuntimeError(f"Cannot materialize {prim.GetPath()} at {target}.")
                    self._rebase_connections(self.stage.GetPrimAtPath(target), source_path, destination)
            root = self.stage.GetPrimAtPath(env_paths[env_id])
            if root and plan.positions is not None:
                column = list(plan.env_ids).index(env_id)
                # Clone roots carry layout translation, including physics-only copies.
                UsdGeom.XformCommonAPI(root).SetTranslate(
                    Gf.Vec3d(*map(float, plan.positions[column] / UsdGeom.GetStageMetersPerUnit(self.stage)))
                )

        excluded = tuple(Sdf.Path(path) for index, path in enumerate(env_paths) if index != env_id)
        # Relationship dependencies may live in a different prototype environment.
        # Retain resources, but never pull a foreign physical entity into the export.
        copied = {}
        pending = list(self.stage.Traverse(Usd.TraverseInstanceProxies()))
        for prim in pending:
            if any(prim.GetPath().HasPrefix(root) for root in excluded):
                continue
            for prop in prim.GetProperties():
                relationship = isinstance(prop, Usd.Relationship)
                targets = prop.GetTargets() if relationship else prop.GetConnections()
                rewritten = []
                for target in targets:
                    if not any(target.HasPrefix(root) for root in excluded):
                        rewritten.append(target)
                        continue
                    # Cross-environment collision isolation has no foreign members now.
                    if prim.IsA(UsdPhysics.CollisionGroup) or prop.GetName() == "physics:filteredPairs":
                        continue
                    source = target.GetPrimPath()
                    dependency = self.stage.GetPrimAtPath(source)
                    if not dependency or any(
                        child.HasAPI(UsdPhysics.RigidBodyAPI) or child.HasAPI(UsdPhysics.CollisionAPI)
                        for child in Usd.PrimRange(dependency)
                    ):
                        raise NotImplementedError(f"Cross-environment physical dependency {prop.GetPath()}: {target}")
                    destination = copied.get(source)
                    if destination is None:
                        destination = Sdf.Path("/__ExportResources").AppendChild(f"resource_{len(copied)}")
                        UsdGeom.Scope.Define(self.stage, destination.GetParentPath())
                        if not Sdf.CopySpec(layer, source, layer, destination):
                            raise RuntimeError(f"Cannot preserve dependency {source}.")
                        copied[source] = destination
                        self._rebase_connections(self.stage.GetPrimAtPath(destination), source, destination)
                        # Newly copied resources can themselves depend on excluded prototypes.
                        pending.extend(Usd.PrimRange(self.stage.GetPrimAtPath(destination)))
                    rewritten.append(target.ReplacePrefix(source, destination))
                if rewritten != targets:
                    prim = self.writable_prim(prim.GetPath())
                    prop = prim.GetProperty(prop.GetName())
                    prop.SetTargets(rewritten) if relationship else prop.SetConnections(rewritten)
        for root in excluded:
            self.stage.RemovePrim(root)

    @staticmethod
    def _rebase_connections(root: Usd.Prim, source: str | Sdf.Path, destination: str | Sdf.Path) -> None:
        """Rebase internal relationships and shader connections in a copied subtree."""
        source, destination = Sdf.Path(source), Sdf.Path(destination)
        for prim in Usd.PrimRange(root):
            for prop in prim.GetProperties():
                relationship = isinstance(prop, Usd.Relationship)
                targets = prop.GetTargets() if relationship else prop.GetConnections()
                rebased = [target.ReplacePrefix(source, destination) for target in targets]
                if rebased != targets:
                    prop.SetTargets(rebased) if relationship else prop.SetConnections(rebased)

    def resolve_paths(self, paths: AssetPaths) -> AssetPaths:
        """Map native prototype identities to this environment's authored clone paths."""
        from isaaclab.cloner.query import path_to_clone, path_to_source

        def resolve(path):
            if self.clone_plan is None:
                return path
            source = path_to_source(self.clone_plan, path)
            prototype = source[0] + source[2] if source is not None else path
            return path_to_clone(self.clone_plan, prototype, self.env_id) or path

        return AssetPaths(
            [(resolve(path), row) for path, row in paths.bodies],
            [(resolve(path), row) for path, row in paths.joints],
            paths.joint_axes.copy(),
        )

    ##
    # Declared physical properties.
    ##

    def _property_values(self, data: object, source: str) -> np.ndarray:
        """Read one public property once, preserving the owner's environment/row layout."""
        values = self._values.setdefault(id(data), (data, {}))[1]
        if source not in values:
            value = getattr(data, source)
            if hasattr(value, "torch"):
                value = value.torch.detach().cpu().numpy()
            values[source] = np.asarray(value).copy()
        return values[source]

    def bound_properties(self, path, axis, data, row, scope, env_index, *, fields: Set[str] | None = None):
        """Yield converted declaration/value pairs for writing or equivalence checks."""
        if fields is not None and not fields:
            return
        declarations = usd_fields(type(data), scope)
        if not declarations:
            raise NotImplementedError(f"No {scope} USD declarations on {type(data).__name__}.")
        for source, targets in declarations.items():
            if fields is not None and source not in fields:
                continue
            values = self._property_values(data, source)[env_index]
            value = values if row is None else values[row]
            _, _, _, transform, inputs, angular_conversion = property_metadata(type(data), source, "_usd_field")
            arguments = [self._property_values(data, name)[env_index][row] for name in inputs]
            outputs = transform(value, *arguments) if transform is not None else None
            for target in targets:
                if target.condition and not self._property_values(data, target.condition)[env_index][row]:
                    continue
                if target.axes is not None and axis not in target.axes:
                    continue
                selected = outputs[target.output] if outputs is not None else value
                if target.component is not None:
                    selected = selected[target.component]
                # USD uses degrees for angular coordinates, and inverse degrees for gains.
                if angular_conversion is not None and axis in {"angular", "rotX", "rotY", "rotZ"}:
                    selected = angular_conversion(selected)
                yield target, selected

    def write_properties(
        self,
        path: str,
        axis: str | None,
        data: object,
        *,
        row: int,
        scope: str = "joint",
        env_index: int | None = None,
        fields: Set[str] | None = None,
    ) -> None:
        """Write declared properties from one public data row using the declared angular conversion."""
        for target, value in self.bound_properties(
            path, axis, data, row, scope, self.env_index if env_index is None else env_index, fields=fields
        ):
            self.write_attribute(path, target, value, axis=axis)

    def write_array_properties(
        self, path: str, data: object, *, env_index: int, fields: Set[str] | None = None
    ) -> None:
        """Write an owner's complete array row, without interpreting its element identities."""
        for target, value in self.bound_properties(path, None, data, None, "array", env_index, fields=fields):
            self.write_attribute(path, target, value)

    def register_bodies(self, paths: list[tuple[str, int]], count: int) -> None:
        """Account for every body independently of whether its properties need writes."""
        if sorted(row for _, row in paths) != list(range(count)):
            raise RuntimeError("Incomplete body identities for fixed configuration.")
        for path, _ in paths:
            prim = self.stage.GetPrimAtPath(path)
            if path in self.body_paths or not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Missing or multiply owned body {path}.")
            self.body_paths.add(path)

    def write_root_placement(self, data, paths: list[tuple[str, int]]) -> None:
        """Apply explicitly selected root placements without baking articulation child poses."""
        poses = self._property_values(data, "body_link_pose_w")[self.env_index].reshape(-1, 7)
        for path, row in paths:
            self._write_body_pose(self.stage.GetPrimAtPath(path), poses[row])

    def write_bodies(self, data, paths: list[tuple[str, int]]) -> None:
        """Write public body placement and declared mass properties, parents first."""
        poses = self._property_values(data, "body_link_pose_w")[self.env_index].reshape(-1, 7)
        if sorted(row for _, row in paths) != list(range(len(poses))):
            raise RuntimeError("Incomplete body identities for fixed configuration.")
        # Parent transforms must be updated before converting a child world pose to local space.
        for path, row in sorted(paths, key=lambda item: Sdf.Path(item[0]).pathElementCount):
            prim = self.stage.GetPrimAtPath(path)
            if path in self.body_paths or not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Missing or multiply owned body {path}.")
            self._write_body_pose(prim, poses[row])
            self.write_properties(path, None, data, row=row, scope="body")
            self.body_paths.add(path)

    def _write_body_pose(self, prim: Usd.Prim, pose: np.ndarray) -> None:
        """Write body-link world placement [m, xyzw], retaining geometry scale."""
        length = UsdGeom.GetStageMetersPerUnit(prim.GetStage())
        previous = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        scale = Gf.Transform(previous).GetScale()
        transform = Gf.Matrix4d(1.0).SetRotate(Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float, pose[3:6]))))
        transform.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3] / length)))
        parent = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
        # Retain geometry scale while replacing the physical pose relative to its parent.
        local = Gf.Matrix4d().SetScale(scale) * transform * parent.GetInverse()
        if np.allclose(np.asarray(previous * parent.GetInverse()), np.asarray(local), rtol=1e-6, atol=1e-7):
            return
        self._moved_body_paths.add(str(prim.GetPath()))
        prim = self.writable_prim(prim.GetPath())
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp(opSuffix="export").Set(local)

    def write_attribute(self, path: str, target: UsdAttribute, value, *, axis: str | None = None) -> None:
        """Write a scalar, vector or scalar-array target with a declared schema/type."""
        if target.axes is not None and axis not in target.axes:
            return
        # Several native axes can share one scalar USD target; never silently keep the last value.
        if target.require_uniform:
            key = (path, target.attribute.format(axis=axis))
            previous = self._joint_shared_values.setdefault(key, np.asarray(value).copy())
            if not np.array_equal(previous, value):
                raise NotImplementedError(f"Distinct per-axis values cannot share {key}.")
        if ("{axis}" in target.attribute or "{axis}" in (target.schema or "")) and axis not in {
            "angular",
            "linear",
            "rotX",
            "rotY",
            "rotZ",
            "transX",
            "transY",
            "transZ",
        }:
            raise ValueError(f"Missing supported axis for {target.attribute}.")
        attr = self._attribute(path, target, axis, author=False)
        if attr and self.equivalent(attr.Get(), value):
            return
        prim = self.writable_prim(path)
        for name in target.replaces:
            prim.RemoveProperty(name)
        attr = self._attribute(path, target, axis)
        converted = np.asarray(value)
        value_type = attr.GetTypeName()
        default = value_type.defaultValue
        if isinstance(default, (Gf.Quatf, Gf.Quatd)):
            if converted.shape != (4,):
                raise ValueError(f"Expected xyzw quaternion for {attr.GetPath()}.")
            vector = Gf.Vec3f if isinstance(default, Gf.Quatf) else Gf.Vec3d
            attr.Set(type(default)(float(converted[3]), vector(*map(float, converted[:3]))))
            return
        dimension = getattr(default, "dimension", None)
        if value_type.isArray:
            if converted.ndim != 1:
                raise NotImplementedError(f"Only scalar arrays are supported at {attr.GetPath()}.")
            if not attr.Set(type(default)(converted.tolist())):
                raise RuntimeError(f"Could not author {attr.GetPath()}.")
            return
        if dimension is not None and not isinstance(dimension, int):
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

    @staticmethod
    def equivalent(authored, value) -> bool:
        """Compare physical values without turning float32 conversion noise into edits."""
        if authored is None:
            return False
        if isinstance(authored, (Gf.Quatf, Gf.Quatd)):
            authored = np.array([*authored.GetImaginary(), authored.GetReal()])
            return UsdWriter.equivalent(authored, value) or UsdWriter.equivalent(-authored, value)
        first, second = np.asarray(authored), np.asarray(value)
        if first.shape != second.shape:
            return False
        if first.dtype.kind not in "fc" and second.dtype.kind not in "fc":
            return bool(np.array_equal(first, second))
        # Relative-only tolerance preserves small nonzero quantities and automatic sentinels.
        return bool(np.allclose(first, second, rtol=8 * np.finfo(np.float32).eps, atol=0, equal_nan=False))

    def _attribute(self, path: str, target: UsdAttribute, axis: str | None, *, author: bool = True) -> Usd.Attribute:
        """Resolve the exact registered schema attribute or an explicitly typed extension."""
        prim = self.stage.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError(f"Missing export prim {path}.")
        name = target.attribute.format(axis=axis)
        needs_schema = False
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
                    needs_schema = not (prim.HasAPI(api, instance) if multiple else prim.HasAPI(api))
                    if author:
                        applied = api.Apply(prim, instance) if multiple else api.Apply(prim)
                        if not applied:
                            raise RuntimeError(f"Could not apply {target.schema} at {path}.")
                elif not prim.IsA(api):
                    raise ValueError(f"{path} is not a {schema} prim.")
            elif target.type_name is None:
                raise NotImplementedError(f"Unregistered schema {schema} requires an explicit target type.")
            elif author and not prim.AddAppliedSchema(target.schema.format(axis=axis)):
                raise RuntimeError(f"Could not apply extension {target.schema} at {path}.")
        attr = prim.GetAttribute(name)
        if target.type_name:
            value_type = Sdf.ValueTypeNames.Find(target.type_name)
            if not value_type or (attr and attr.GetTypeName() != value_type):
                raise ValueError(f"Invalid or conflicting type {target.type_name} for {path}.{name}.")
            if author and not attr:
                attr = prim.CreateAttribute(name, value_type, custom=False)
        if author and not attr:
            raise NotImplementedError(f"No declared USD attribute/type for {path}.{name}.")
        # An attribute without its required API is not an effective physics opinion.
        return Usd.Attribute() if needs_schema and not author else attr

    def write_fixed_root_frames(self) -> None:
        """Update world anchors moved by fixed-base default root-pose initialization."""
        cache = UsdGeom.XformCache()
        for prim in self.stage.Traverse(Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdPhysics.FixedJoint):
                continue
            joint = UsdPhysics.Joint(prim)
            body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not body0 and len(body1) == 1 and str(body1[0]) in self._moved_body_paths:
                other, world = 1, 0
                body = body1[0]
            elif not body1 and len(body0) == 1 and str(body0[0]) in self._moved_body_paths:
                other, world = 0, 1
                body = body0[0]
            else:
                continue
            position = prim.GetAttribute(f"physics:localPos{other}").Get()
            rotation = prim.GetAttribute(f"physics:localRot{other}").Get()
            local = Gf.Matrix4d(1.0).SetRotate(Gf.Quatd(rotation))
            local.SetTranslateOnly(Gf.Vec3d(position))
            anchor = local * cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(body))
            rotation = anchor.ExtractRotationQuat()
            self.write_attribute(
                str(prim.GetPath()),
                UsdAttribute(f"physics:localPos{world}", "PhysicsJoint"),
                anchor.ExtractTranslation(),
            )
            self.write_attribute(
                str(prim.GetPath()),
                UsdAttribute(f"physics:localRot{world}", "PhysicsJoint"),
                [*rotation.GetImaginary(), rotation.GetReal()],
            )

    def clear_initial_velocities(self) -> None:
        """Start deployment at rest, including velocities inherited from source layers."""
        axes = {"angular", "linear", "rotX", "rotY", "rotZ", "transX", "transY", "transZ"}
        for prim in self.stage.Traverse():
            if prim.IsA(UsdGeom.PointBased):
                points = UsdGeom.PointBased(prim)
                if points.GetVelocitiesAttr().HasAuthoredValue():
                    points.GetVelocitiesAttr().Clear()
                    points.CreateVelocitiesAttr().Set(Vt.Vec3fArray(len(points.GetPointsAttr().Get() or [])))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body = UsdPhysics.RigidBodyAPI(prim)
                for attr in (body.CreateVelocityAttr(), body.CreateAngularVelocityAttr()):
                    attr.Clear()
                    attr.Set(Gf.Vec3f(0))
            if prim.IsA(UsdPhysics.Joint):
                for attr in prim.GetAttributes():
                    tokens = attr.GetName().split(":")
                    if (
                        len(tokens) == 4
                        and tokens[0] == "state"
                        and tokens[1] in axes
                        and tokens[2:] == ["physics", "velocity"]
                    ) or (len(tokens) == 3 and tokens[0] == "newton" and tokens[1] in axes and tokens[2] == "velocity"):
                        attr.Clear()
                        attr.Set(0.0)

    def write_physx_timestep(self, scene_path: str, dt: float) -> None:
        """Author the PhysX integer scene frequency from a timestep [s]."""
        frequency = 1 / dt
        if not math.isclose(frequency, round(frequency), rel_tol=1e-6):
            raise NotImplementedError("PhysX USD timeStepsPerSecond cannot represent this timestep.")
        self.write_attribute(
            scene_path,
            UsdAttribute("physxScene:timeStepsPerSecond", "PhysxSceneAPI", type_name="uint"),
            round(frequency),
        )

    def write_gravity(self, scene_path: str, gravity) -> None:
        """Author effective gravity [m/s²] from a backend's selected world."""
        gravity = np.asarray(gravity, dtype=float)
        magnitude = float(np.linalg.norm(gravity))
        physics = UsdPhysics.Scene(self.stage.GetPrimAtPath(scene_path))
        direction = physics.GetGravityDirectionAttr().Get()
        previous = physics.GetGravityMagnitudeAttr().Get()
        if previous is not None and direction is not None and np.linalg.norm(direction):
            if self.equivalent(np.asarray(direction) / np.linalg.norm(direction) * previous, gravity):
                return
        self.write_attribute(scene_path, UsdAttribute("physics:gravityMagnitude", "PhysicsScene"), magnitude)
        self.write_attribute(
            scene_path,
            UsdAttribute("physics:gravityDirection", "PhysicsScene"),
            gravity / magnitude if magnitude else (0, 0, -1),
        )

    ##
    # Contacts and materials.
    ##

    def write_body_contacts(self, data, body_path: str) -> None:
        """Write resolved body flags and unambiguous contacts without crossing into child bodies."""
        self.write_properties(body_path, None, data, row=0, scope="body", env_index=0)
        if self.preserve_source_contacts:
            return
        body = self.stage.GetPrimAtPath(body_path)
        descendants = iter(Usd.PrimRange(body))
        colliders = []
        for prim in descendants:
            # Nested rigid bodies own their own shape buffers.
            if prim != body and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                descendants.PruneChildren()
            elif prim.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(str(prim.GetPath()))
        if colliders:
            data.validate_uniform_contacts(body_path)
        for path in colliders:
            self.write_properties(path, None, data, row=0, scope="collision", env_index=0)
            self.write_material_override(path, data, 0, env_index=0)

    def write_material_override(
        self, collider_path: str, data: object, row: int, *, env_index: int | None = None
    ) -> None:
        """Preserve an equivalent binding; otherwise copy and write declared material values."""
        env_index = self.env_index if env_index is None else env_index
        collider = self.stage.GetPrimAtPath(collider_path)
        original, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
        if original:
            values = self.bound_properties(str(original.GetPath()), None, data, row, "material", env_index)
            if all(
                original.GetPrim().GetAttribute(target.attribute).Get() is not None
                and self.equivalent(original.GetPrim().GetAttribute(target.attribute).Get(), value)
                for target, value in values
            ):
                return
        material = self.material_for_override(collider)
        self.write_properties(str(material.GetPath()), None, data, row=row, scope="material", env_index=env_index)

    def write_material_overrides(self, data: object, selections: dict[str, tuple[int, frozenset[str]]]) -> None:
        """Share material edits only when every retained consumer requests the same override."""
        groups = {}
        consumers = {}
        for prim in self.stage.Traverse(Usd.TraverseInstanceProxies()):
            # Unknown or non-rigid consumers must not inherit a collider-only material edit.
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            if material:
                consumers.setdefault(str(material.GetPath()), set()).add(str(prim.GetPath()))
        for path, (row, fields) in selections.items():
            material, _ = UsdShade.MaterialBindingAPI(self.stage.GetPrimAtPath(path)).ComputeBoundMaterial("physics")
            source = str(material.GetPath()) if material else ""
            values = tuple(
                (target, float(value))
                for target, value in self.bound_properties(path, None, data, row, "material", 0, fields=fields)
            )
            if source and all(
                self.equivalent(material.GetPrim().GetAttribute(target.attribute).Get(), value)
                for target, value in values
            ):
                continue
            groups.setdefault((source, values), []).append(path)
        for (source, values), paths in groups.items():
            if source and set(paths) == consumers.get(source):
                material_path = source
            else:
                material_path = str(self.material_for_override(self.writable_prim(paths[0])).GetPath())
                for path in paths[1:]:
                    binding = UsdShade.MaterialBindingAPI.Apply(self.writable_prim(path))
                    binding.Bind(
                        UsdShade.Material(self.stage.GetPrimAtPath(material_path)),
                        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
                        materialPurpose="physics",
                    )
                    if str(binding.ComputeBoundMaterial("physics")[0].GetPath()) != material_path:
                        raise NotImplementedError(f"An ancestor material binding prevents contact overrides at {path}.")
            for target, value in values:
                # Native contact readers identify physics materials through this API.
                material_prim = self.stage.GetPrimAtPath(material_path)
                if not material_prim.HasAPI(UsdPhysics.MaterialAPI):
                    UsdPhysics.MaterialAPI.Apply(self.writable_prim(material_path))
                self.write_attribute(material_path, target, value)

    def material_for_override(self, prim: Usd.Prim) -> UsdShade.Material:
        """Return an independently bound physics material, preserving its existing resources."""
        prim = self.writable_prim(prim.GetPath())
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        original, _ = binding.ComputeBoundMaterial("physics")
        destination = prim.GetPath().AppendChild("ExportPhysicsMaterial")
        existing = self.stage.GetPrimAtPath(destination)
        if existing and existing.GetCustomDataByKey("isaaclab:exportMaterial"):
            return UsdShade.Material(existing)
        if self.stage.GetPrimAtPath(destination):
            raise RuntimeError(f"Export material path already exists: {destination}")
        if original:
            layer = self.stage.GetRootLayer()
            if not Sdf.CopySpec(layer, original.GetPath(), layer, destination):
                raise RuntimeError(f"Cannot preserve bound material {original.GetPath()}.")
            material = UsdShade.Material(self.stage.GetPrimAtPath(destination))
            self._rebase_connections(material.GetPrim(), original.GetPath(), destination)
        else:
            material = UsdShade.Material.Define(self.stage, destination)
        binding.Bind(material, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics")
        effective, _ = binding.ComputeBoundMaterial("physics")
        if effective.GetPath() != destination:
            raise NotImplementedError(f"An ancestor material binding prevents contact overrides at {prim.GetPath()}.")
        material.GetPrim().SetCustomDataByKey("isaaclab:exportMaterial", True)
        return material

    def remove_ineffective_material_bindings(self) -> None:
        """Clear missing direct material targets only when resolved materials stay unchanged."""
        purposes = set(UsdShade.MaterialBindingAPI.GetMaterialPurposes()) | {"physics"}
        candidates = []
        for prim in self.stage.Traverse():
            for relationship in prim.GetRelationships():
                tokens = relationship.GetName().split(":")
                if tokens[:2] != ["material", "binding"]:
                    continue
                if len(tokens) == 5 and tokens[2] == "collection":
                    purposes.add(tokens[3])
                elif len(tokens) in (2, 3):
                    purposes.add(tokens[2] if len(tokens) == 3 else "")
                    targets = relationship.GetTargets()
                    if len(targets) == 1 and targets[0].IsPrimPath() and not self.stage.GetPrimAtPath(targets[0]):
                        candidates.append(relationship)

        for relationship in candidates:
            bindings = [UsdShade.MaterialBindingAPI(prim) for prim in Usd.PrimRange(relationship.GetPrim())]

            def resolved_materials():
                return [
                    binding.ComputeBoundMaterial(purpose)[0].GetPath() for binding in bindings for purpose in purposes
                ]

            before = resolved_materials()
            targets = relationship.GetTargets()
            relationship.SetTargets([])
            # A missing direct target can still mask an inherited material.
            if resolved_materials() != before:
                relationship.SetTargets(targets)

    ##
    # Completeness and saving.
    ##

    def validate(self) -> None:
        """Reject physical bodies retained from USD without a corresponding initialized backend data."""
        authored = {
            str(prim.GetPath())
            for prim in self.stage.Traverse(Usd.TraverseInstanceProxies())
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
        for prim in self.stage.Traverse(Usd.TraverseInstanceProxies()):
            for prop in prim.GetProperties():
                targets = prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.GetConnections()
                for path in targets:
                    target = (
                        self.stage.GetPropertyAtPath(path) if path.IsPropertyPath() else self.stage.GetPrimAtPath(path)
                    )
                    if not target:
                        if (str(prop.GetPath()), path) in self._source_missing_bindings:
                            continue
                        raise RuntimeError(f"Unresolved export dependency {prop.GetPath()}: {path}")
        _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(self.stage.GetRootLayer().identifier))
        unresolved = [p for p in unresolved if p not in {"OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl"}]
        if unresolved:
            from isaaclab.utils.assets import check_file_path

            # USD's filesystem localization can turn a valid https:// URI into https:/.
            # Verify the original authored URI; never repair or accept an unknown dependency.
            authored_uris = {}
            for prim in self.stage.Traverse():
                for attr in prim.GetAttributes():
                    if attr.GetTypeName() not in (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray):
                        continue
                    for time in [Usd.TimeCode.Default(), *attr.GetTimeSamples()]:
                        value = attr.Get(time)
                        paths = [value] if isinstance(value, Sdf.AssetPath) else (value or [])
                        for path in paths:
                            if "://" in path.path:
                                authored_uris[os.path.normpath(path.path)] = path.path
            unresolved = [
                path for path in unresolved if path not in authored_uris or not check_file_path(authored_uris[path])
            ]
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
