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
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils, Vt

from isaaclab.assets.physics_properties import (
    UsdAttribute,
    usd_fields,
)

if TYPE_CHECKING:
    from isaaclab.assets import (
        BaseArticulationData,
        BaseRigidObjectCollectionData,
        BaseRigidObjectData,
    )
    from isaaclab.cloner.clone_plan import ClonePlan


@dataclass(frozen=True)
class AssetPaths:
    """Prim identities paired with public body/DOF row indices in one environment."""

    bodies: list[tuple[str, int]]
    joints: list[tuple[str, int]]
    joint_axes: dict[int, str] = field(default_factory=dict)


class UsdWriter:
    """Write declared fixed properties onto one preserved, isolated SI stage.

    Registered schemas provide exact target names and types, not source fields or
    units. Unregistered extensions need explicit declarations. Data properties use
    [environment, row, ...] layout and are read once per writer; getter discovery
    never evaluates them. Tasks, spawning and training are outside this object.
    """

    def __init__(self, stage: Usd.Stage):
        self.stage = stage
        self.env_index = 0
        self.preserve_source_contacts = False
        self.env_id = 0
        self.clone_plan: ClonePlan | None = None
        self.body_paths: set[str] = set()
        self.deformable_paths: set[str] = set()
        self.represented_collider_paths: set[str] = set()
        self._material_paths: set[str] = set()
        self._joint_shared_values: dict[tuple[str, str], float] = {}
        # Retain owners with their arrays, so ids cannot be reused during an export.
        self._values: dict[int, tuple[object, dict[str, np.ndarray]]] = {}

    @classmethod
    def from_stage(cls, stage: Usd.Stage) -> UsdWriter:
        """Preserve composition in an isolated writable SI stage without changing the source."""
        if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdPhysics.GetStageKilogramsPerUnit(stage) != 1:
            raise NotImplementedError("Fixed export requires SI stage units.")
        snapshot = Usd.Stage.Open(stage.Flatten())
        # Expanding an outer instance can expose nested instances on the next traversal.
        while True:
            instances = [prim for prim in snapshot.Traverse() if prim.IsInstance()]
            if not instances:
                writer = cls(Usd.Stage.Open(snapshot.Flatten()))
                writer._remove_unused_material_bindings()
                return writer
            for prim in instances:
                prim.SetInstanceable(False)

    def _remove_unused_material_bindings(self) -> None:
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
            # An invalid direct binding can mask an inherited material. Keep rejecting
            # that case instead of changing the appearance or physics-material fallback.
            if resolved_materials() != before:
                relationship.SetTargets(targets)

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
                UsdGeom.XformCommonAPI(root).SetTranslate(Gf.Vec3d(*map(float, plan.positions[column])))

        excluded = tuple(Sdf.Path(path) for index, path in enumerate(env_paths) if index != env_id)
        # Relationship dependencies may live in a different prototype environment.
        # Retain resources, but never pull a foreign physical entity into the export.
        copied = {}
        pending = list(self.stage.Traverse())
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

    def write_properties(self, path: str, axis: str | None, data: object, *, row: int) -> None:
        """Write inherited property declarations from the selected asset instance."""
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
                if array.ndim < 2 or not 0 <= self.env_index < array.shape[0]:
                    raise ValueError(f"Missing selected environment or row dimension for {source}.")
                values[source] = array[self.env_index].copy()
            for target in targets:
                value = values[source][row]
                if target.component is not None:
                    value = value[target.component]
                self.write_attribute(path, target, value, axis=axis)

    def write_attribute(self, path: str, target: UsdAttribute, value, *, axis: str | None = None) -> None:
        """Write a scalar, vector or scalar-array target with a declared schema/type."""
        if axis in {"rotX", "rotY", "rotZ", "transX", "transY", "transZ"}:
            if target.attribute in {"physics:lowerLimit", "physics:upperLimit"}:
                bound = "low" if target.attribute == "physics:lowerLimit" else "high"
                target = replace(target, attribute=f"limit:{{axis}}:physics:{bound}", schema="PhysicsLimitAPI:{axis}")
            elif target.attribute in {"physxJoint:armature", "physxJoint:maxJointVelocity"}:
                # Per-axis declarations already carry these; a joint-wide alias would overwrite other axes.
                return
            elif target.attribute in {"newton:limitStiffness", "newton:limitDamping"}:
                self.stage.GetPrimAtPath(path).RemoveProperty(target.attribute)
                target = replace(target, attribute=target.attribute.replace("newton:", "newton:{axis}:"))
            elif target.attribute in {"newton:armature", "newton:friction"}:
                # Newton's D6 importer currently accepts only joint-wide values for these fields.
                key = (path, target.attribute)
                previous = self._joint_shared_values.setdefault(key, float(value))
                if previous != float(value):
                    raise NotImplementedError(f"Newton cannot import distinct per-axis {target.attribute} at {path}.")
        if (target.angular_power or "{axis}" in target.attribute or "{axis}" in (target.schema or "")) and axis not in {
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
        attr = self._attribute(path, target, axis)
        converted = np.asarray(value)
        if axis in {"angular", "rotX", "rotY", "rotZ"} and target.angular_power:
            # Derivative quantities such as stiffness need the inverse angular conversion.
            converted = converted * (180 / math.pi) ** target.angular_power
        value_type = attr.GetTypeName()
        default = value_type.defaultValue
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
        """Write body placement and mass properties without copying transient velocities."""
        poses = data.body_link_pose_w.torch[self.env_index].detach().cpu().numpy().reshape(-1, 7).copy()
        masses = data.body_mass.torch[self.env_index].detach().cpu().numpy().reshape(-1)
        inertias = data.body_inertia.torch[self.env_index].detach().cpu().numpy().reshape(-1, 3, 3)
        coms = data.body_com_pose_b.torch[self.env_index].detach().cpu().numpy().reshape(-1, 7)
        if sorted(row for _, row in paths) != list(range(len(poses))):
            raise RuntimeError("Incomplete body identities for fixed configuration.")
        # Update parents first so child transforms retain their world placement.
        for path, row in sorted(paths, key=lambda item: Sdf.Path(item[0]).pathElementCount):
            prim = self.stage.GetPrimAtPath(path)
            if path in self.body_paths or not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Missing or multiply owned body {path}.")
            self._write_body_pose(prim, poses[row])
            self.write_mass_properties(prim, masses[row], inertias[row], coms[row])
            self.body_paths.add(path)

    def write_mass_properties(self, prim: Usd.Prim, mass: float, inertia: np.ndarray, com: np.ndarray) -> None:
        """Write mass [kg], COM pose [m, xyzw] and link-frame inertia [kg*m²]."""
        if not np.isfinite(mass) or mass < 0 or not np.isfinite(inertia).all() or not np.isfinite(com).all():
            raise ValueError(f"Invalid mass properties at {prim.GetPath()}.")
        if not np.allclose(inertia, inertia.T, atol=1e-7):
            raise ValueError(f"Non-symmetric inertia at {prim.GetPath()}.")
        rotation = Gf.Quatd(float(com[6]), Gf.Vec3d(*map(float, com[3:6])))
        axes = np.asarray(Gf.Matrix3d(rotation)).T
        # USD stores principal moments and axes, while public data supplies a full link-frame tensor.
        principal = axes.T @ inertia @ axes
        if np.allclose(principal, np.diag(np.diag(principal)), atol=1e-7):
            moments = np.diag(principal)
        else:
            moments, axes = np.linalg.eigh(inertia)
            # An eigenbasis may be reflected; quaternions require a right-handed frame.
            if np.linalg.det(axes) < 0:
                axes[:, 0] *= -1
            rotation = Gf.Matrix3d(*map(float, axes.T.flatten())).ExtractRotation().GetQuat()
        if np.any(moments < -1e-7):
            raise ValueError(f"Negative inertia at {prim.GetPath()}.")
        physics = UsdPhysics.MassAPI.Apply(prim)
        physics.CreateMassAttr().Set(float(mass))
        physics.CreateCenterOfMassAttr().Set(Gf.Vec3f(*map(float, com[:3])))
        physics.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, np.maximum(moments, 0))))
        physics.CreatePrincipalAxesAttr().Set(Gf.Quatf(rotation))

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
        deformables = {
            str(prim.GetPath())
            for prim in self.stage.Traverse()
            if any(
                api in {"PhysicsDeformableBodyAPI", "OmniPhysicsDeformableBodyAPI"}
                for api in prim.GetPrimTypeInfo().GetAppliedAPISchemas()
            )
        }
        if deformables != self.deformable_paths:
            raise RuntimeError(f"Incomplete deformable export: {deformables ^ self.deformable_paths}.")
        self.validate_dependencies()

    def write_deformable_points(self, path: str, positions: np.ndarray) -> Usd.Prim:
        """Retain rest geometry and place simulation nodes from world positions [m]."""
        from isaaclab.scene_data.deformable_discovery import _classify_deformable_meshes

        root = self.stage.GetPrimAtPath(path)
        _, mesh, _, count, *_ = _classify_deformable_meshes(root)
        positions = np.asarray(positions)[:count]
        if positions.shape != (count, 3):
            raise ValueError(f"Incomplete simulation nodes for {path}: {positions.shape}.")
        # USD mesh points are local, whereas backend simulation nodes are world-space.
        transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh).GetInverse())
        points = positions @ transform[:3, :3] + transform[3, :3]
        UsdGeom.PointBased(mesh).GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        self.deformable_paths.add(path)
        return mesh

    def write_gravity(self, scene_path: str, gravity) -> None:
        """Author effective gravity [m/s²] from a backend's selected world."""
        gravity = np.asarray(gravity, dtype=float)
        magnitude = float(np.linalg.norm(gravity))
        physics = UsdPhysics.Scene(self.stage.GetPrimAtPath(scene_path))
        physics.CreateGravityMagnitudeAttr().Set(magnitude)
        physics.CreateGravityDirectionAttr().Set(Gf.Vec3f(*(gravity / magnitude if magnitude else (0, 0, -1))))

    def write_body_contacts(self, path: str, disabled: bool, materials, offsets, rest_offsets) -> None:
        """Author body gravity and PhysX contact buffers without guessing shape ordering.

        Pre-startup export preserves source contacts. Otherwise, native tensor views
        expose body identities but not collider paths. A single
        collider or equal per-body values are unambiguous; distinct per-shape values
        require a backend identity API and are rejected.
        """
        body = self.stage.GetPrimAtPath(path)
        self.write_attribute(
            path, UsdAttribute("physxRigidBody:disableGravity", "PhysxRigidBodyAPI", type_name="bool"), disabled
        )
        if self.preserve_source_contacts:
            # Before buffer randomization, authored bindings and native defaults remain authoritative.
            return
        colliders = []
        descendants = iter(Usd.PrimRange(body))
        for prim in descendants:
            if prim != body and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                descendants.PruneChildren()
            elif prim.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(prim)
        materials = np.asarray(materials).reshape(-1, 3)
        offsets, rest_offsets = np.asarray(offsets).reshape(-1), np.asarray(rest_offsets).reshape(-1)
        count = len(colliders)
        if count == 0:
            return
        # A cooked mesh can yield several native convex shapes. Uniform values
        # remain representable on its original geometry without reconstructing it.
        if not len(materials) or any(len(values) != len(materials) for values in (offsets, rest_offsets)):
            raise NotImplementedError(f"Unmatched collider identities/count at {path}.")
        if any(not np.all(values == values[0]) for values in (materials, offsets, rest_offsets)):
            raise NotImplementedError(f"Distinct per-shape contact values lack stable collider identities at {path}.")
        for prim in colliders:
            for name, value in (("contactOffset", offsets[0]), ("restOffset", rest_offsets[0])):
                self.write_attribute(
                    str(prim.GetPath()),
                    UsdAttribute(f"physxCollision:{name}", "PhysxCollisionAPI", type_name="float"),
                    float(value),
                )
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            original, _ = binding.ComputeBoundMaterial("physics")
            names = ("staticFriction", "dynamicFriction", "restitution")
            if original and all(
                original.GetPrim().GetAttribute(f"physics:{name}").Get() == float(value)
                for name, value in zip(names, materials[0])
            ):
                continue
            material = self.material_for_override(prim)
            destination = material.GetPath()
            for name, value in zip(names, materials[0]):
                self.write_attribute(
                    str(destination), UsdAttribute(f"physics:{name}", "PhysicsMaterialAPI"), float(value)
                )

    def material_for_override(self, prim: Usd.Prim) -> UsdShade.Material:
        """Return an independently bound physics material, preserving its existing resources."""
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        original, _ = binding.ComputeBoundMaterial("physics")
        destination = prim.GetPath().AppendChild("ExportPhysicsMaterial")
        if str(destination) in self._material_paths:
            return UsdShade.Material(self.stage.GetPrimAtPath(destination))
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
        self._material_paths.add(str(destination))
        return material

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
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp(opSuffix="export").Set(local)
