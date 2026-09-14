# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export complete authored scenes and their initialized physical configuration.

:class:`SceneExporter` owns environment selection, asset traversal, dependency and coverage
checks, and atomic saving. Its fixed-config entry point uses normal scene construction,
then exports the configured default state before stepping or task randomization. Authored
USD carries geometry, materials, filtering and schema properties; public asset data supplies
resolved actuator solver values and initial state. Backend adapters supply provenance and
properties whose semantics are absent from the common data contract.

The legacy runtime entry points also capture supported buffer overrides. They retain their
explicit support boundaries; neither export mode serializes a running policy or solver cache.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

from isaaclab.assets.physics_properties import (
    BodyPhysicsProperties,
    read_joint_properties,
    validate_configuration_coverage,
)
from isaaclab.sim.utils import safe_set_attribute_on_usd_prim

if TYPE_CHECKING:
    from isaaclab.assets import BaseArticulation, BaseRigidObject, BaseRigidObjectCollection
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg

# Drive token per joint type. UsdPhysics names the drive after the motion it actuates, so a
# prismatic joint's gains live under "linear" and a revolute joint's under "angular"; reading the
# wrong one returns an unauthored drive rather than an error.
_DRIVE_TOKEN = {"PhysicsPrismaticJoint": "linear", "PhysicsRevoluteJoint": "angular"}

# Revolute joints are radians in the simulation and degrees on the stage: limits scale by 180/pi;
# drive gains and the viscous friction coefficient, being per unit angle or angular rate, by pi/180.
# Prismatic joints are metres on both sides.
_DEGREE_LIMIT_JOINT = "PhysicsRevoluteJoint"
_PER_DEGREE = math.pi / 180.0

# Armature and joint friction have no UsdPhysics home. They are authored under the PhysX add-on
# namespace directly rather than through ``PhysxSchema``, which the kitless runtimes do not ship.
# Friction is the per-axis static/dynamic/viscous model the runtimes simulate; the legacy scalar
# ``physxJoint:jointFriction`` is not read back into it, so it is not written. Once the per-axis
# schema is applied it shadows the joint-level armature, so armature is authored in both places.
_PHYSX_JOINT_SCHEMA = "PhysxJointAPI"
_PHYSX_JOINT_AXIS_SCHEMA = "PhysxJointAxisAPI"
_AXIS_ATTRIBUTES = ("armature", "staticFrictionEffort", "dynamicFrictionEffort", "viscousFrictionCoefficient")


@dataclass(frozen=True)
class ArticulationPrimPaths:
    """One environment's body and joint prim paths, in the backend's own index order.

    Backends record provenance differently -- PhysX reads it off the tensor view, others resolve it
    from the stage -- so each supplies this rather than the exporter guessing.

    Attributes:
        bodies: Prim path of each body, indexed in backend order.
        joints: Prim path of each joint, indexed in backend order.
    """

    bodies: list[str]
    joints: list[str]


PrimPathResolver = Callable[["BaseArticulation", int], ArticulationPrimPaths]
"""Backend hook returning one environment's prim paths for an initialized articulation."""


class ArticulationExporter:
    """Writes a running articulation's simulated state back to USD.

    The values are read through the backend-independent :class:`~isaaclab.assets.BaseArticulationData`
    and written as standard ``UsdPhysics`` plus PhysX add-on attributes. The one backend-specific step,
    recovering the prim each body and joint came from, is the ``resolver`` the backend passes in.

    Args:
        articulation: The articulation to export. It must be initialized, since the values come from
            the running simulation rather than from its configuration.
        resolver: Backend hook mapping ``(articulation, env_index)`` to :class:`ArticulationPrimPaths`.
    """

    def __init__(self, articulation: BaseArticulation, resolver: PrimPathResolver) -> None:
        self.articulation = articulation
        self._resolver = resolver

    def prim_paths(self, env_index: int = 0) -> ArticulationPrimPaths:
        """Prim paths of one environment's bodies and joints, in backend index order."""
        return self._resolver(self.articulation, env_index)

    def write_to_stage(
        self, env_index: int = 0, *, stage: Usd.Stage | None = None, fixed_configuration: bool = False
    ) -> list[str]:
        """Author the simulated state onto the prims the articulation was spawned from.

        Body masses and joint drive gains, armature, friction and limits are read from the simulation
        and written onto the stage, replacing the spawn-time values it still carries. Schemas are
        applied only where absent; existing attributes are overwritten in place.

        Args:
            env_index: Environment whose state to author. Defaults to ``0``.
            fixed_configuration: Preserve authored mass/inertia and physical schemas while
                writing initialized body/joint state and resolved solver drive properties.
            stage: Stage to author onto; it must hold the same prim paths as the live stage. Defaults
                to the live stage itself, which the simulation keeps reading: on PhysX, applying a
                schema to a prim that is an articulation root invalidates every articulation view on
                the stage for the rest of the session. Pass a flattened copy, as :meth:`export` does,
                to leave the running simulation untouched.

        Returns:
            The prim paths written, bodies first.

        Raises:
            ValueError: If ``env_index`` is not an environment of the articulation.
            RuntimeError: If a resolved path is not a prim on the stage, or a backend-order name is
                absent from the public order. Both are contract violations -- the view is built from
                the stage and the two orders are permutations of one another -- so the stage no longer
                describes what the backend is simulating and a partial export would hide that.
        """
        articulation = self.articulation
        if not 0 <= env_index < articulation.num_instances:
            raise ValueError(
                f"Environment {env_index} is out of range for an articulation with"
                f" {articulation.num_instances} instance(s)."
            )
        prim_paths = self.prim_paths(env_index)
        data = articulation.data
        stage = articulation.stage if stage is None else stage

        # Backend order indexes the paths; public order indexes the data. Join by name -- a reordered
        # articulation would otherwise take every value from the wrong row.
        body_row = {name: index for index, name in enumerate(articulation.body_names)}
        joint_row = {name: index for index, name in enumerate(articulation.joint_names)}

        # One host transfer per array; indexing a device tensor per element would sync on every read.
        bodies = BodyPhysicsProperties.from_data(data, env_index)
        properties = {name: value[env_index].tolist() for name, value in read_joint_properties(data).items()}
        expected = {
            "stiffness",
            "damping",
            "armature",
            "friction",
            "dynamic_friction",
            "viscous_friction",
            "joint_effort_limit",
            "joint_velocity_limit",
        }
        if set(properties) != expected:
            raise NotImplementedError(f"Missing USD joint-property semantics: {set(properties) ^ expected}")
        limits = data.joint_pos_limits.torch[env_index].tolist()
        positions = data.joint_pos.torch[env_index].tolist()
        joint_velocities = data.joint_vel.torch[env_index].tolist()

        def resolve(path: str, name: str, rows: dict[str, int], kind: str) -> tuple[Usd.Prim, int]:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"{kind} '{name}' was resolved to '{path}', which is not a prim on the stage.")
            row = rows.get(name)
            if row is None:
                raise RuntimeError(f"{kind} '{name}' is in the backend order but absent from the public order.")
            return prim, row

        # Read everything before writing anything. Applying a schema to a prim on a live PhysX stage
        # invalidates the tensor view the articulation reads its names through, so a read interleaved
        # with the writes fails on assets that did not already carry the schema.
        body_names = list(articulation.backend_body_names)
        joint_names = list(articulation.backend_joint_names)
        if len(prim_paths.bodies) != len(body_names) or len(prim_paths.joints) != len(joint_names):
            raise RuntimeError("Articulation provenance does not cover every backend body and DOF.")
        body_targets = [resolve(path, body_names[i], body_row, "Body") for i, path in enumerate(prim_paths.bodies)]
        joint_targets = [resolve(path, joint_names[i], joint_row, "Joint") for i, path in enumerate(prim_paths.joints)]
        for prim, _ in joint_targets:
            if prim.GetTypeName() not in _DRIVE_TOKEN:
                raise NotImplementedError(
                    f"No USD axis mapping for driven joint {prim.GetPath()} ({prim.GetTypeName()})."
                )

        written: list[str] = []
        written.extend(
            write_body_properties(
                stage,
                prim_paths.bodies,
                bodies,
                [row for _, row in body_targets],
                preserve_authored_mass=fixed_configuration,
            )
        )
        for (prim, row), path in zip(joint_targets, prim_paths.joints):
            self._author_joint(
                prim,
                stiffness=properties["stiffness"][row],
                damping=properties["damping"][row],
                armature=properties["armature"][row],
                friction=tuple(properties[name][row] for name in ("friction", "dynamic_friction", "viscous_friction")),
                lower_limit=limits[row][0],
                upper_limit=limits[row][1],
            )
            token = _DRIVE_TOKEN.get(prim.GetTypeName())
            if token:
                UsdPhysics.DriveAPI(prim, token).CreateMaxForceAttr().Set(float(properties["joint_effort_limit"][row]))
                velocity = float(properties["joint_velocity_limit"][row])
                if token == "angular":
                    velocity = math.degrees(velocity)
                safe_set_attribute_on_usd_prim(prim, "physxJoint:maxJointVelocity", velocity, camel_case=False)
                safe_set_attribute_on_usd_prim(
                    prim, f"physxJointAxis:{token}:maxJointVelocity", velocity, camel_case=False
                )
                prim.AddAppliedSchema(f"PhysicsJointStateAPI:{token}")
                scale = 180.0 / math.pi if token == "angular" else 1.0
                for name, value in (("position", positions[row]), ("velocity", joint_velocities[row])):
                    safe_set_attribute_on_usd_prim(
                        prim, f"state:{token}:physics:{name}", value * scale, camel_case=False
                    )
            written.append(path)
        return written

    def export(self, usd_path: str, env_index: int = 0) -> str:
        """Export this articulation and its required USD dependencies.

        Other scene assets are excluded. For a complete environment, use :class:`SceneExporter`
        with the scene registry.

        The live stage is flattened first and the state is authored onto that snapshot, so the running
        simulation never sees the edits. The live stage's own file is not saved.

        Args:
            usd_path: Destination path for the USD file.
            env_index: Environment to export. Defaults to ``0``.

        Returns:
            The path the stage was written to.
        """
        snapshot = Usd.Stage.Open(self.articulation.stage.Flatten())
        paths = self.prim_paths(env_index)
        root = Sdf.Path(paths.bodies[0])
        for path in paths.bodies + paths.joints:
            root = root.GetCommonPrefix(Sdf.Path(path))
        # Include the asset scope containing controllers and articulation-root schemas.
        from isaaclab.sim.utils import find_matching_prim_paths

        asset_roots = [
            Sdf.Path(path)
            for path in find_matching_prim_paths(self.articulation.cfg.prim_path, stage=snapshot)
            if root.HasPrefix(Sdf.Path(path))
        ]
        if len(asset_roots) == 1:
            root = asset_roots[0]
        if root == Sdf.Path.absoluteRootPath or root == Sdf.Path("/World"):
            raise RuntimeError("Articulation prims have no isolated asset root; export the registered scene instead.")
        retain_stage_objects(snapshot, [root])
        self.write_to_stage(env_index, stage=snapshot)
        _validate_dependencies(snapshot)
        return save_environment_snapshot(snapshot, usd_path)

    @staticmethod
    def _author_joint(
        prim: Usd.Prim,
        *,
        stiffness: float,
        damping: float,
        armature: float,
        friction: tuple[float, float, float],
        lower_limit: float,
        upper_limit: float,
    ) -> None:
        """Write one joint's simulated properties onto its prim.

        ``friction`` is the (static effort, dynamic effort, viscous coefficient) triple of the drive
        axis.
        """
        token = _DRIVE_TOKEN.get(prim.GetTypeName())
        if token is not None:
            gain_scale = _PER_DEGREE if prim.GetTypeName() == _DEGREE_LIMIT_JOINT else 1.0
            drive = (
                UsdPhysics.DriveAPI(prim, token)
                if prim.HasAPI(UsdPhysics.DriveAPI, token)
                else UsdPhysics.DriveAPI.Apply(prim, token)
            )
            drive.CreateStiffnessAttr().Set(float(stiffness) * gain_scale)
            drive.CreateDampingAttr().Set(float(damping) * gain_scale)

            axis_schema = f"{_PHYSX_JOINT_AXIS_SCHEMA}:{token}"
            if axis_schema not in prim.GetAppliedSchemas():
                prim.AddAppliedSchema(axis_schema)
            static_friction, dynamic_friction, viscous_friction = friction
            axis_values = (armature, static_friction, dynamic_friction, viscous_friction * gain_scale)
            for name, value in zip(_AXIS_ATTRIBUTES, axis_values):
                safe_set_attribute_on_usd_prim(prim, f"physxJointAxis:{token}:{name}", float(value), camel_case=False)

        if _PHYSX_JOINT_SCHEMA not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema(_PHYSX_JOINT_SCHEMA)
        safe_set_attribute_on_usd_prim(prim, "physxJoint:armature", float(armature), camel_case=False)

        if prim.GetTypeName() == _DEGREE_LIMIT_JOINT:
            lower_limit, upper_limit = math.degrees(lower_limit), math.degrees(upper_limit)
        for name, value in (("physics:lowerLimit", lower_limit), ("physics:upperLimit", upper_limit)):
            attribute = prim.GetAttribute(name)
            if attribute:
                attribute.Set(float(value))


def write_articulation_state_to_stage(
    articulation: BaseArticulation,
    prim_paths: ArticulationPrimPaths,
    env_index: int = 0,
    *,
    stage: Usd.Stage | None = None,
) -> list[str]:
    """Author an articulation's simulated state onto pre-resolved prims.

    Functional form of :meth:`ArticulationExporter.write_to_stage` for callers that already hold the
    environment's :class:`ArticulationPrimPaths`.
    """
    return ArticulationExporter(articulation, lambda _articulation, _env: prim_paths).write_to_stage(
        env_index, stage=stage
    )


def export_articulation_to_usd(
    articulation: BaseArticulation, prim_paths: ArticulationPrimPaths, usd_path: str, env_index: int = 0
) -> str:
    """Export one environment's articulation, as simulated, to a USD file from pre-resolved prims.

    Functional form of :meth:`ArticulationExporter.export`.
    """
    return ArticulationExporter(articulation, lambda _articulation, _env: prim_paths).export(usd_path, env_index)


def author_inertia(mass_api: UsdPhysics.MassAPI, inertia: np.ndarray) -> None:
    """Author a body-frame inertia tensor [kg*m^2], shape [3, 3], including principal axes."""
    tensor = np.asarray(inertia, dtype=np.float64).reshape(3, 3)
    if not np.allclose(tensor, tensor.T, atol=1e-7, rtol=1e-5):
        raise ValueError(f"Non-symmetric inertia at {mass_api.GetPath()}: {tensor}")
    if np.allclose(tensor, np.diag(np.diag(tensor)), atol=1e-12, rtol=0):
        moments, axes = np.diag(tensor), np.eye(3)
    else:
        moments, axes = np.linalg.eigh(tensor)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
    rotation = Gf.Matrix3d(*axes.T.flatten()).ExtractRotation().GetQuat()
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, moments)))
    # Always overwrite: a diagonal runtime tensor may replace a rotated spawn-time tensor.
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(rotation))


def author_mass_properties(prim: Usd.Prim, mass: float, inertia: np.ndarray, com_pose: np.ndarray) -> None:
    """Write mass [kg], body-frame inertia about COM [kg*m^2] and body-local COM pose [m, xyzw]."""
    length = UsdGeom.GetStageMetersPerUnit(prim.GetStage())
    mass_unit = UsdPhysics.GetStageKilogramsPerUnit(prim.GetStage())
    com_pose = np.asarray(com_pose)
    tensor = np.asarray(inertia).reshape(3, 3)
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr().Set(float(mass) / mass_unit)
    api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*map(float, com_pose[:3] / length)))
    author_inertia(api, tensor / (mass_unit * length**2))


def author_gravity(stage: Usd.Stage, scene_path: str, gravity: np.ndarray) -> None:
    """Write effective world gravity [m/s^2], including the zero-gravity case."""
    gravity = np.asarray(gravity, dtype=np.float64)
    magnitude = float(np.linalg.norm(gravity))
    api = UsdPhysics.Scene.Define(stage, scene_path)
    direction = gravity / magnitude if magnitude else np.array([0.0, 0.0, -1.0])
    api.CreateGravityDirectionAttr().Set(Gf.Vec3f(*map(float, direction)))
    api.CreateGravityMagnitudeAttr().Set(magnitude / UsdGeom.GetStageMetersPerUnit(stage))


def author_physics_material(prim: Usd.Prim, values: np.ndarray) -> None:
    """Write static/dynamic friction and restitution to a private physics material.

    Copy the bound material first to preserve combine modes and other authored properties.
    A shared spawn material must not make one object's runtime override affect another object.
    """
    stage = prim.GetStage()
    path = prim.GetPath().AppendChild("ExportPhysicsMaterial")
    bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    if bound and bound.GetPath() != path:
        Sdf.CopySpec(stage.GetRootLayer(), bound.GetPath(), stage.GetRootLayer(), path)
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    for create, value in zip(
        (api.CreateStaticFrictionAttr, api.CreateDynamicFrictionAttr, api.CreateRestitutionAttr), values
    ):
        create().Set(float(value))
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics"
    )


def copy_scene_stage(stage: Usd.Stage) -> Usd.Stage:
    """Flatten composition and make instances writable without touching the source stage."""
    snapshot = Usd.Stage.Open(stage.Flatten())
    while True:
        instances = [prim for prim in snapshot.Traverse() if prim.IsInstance()]
        if not instances:
            return Usd.Stage.Open(snapshot.Flatten())
        for prim in instances:
            prim.SetInstanceable(False)


def create_environment_snapshot(scene: InteractiveScene, env_index: int = 0) -> Usd.Stage:
    """Copy one environment's authored content and shared resources without editing the live stage.

    ClonePlan selects source variants only. Runtime physics values must subsequently be written by
    the backend adapter. The selected environment keeps its original paths and world placement.

    Args:
        scene: Initialized scene with an active ClonePlan and registered objects.
        env_index: Environment id to export (not a ClonePlan mask column).

    Returns:
        Independent, flattened stage containing the selected environment and shared scene content.

    Raises:
        ValueError: If the plan does not contain the requested environment.
        RuntimeError: If a required source or a non-filter dependency is missing.
        NotImplementedError: If the scene contains unsupported physical object families.
    """
    from isaaclab.cloner import query

    plan = scene.clone_plan
    for family in ("deformable_objects", "cable_objects", "surface_grippers"):
        if getattr(scene, family):
            raise NotImplementedError(f"Environment export does not support {family}: {list(getattr(scene, family))}")
    if plan is None:
        return _snapshot_single_environment(scene, env_index)
    env_ids = list(range(plan.clone_mask.shape[1])) if plan.env_ids is None else plan.env_ids.tolist()
    if env_index not in env_ids:
        raise ValueError(f"Environment {env_index} is out of range for ClonePlan environments {env_ids}.")
    snapshot = copy_scene_stage(scene.sim.stage)
    layer = snapshot.GetRootLayer()
    selected = []
    for template in dict.fromkeys(plan.destinations):
        target = template.format(env_index)
        source = query.path_to_source(plan, target, env_index)
        if source is None:
            continue
        source_path = source[0] + source[2]
        selected.append(Sdf.Path(target))
        if source_path == target:
            continue
        if not snapshot.GetPrimAtPath(source_path):
            raise RuntimeError(f"Missing export source {source_path} for {target}.")
        clone_layer = Sdf.Layer.CreateAnonymous()
        Sdf.CreatePrimInLayer(clone_layer, Sdf.Path(target).GetParentPath())
        Sdf.CopySpec(layer, source_path, clone_layer, target)
        clone_stage = Usd.Stage.Open(clone_layer)
        for prim in Usd.PrimRange(clone_stage.GetPrimAtPath(target)):
            for prop in prim.GetProperties():
                paths = prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.GetConnections()
                mapped = [p.ReplacePrefix(Sdf.Path(source_path), Sdf.Path(target)) for p in paths]
                if mapped != paths:
                    if isinstance(prop, Usd.Relationship):
                        prop.SetTargets(mapped)
                    else:
                        prop.SetConnections(mapped)
        # Runtime-authored USD overrides in the destination remain stronger than its prototype.
        UsdUtils.StitchLayers(layer, clone_layer)
    # The registry contributes shared assets and validates the selected variants, including extras.
    for name in scene.keys():
        obj = scene[name]
        if obj is None or name in scene.sensors or name in scene.visual_materials:
            continue
        cfg = getattr(obj, "cfg", obj)
        configs = getattr(cfg, "rigid_objects", None)
        configs = configs.values() if configs is not None else (cfg,)
        for cfg in configs:
            expr = cfg.prim_path
            source = query.path_to_source(plan, expr, env_index)
            if source is not None:
                path = query.path_to_clone(plan, source[0] + source[2], env_index)
                if path and not snapshot.GetPrimAtPath(path):
                    raise RuntimeError(f"Registered object {name!r} is missing at {path}.")
            elif not any(query.iter_sources(plan, expr)) and not snapshot.GetPrimAtPath(expr):
                raise RuntimeError(f"Shared registered object {name!r} is missing at {expr}.")
    removed = []
    for template in dict.fromkeys(plan.destinations):
        for other in env_ids:
            path = Sdf.Path(template.format(other))
            if path not in selected:
                removed.append(path)
    removed.extend(Sdf.Path(p) for p in plan.sources if Sdf.Path(p) not in selected)
    # Remove whole other environment roots too, including scene decoration outside the plan rows.
    for other, path in zip(env_ids, scene.env_prim_paths):
        if other != env_index:
            removed.append(Sdf.Path(path))
    removed_roots = []
    for path in sorted(set(removed), key=lambda p: len(str(p)), reverse=True):
        if not any(p.HasPrefix(path) for p in selected):
            snapshot.RemovePrim(path)
            removed_roots.append(path)
    _validate_dependencies(snapshot, removed_roots)
    snapshot.GetRootLayer().customLayerData = {
        **snapshot.GetRootLayer().customLayerData,
        "isaaclab:environment": env_index,
        "isaaclab:snapshot": "initialized physical configuration; controllers and sensor runtime excluded",
    }
    return snapshot


def _snapshot_single_environment(scene: InteractiveScene, env_index: int) -> Usd.Stage:
    """Validate a normally constructed scene that did not require replication."""
    if scene.num_envs != 1 or env_index != 0:
        raise ValueError("Replicated environment export requires the scene's ClonePlan.")
    snapshot = copy_scene_stage(scene.sim.stage)
    from isaaclab.sim.utils import find_matching_prim_paths

    for name in scene.keys():
        obj = scene[name]
        if obj is None or name in scene.sensors or name in scene.visual_materials:
            continue
        cfg = getattr(obj, "cfg", obj)
        configs = getattr(cfg, "rigid_objects", None)
        for cfg in configs.values() if configs is not None else (cfg,):
            if not find_matching_prim_paths(cfg.prim_path, stage=snapshot):
                raise RuntimeError(f"Registered object {name!r} is missing at {cfg.prim_path}.")
    _validate_dependencies(snapshot)
    return snapshot


def retain_stage_objects(stage: Usd.Stage, roots: list[Sdf.Path]) -> None:
    """Retain isolated asset roots, physics scenes and their transitive USD dependencies."""
    retained = set(roots)
    retained.update(prim.GetPath() for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene))
    while True:
        before = set(retained)
        for prim in stage.Traverse():
            if not any(prim.GetPath().HasPrefix(root) for root in retained):
                continue
            for prop in prim.GetProperties():
                if prop.GetName() == "physics:filteredPairs" or prop.GetName().startswith("collection:"):
                    continue
                targets = prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.GetConnections()
                retained.update(target.GetPrimPath() for target in targets)
        if before == retained:
            break
    removed_roots = []
    for prim in reversed(list(stage.Traverse())):
        path = prim.GetPath()
        if not any(path.HasPrefix(root) or root.HasPrefix(path) for root in retained):
            stage.RemovePrim(path)
            removed_roots.append(path)
    _validate_dependencies(stage, removed_roots)


def _validate_dependencies(stage: Usd.Stage, removed_roots: Sequence[Sdf.Path] = ()) -> None:
    """Prune deleted-replica filter memberships and reject broken physical or visual dependencies."""
    for prim in stage.Traverse():
        for prop in prim.GetProperties():
            paths = prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.GetConnections()
            missing = [
                p for p in paths if not (stage.GetPropertyAtPath(p) if p.IsPropertyPath() else stage.GetPrimAtPath(p))
            ]
            if not missing:
                continue
            # Filter/collection membership in deleted replicas is intentionally discarded. Physical
            # joints and shader connections must never be silently severed.
            filtering = prop.GetName() == "physics:filteredPairs" or prop.GetName().startswith("collection:")
            if (
                filtering
                and isinstance(prop, Usd.Relationship)
                and all(any(p.HasPrefix(root) for root in removed_roots) for p in missing)
            ):
                prop.SetTargets([p for p in paths if p not in missing])
            else:
                raise RuntimeError(f"Unresolved export dependency {prop.GetPath()}: {missing}")


def save_environment_snapshot(stage: Usd.Stage, usd_path: str) -> str:
    """Save a complete snapshot atomically, leaving an existing destination intact on failure."""
    _validate_dependencies(stage)
    _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(stage.GetRootLayer().identifier))
    # Isaac Sim supplies these MDL modules through its renderer's search path. Kitless
    # USD cannot resolve them as files; their authored shader references remain intact.
    unresolved = [path for path in unresolved if path not in {"OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl"}]
    if unresolved:
        raise RuntimeError(f"Unresolved export asset dependencies: {unresolved}")
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


def write_rigid_object_state_to_stage(
    asset: BaseRigidObject, body_paths: list[str], row: int, stage: Usd.Stage
) -> list[str]:
    """Write all registered rigid-object bodies through their public data interface."""
    return write_body_properties(stage, body_paths, BodyPhysicsProperties.from_data(asset.data, row))


def write_body_properties(
    stage: Usd.Stage,
    paths: list[str],
    properties: BodyPhysicsProperties,
    rows: list[int] | None = None,
    *,
    preserve_authored_mass: bool = False,
) -> list[str]:
    """Author the common rigid-body contract for links, rigid objects and collection members."""
    rows = list(range(len(properties.mass))) if rows is None else rows
    if len(paths) != len(rows) or len(set(paths)) != len(paths):
        raise RuntimeError(f"Body coverage mismatch: {paths}")
    for path, row in sorted(zip(paths, rows), key=lambda item: Sdf.Path(item[0]).pathElementCount):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Missing rigid body {path}.")
        if not preserve_authored_mass:
            author_mass_properties(prim, properties.mass[row], properties.inertia[row], properties.com_pose[row])
        author_body_state(prim, properties.pose[row], properties.velocity[row])
    return paths


def environment_asset_rows(scene: InteractiveScene, path_expr: str, roots: list[str], env_index: int) -> list[int]:
    """Select backend rows by stable prim identity, including shared and partially populated assets."""
    from isaaclab.cloner import query

    plan = scene.clone_plan
    if plan is None:
        from isaaclab.sim.utils import find_matching_prim_paths

        targets = [Sdf.Path(path) for path in find_matching_prim_paths(path_expr, stage=scene.sim.stage)]
        rows = [i for i, root in enumerate(roots) if any(Sdf.Path(root).HasPrefix(target) for target in targets)]
        if not rows:
            raise RuntimeError(f"No backend instance for registered object {path_expr}.")
        return rows
    source = query.path_to_source(plan, path_expr, env_index)
    if source is None:
        if any(query.iter_sources(plan, path_expr)):
            return []
        target = Sdf.Path(path_expr)
    else:
        target = Sdf.Path(query.path_to_clone(plan, source[0] + source[2], env_index))
    rows = [i for i, root in enumerate(roots) if Sdf.Path(root).HasPrefix(target)]
    if not rows:
        raise RuntimeError(f"No backend instance for registered object {path_expr} at {target}.")
    return rows


def export_environment_to_usd(scene: InteractiveScene, usd_path: str, env_index: int = 0) -> str:
    """Export one initialized environment and its effective physical configuration.

    Call after initialization, reset/startup events and runtime parameter writes, with stepping
    paused. This exports configuration, not a resumable solver checkpoint. Source geometry,
    materials, lights and shared resources are retained; backend buffers override authored physical
    values. Controllers, actuator histories, observation code, sensor buffers and external forces
    are not serialized. Same-backend validation is required before deployment; cross-backend
    physical semantics need separate validation.

    Args:
        scene: Scene whose registered objects and ClonePlan define the export boundary.
        usd_path: Destination USD file.
        env_index: Selected environment id.

    Returns:
        The destination path after the complete export succeeds.
    """
    return SceneExporter(scene).export(usd_path, env_index)


def write_collision_properties(
    stage: Usd.Stage,
    body_paths: list[str],
    materials: np.ndarray,
    contact_offsets: np.ndarray,
    rest_offsets: np.ndarray,
) -> None:
    """Write collider buffers when their association with USD prims is unambiguous.

    Tensor views do not expose collider paths. Multiple distinct rows cannot safely be associated
    by USD traversal order. Reject that case rather than attaching values to the wrong geometry.
    """
    colliders = []
    for path in body_paths:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                colliders.append(prim)
    arrays = [
        np.asarray(a).reshape(-1, width) for a, width in ((materials, 3), (contact_offsets, 1), (rest_offsets, 1))
    ]
    if not colliders:
        return
    count = len(colliders)
    for array in arrays:
        if len(array) < count:
            raise RuntimeError(f"Collider coverage mismatch for {body_paths}: USD={count}, backend={len(array)}")
        if not np.all(array[:count] == array[0]):
            raise NotImplementedError(
                f"Distinct per-shape values for {body_paths} need backend collider-path provenance;"
                " tensor rows cannot be joined by USD traversal order."
            )
    length = UsdGeom.GetStageMetersPerUnit(stage)
    for prim in colliders:
        author_physics_material(prim, arrays[0][0])
        prim.AddAppliedSchema("PhysxCollisionAPI")
        for name, value in (("contactOffset", arrays[1][0, 0]), ("restOffset", arrays[2][0, 0])):
            safe_set_attribute_on_usd_prim(prim, f"physxCollision:{name}", float(value) / length, camel_case=False)


def check_articulation_export(articulation: BaseArticulation) -> None:
    """Reject runtime structures without an effective-value writer before saving a file."""
    if articulation.num_fixed_tendons or articulation.num_spatial_tendons:
        raise NotImplementedError("Environment export of runtime tendon properties is not supported.")


def author_world_transform(prim: Usd.Prim, transform: Gf.Matrix4d) -> None:
    """Author a world transform in stage units while preserving the existing parent hierarchy."""
    parent = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
    local = transform * parent.GetInverse()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp(opSuffix="export").Set(local)


@dataclass(frozen=True)
class RigidBodyExportProperties:
    """Backend-only body and collider properties for one tensor-view row.

    Attributes:
        disable_gravity: Gravity-disable flags, in backend body order.
        materials: Static friction, dynamic friction and restitution per collider.
        contact_offsets: Collision contact offsets [m].
        rest_offsets: Collision rest offsets [m].
    """

    disable_gravity: np.ndarray
    materials: np.ndarray
    contact_offsets: np.ndarray
    rest_offsets: np.ndarray


class SceneExporter:
    """Export a complete scene environment while preserving its authored USD content.

    Args:
        scene: Initialized scene; stepping must be paused during export.
        fixed_configuration: Preserve spawn-time physical schemas and export initialized
            state and resolved actuator solver properties. Requires one environment before
            its first physics step. Use :meth:`export_from_cfg` to construct this snapshot
            without training events or runtime randomization. Otherwise capture supported
            effective runtime properties, as :func:`export_environment_to_usd` does.
    """

    def __init__(self, scene: InteractiveScene, *, fixed_configuration: bool = False) -> None:
        self.scene = scene
        self.fixed_configuration = fixed_configuration

    @classmethod
    def export_from_cfg(cls, scene_cfg: InteractiveSceneCfg, sim_cfg: SimulationCfg, usd_path: str) -> str:
        """Construct and export a fixed single-environment deployment configuration.

        Uses normal scene construction and asset initialization; no task events run. The
        snapshot follows application of the configured default state and precedes stepping.
        Launch the selected backend normally before calling this method; it owns a fresh
        simulation context and cannot be nested in an existing one. Inputs are copied.

        Args:
            scene_cfg: Fixed scene configuration with ``num_envs=1``.
            sim_cfg: Simulation configuration, including timestep [s] and gravity [m/s^2].
            usd_path: Destination USD file.

        Returns:
            The destination after complete export succeeds.
        """
        from copy import deepcopy

        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext, build_simulation_context

        if scene_cfg.num_envs != 1:
            raise ValueError("Fixed deployment export requires a single-environment scene configuration.")
        if SimulationContext.instance() is not None:
            raise RuntimeError("export_from_cfg owns a fresh simulation context; an active context already exists.")
        with build_simulation_context(sim_cfg=deepcopy(sim_cfg)) as sim:
            scene = InteractiveScene(deepcopy(scene_cfg))
            sim.reset()
            scene.reset_to_default()
            sim.forward()
            scene.update(0.0)
            return cls(scene, fixed_configuration=True).export(usd_path)

    def create_snapshot(self, env_index: int = 0) -> Usd.Stage:
        """Copy the selected authored environment and validate its physical scope."""
        if self.fixed_configuration:
            if self.scene.num_envs != 1 or env_index != 0:
                raise ValueError("Fixed deployment export requires exactly one environment (id 0).")
            if self.scene.sim.get_physics_step_count() != 0:
                raise ValueError("Fixed deployment export must precede the first physics step.")
            from isaaclab.actuators.actuator_base_cfg import _is_implicit_actuator_cfg

            for assets in (self.scene.articulations, self.scene.rigid_objects, self.scene.rigid_object_collections):
                for asset in assets.values():
                    validate_configuration_coverage(asset.cfg)
                    for cfg in getattr(asset.cfg, "rigid_objects", {}).values():
                        validate_configuration_coverage(cfg)
            for asset in self.scene.articulations.values():
                for name, cfg in asset.cfg.actuators.items():
                    validate_configuration_coverage(cfg, actuator=True)
                    if not _is_implicit_actuator_cfg(cfg) and name not in asset.actuators.usd_actuator_groups:
                        raise NotImplementedError(
                            f"Controller {name!r} on {asset.cfg.prim_path} has no USD representation. "
                            "Enable native actuators or supply a supported deployment controller."
                        )
        snapshot = create_environment_snapshot(self.scene, env_index)
        if self.fixed_configuration:
            snapshot.GetRootLayer().customLayerData = {
                **snapshot.GetRootLayer().customLayerData,
                "isaaclab:snapshot": "fixed configuration after default-state initialization, before stepping",
                "isaaclab:sensorRuntime": ", ".join(self.scene.sensors),
            }
        return snapshot

    def export(self, usd_path: str, env_index: int = 0) -> str:
        """Export one complete environment and shared resources without modifying the live stage."""
        from importlib import import_module

        module = self.scene.sim.physics_manager.__module__
        package = module.split(".")[0]
        if package not in {"isaaclab_physx", "isaaclab_ov", "isaaclab_newton"}:
            raise NotImplementedError(f"Scene export is unavailable for {module}.")
        adapter = import_module(f"{package}.sim.usd_export")
        return adapter.export_scene(self, usd_path, env_index)

    def _export_with_adapter(
        self, usd_path, env_index, resolver, read_properties, gravity, root_paths=None, write_scene=None
    ) -> str:
        """Resolve backend identities, then use the common rigid-body/joint USD writers."""
        scene = self.scene
        root_paths = root_paths or (lambda asset: asset.root_view.prim_paths)
        snapshot = self.create_snapshot(env_index)
        if UsdGeom.GetStageMetersPerUnit(snapshot) != 1.0 or UsdPhysics.GetStageKilogramsPerUnit(snapshot) != 1.0:
            raise NotImplementedError("Environment joint export currently requires SI stage units.")
        written = set()
        for articulation, assets in ((True, scene.articulations), (False, scene.rigid_objects)):
            for asset in assets.values():
                roots = root_paths(asset)
                for row in environment_asset_rows(scene, asset.cfg.prim_path, roots, env_index):
                    if articulation:
                        if not self.fixed_configuration:
                            check_articulation_export(asset)
                        paths = resolver(asset, row, snapshot)
                        ArticulationExporter(asset, lambda _asset, _row: paths).write_to_stage(
                            row, stage=snapshot, fixed_configuration=self.fixed_configuration
                        )
                        bodies = paths.bodies
                    else:
                        bodies = [roots[row]]
                        write_body_properties(
                            snapshot,
                            bodies,
                            BodyPhysicsProperties.from_data(asset.data, row),
                            preserve_authored_mass=self.fixed_configuration,
                        )
                    written.update(bodies)
                    if self.fixed_configuration:
                        continue
                    properties = read_properties(asset, row, articulation)
                    flags = np.asarray(properties.disable_gravity).reshape(-1)
                    if len(flags) != len(bodies):
                        raise RuntimeError(f"Backend body coverage differs for {asset.cfg.prim_path}.")
                    for path, flag in zip(bodies, flags):
                        prim = snapshot.GetPrimAtPath(path)
                        prim.AddAppliedSchema("PhysxRigidBodyAPI")
                        safe_set_attribute_on_usd_prim(
                            prim, "physxRigidBody:disableGravity", bool(flag), camel_case=False
                        )
                    write_collision_properties(
                        snapshot, bodies, properties.materials, properties.contact_offsets, properties.rest_offsets
                    )
                    written.update(bodies)
        for asset in scene.rigid_object_collections.values():
            roots = root_paths(asset)
            for name in asset.body_names:
                cfg = asset.cfg.rigid_objects[name]
                for row in environment_asset_rows(scene, cfg.prim_path, roots, env_index):
                    # RigidObjectCollection's public contract is body-major in the view, env-major in data.
                    body, instance = divmod(row, asset.num_instances)
                    prim = snapshot.GetPrimAtPath(roots[row])
                    write_body_properties(
                        snapshot,
                        [roots[row]],
                        BodyPhysicsProperties.from_data(asset.data, instance),
                        [body],
                        preserve_authored_mass=self.fixed_configuration,
                    )
                    written.add(roots[row])
                    if self.fixed_configuration:
                        continue
                    properties = read_properties(asset, row, False)
                    prim.AddAppliedSchema("PhysxRigidBodyAPI")
                    safe_set_attribute_on_usd_prim(
                        prim, "physxRigidBody:disableGravity", bool(properties.disable_gravity), camel_case=False
                    )
                    write_collision_properties(
                        snapshot,
                        [roots[row]],
                        properties.materials,
                        properties.contact_offsets,
                        properties.rest_offsets,
                    )
                    written.add(roots[row])
        if self.fixed_configuration:
            author_fixed_root_frames(snapshot, written)
        check_body_coverage(snapshot, written)
        author_gravity(snapshot, scene.physics_scene_path, gravity)
        physics_scene = snapshot.GetPrimAtPath(scene.physics_scene_path)
        physics_scene.AddAppliedSchema("PhysxSceneAPI")
        frequency = 1.0 / scene.sim.get_physics_dt()
        if not math.isclose(frequency, round(frequency), rel_tol=1e-6):
            raise NotImplementedError("PhysX USD timeStepsPerSecond cannot represent this simulation timestep exactly.")
        physics_scene.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(round(frequency))
        if write_scene is not None:
            write_scene(snapshot)
        _validate_dependencies(snapshot)
        return save_environment_snapshot(snapshot, usd_path)


def export_stage_environment(
    scene: InteractiveScene,
    usd_path: str,
    env_index: int,
    resolver: Callable[[BaseArticulation, int, Usd.Stage], ArticulationPrimPaths],
    read_properties: Callable[
        [BaseArticulation | BaseRigidObject | BaseRigidObjectCollection, int, bool], RigidBodyExportProperties
    ],
    gravity: np.ndarray,
) -> str:
    """Compatibility entry for stage-backed adapters; new callers can use :class:`SceneExporter`."""
    return SceneExporter(scene)._export_with_adapter(usd_path, env_index, resolver, read_properties, gravity)


def check_body_coverage(stage: Usd.Stage, written: set[str]) -> None:
    """Reject physical bodies retained from USD without a corresponding effective backend snapshot."""
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
