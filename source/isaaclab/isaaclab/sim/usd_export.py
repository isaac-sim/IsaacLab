# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write a running articulation's simulated state back onto its USD stage.

Isaac Lab configures a scene in two places. Properties authored while spawning land on the stage,
so they already describe the simulation; properties written afterwards go to the physics backend's
buffers, which the stage never sees. Saving the stage of a running scene therefore emits a file that
*looks* complete while silently carrying the spawn-time value of everything overridden since --
gains re-tuned by an actuator model, masses replaced by an event term, limits narrowed by a
curriculum.

This module authors the diverged properties back onto the prims they came from.

Layering
--------

Reading values is backend-independent: every backend implements
:class:`~isaaclab.assets.BaseArticulationData`, so the same properties are available whatever is
simulating. Recovering *prim paths* is not -- each backend records provenance its own way. The split
follows that fault line: :class:`ArticulationExporter` owns the value-to-USD half and takes the
backend's path resolution as a callable returning :class:`ArticulationPrimPaths`.

The environment entry point uses the scene registry and ClonePlan to retain one environment plus
shared USD resources. Newton uses the same selection, dependency handling, mass/inertia, materials
and gravity writers; its adapter additionally handles model provenance and collision geometry.

Ordering
--------

Backends index links and DOFs in *backend* order, while the data arrays are in *public* API order,
which :attr:`ArticulationCfg.body_ordering` may permute. The two are joined by name; joining by
index silently mislabels every body of a reordered articulation.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

from isaaclab.sim.utils import safe_set_attribute_on_usd_prim

if TYPE_CHECKING:
    from isaaclab.assets import BaseArticulation, BaseRigidObject, BaseRigidObjectCollection
    from isaaclab.scene import InteractiveScene

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

    def write_to_stage(self, env_index: int = 0, *, stage: Usd.Stage | None = None) -> list[str]:
        """Author the simulated state onto the prims the articulation was spawned from.

        Body masses and joint drive gains, armature, friction and limits are read from the simulation
        and written onto the stage, replacing the spawn-time values it still carries. Schemas are
        applied only where absent; existing attributes are overwritten in place.

        Args:
            env_index: Environment whose state to author. Defaults to ``0``.
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
        masses = data.body_mass.torch[env_index].tolist()
        inertias = data.body_inertia.torch[env_index].cpu().numpy().reshape(-1, 3, 3)
        coms = data.body_com_pose_b.torch[env_index].cpu().numpy()
        poses = data.body_link_pose_w.torch[env_index].cpu().numpy()
        velocities = data.body_com_vel_w.torch[env_index].cpu().numpy()
        stiffness = data.joint_stiffness.torch[env_index].tolist()
        damping = data.joint_damping.torch[env_index].tolist()
        armature = data.joint_armature.torch[env_index].tolist()
        static_friction = data.joint_friction_coeff.torch[env_index].tolist()
        dynamic_friction = data.joint_dynamic_friction_coeff.torch[env_index].tolist()
        viscous_friction = data.joint_viscous_friction_coeff.torch[env_index].tolist()
        limits = data.joint_pos_limits.torch[env_index].tolist()
        effort_limits = data.joint_effort_limits.torch[env_index].tolist()
        velocity_limits = data.joint_vel_limits.torch[env_index].tolist()
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
        body_targets = [resolve(path, body_names[i], body_row, "Body") for i, path in enumerate(prim_paths.bodies)]
        joint_targets = [resolve(path, joint_names[i], joint_row, "Joint") for i, path in enumerate(prim_paths.joints)]

        written: list[str] = []
        for (prim, row), path in zip(body_targets, prim_paths.bodies):
            author_mass_properties(prim, masses[row], inertias[row], coms[row])
            written.append(path)
        for prim, row in sorted(body_targets, key=lambda target: target[0].GetPath().pathElementCount):
            author_body_state(prim, poses[row], velocities[row])
        for (prim, row), path in zip(joint_targets, prim_paths.joints):
            self._author_joint(
                prim,
                stiffness=stiffness[row],
                damping=damping[row],
                armature=armature[row],
                friction=(static_friction[row], dynamic_friction[row], viscous_friction[row]),
                lower_limit=limits[row][0],
                upper_limit=limits[row][1],
            )
            token = _DRIVE_TOKEN.get(prim.GetTypeName())
            if token:
                UsdPhysics.DriveAPI(prim, token).CreateMaxForceAttr().Set(float(effort_limits[row]))
                velocity = float(velocity_limits[row])
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
        """Export a stage copy with this articulation's effective values written back.

        This legacy articulation-only writer does not select a complete environment. For deployment,
        use :func:`export_environment_to_usd` with the scene registry.

        The live stage is flattened first and the state is authored onto that snapshot, so the running
        simulation never sees the edits. The live stage's own file is not saved.

        Args:
            usd_path: Destination path for the USD file.
            env_index: Environment to export. Defaults to ``0``.

        Returns:
            The path the stage was written to.
        """
        snapshot = Usd.Stage.Open(self.articulation.stage.Flatten())
        self.write_to_stage(env_index, stage=snapshot)
        snapshot.Export(str(usd_path))
        return str(usd_path)

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
    if plan is None:
        raise ValueError("Environment export requires the scene's ClonePlan.")
    env_ids = list(range(plan.clone_mask.shape[1])) if plan.env_ids is None else plan.env_ids.tolist()
    if env_index not in env_ids:
        raise ValueError(f"Environment {env_index} is out of range for ClonePlan environments {env_ids}.")
    for family in ("deformable_objects", "cable_objects", "surface_grippers"):
        if getattr(scene, family):
            raise NotImplementedError(f"Environment export does not support {family}: {list(getattr(scene, family))}")
    snapshot = Usd.Stage.Open(scene.sim.stage.Flatten())
    # Make instance children writable on the copy, never on the live stage.
    while True:
        instances = [prim for prim in snapshot.Traverse() if prim.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    snapshot = Usd.Stage.Open(snapshot.Flatten())
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
    for path in sorted(set(removed), key=lambda p: len(str(p)), reverse=True):
        if not any(p.HasPrefix(path) for p in selected):
            snapshot.RemovePrim(path)
    _validate_dependencies(snapshot)
    snapshot.GetRootLayer().customLayerData = {
        **snapshot.GetRootLayer().customLayerData,
        "isaaclab:environment": env_index,
        "isaaclab:snapshot": "initialized physical configuration; controllers and sensor runtime excluded",
    }
    return snapshot


def _validate_dependencies(stage: Usd.Stage) -> None:
    """Prune deleted-replica filter memberships and reject broken physical or visual dependencies."""
    for prim in stage.Traverse():
        for prop in prim.GetProperties():
            paths = prop.GetTargets() if isinstance(prop, Usd.Relationship) else prop.GetConnections()
            missing = [p for p in paths if not stage.GetPrimAtPath(p.GetPrimPath())]
            if not missing:
                continue
            # Filter/collection membership in deleted replicas is intentionally discarded. Physical
            # joints and shader connections must never be silently severed.
            filtering = prop.GetName() == "physics:filteredPairs" or prop.GetName().startswith("collection:")
            if filtering and isinstance(prop, Usd.Relationship):
                prop.SetTargets([p for p in paths if p not in missing])
            else:
                raise RuntimeError(f"Unresolved export dependency {prop.GetPath()}: {missing}")


def save_environment_snapshot(stage: Usd.Stage, usd_path: str) -> str:
    """Save a complete snapshot atomically, leaving an existing destination intact on failure."""
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
    masses = asset.data.body_mass.torch[row].cpu().numpy().reshape(-1)
    inertias = asset.data.body_inertia.torch[row].cpu().numpy().reshape(-1, 3, 3)
    coms = asset.data.body_com_pose_b.torch[row].cpu().numpy().reshape(-1, 7)
    poses = asset.data.body_link_pose_w.torch[row].cpu().numpy().reshape(-1, 7)
    velocities = asset.data.body_com_vel_w.torch[row].cpu().numpy().reshape(-1, 6)
    if len(body_paths) != len(masses):
        raise RuntimeError(f"Body coverage mismatch for {asset.cfg.prim_path}: {body_paths}")
    for path, mass, inertia, com, pose, velocity in zip(body_paths, masses, inertias, coms, poses, velocities):
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise RuntimeError(f"Missing rigid object body {path}.")
        author_mass_properties(prim, mass, inertia, com)
        author_body_state(prim, pose, velocity)
    return body_paths


def environment_asset_rows(scene: InteractiveScene, path_expr: str, roots: list[str], env_index: int) -> list[int]:
    """Select backend rows by stable prim identity, including shared and partially populated assets."""
    from isaaclab.cloner import query

    plan = scene.clone_plan
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
    from importlib import import_module

    # ResolvableString exposes the resolved manager metadata without changing its string type.
    module = scene.sim.physics_manager.__module__
    package = module.split(".")[0]
    if package not in {"isaaclab_physx", "isaaclab_ov", "isaaclab_newton"}:
        raise NotImplementedError(f"Environment export is unavailable for {module}.")
    return import_module(f"{package}.sim.usd_export").export_environment_to_usd(scene, usd_path, env_index)


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
    """Shared environment export for backends retaining their authored USD scene.

    Backend hooks resolve paths and read properties missing from public object-data interfaces.
    All selection, body/joint authoring, dependencies and completeness checks are shared.
    """
    snapshot = create_environment_snapshot(scene, env_index)
    if UsdGeom.GetStageMetersPerUnit(snapshot) != 1.0 or UsdPhysics.GetStageKilogramsPerUnit(snapshot) != 1.0:
        raise NotImplementedError("Environment joint export currently requires SI stage units.")
    written = set()
    for articulation, assets in ((True, scene.articulations), (False, scene.rigid_objects)):
        for asset in assets.values():
            view = asset.root_view
            for row in environment_asset_rows(scene, asset.cfg.prim_path, view.prim_paths, env_index):
                if articulation:
                    check_articulation_export(asset)
                    paths = resolver(asset, row, snapshot)
                    ArticulationExporter(asset, lambda _asset, _row: paths).write_to_stage(row, stage=snapshot)
                    bodies = paths.bodies
                else:
                    bodies = [view.prim_paths[row]]
                    write_rigid_object_state_to_stage(asset, bodies, row, snapshot)
                properties = read_properties(asset, row, articulation)
                flags = np.asarray(properties.disable_gravity).reshape(-1)
                if len(flags) != len(bodies):
                    raise RuntimeError(f"Backend body coverage differs for {asset.cfg.prim_path}.")
                for path, flag in zip(bodies, flags):
                    prim = snapshot.GetPrimAtPath(path)
                    prim.AddAppliedSchema("PhysxRigidBodyAPI")
                    safe_set_attribute_on_usd_prim(prim, "physxRigidBody:disableGravity", bool(flag), camel_case=False)
                write_collision_properties(
                    snapshot, bodies, properties.materials, properties.contact_offsets, properties.rest_offsets
                )
                written.update(bodies)
    for asset in scene.rigid_object_collections.values():
        masses = asset.data.body_mass.torch.cpu().numpy()
        inertias = asset.data.body_inertia.torch.cpu().numpy()
        coms = asset.data.body_com_pose_b.torch.cpu().numpy()
        poses = asset.data.body_link_pose_w.torch.cpu().numpy()
        velocities = asset.data.body_com_vel_w.torch.cpu().numpy()
        roots = asset.root_view.prim_paths
        for name in asset.body_names:
            cfg = asset.cfg.rigid_objects[name]
            for row in environment_asset_rows(scene, cfg.prim_path, roots, env_index):
                # RigidObjectCollection's public contract is body-major in the view, env-major in data.
                body, instance = divmod(row, asset.num_instances)
                prim = snapshot.GetPrimAtPath(roots[row])
                author_mass_properties(prim, masses[instance, body], inertias[instance, body], coms[instance, body])
                author_body_state(prim, poses[instance, body], velocities[instance, body])
                properties = read_properties(asset, row, False)
                prim.AddAppliedSchema("PhysxRigidBodyAPI")
                safe_set_attribute_on_usd_prim(
                    prim, "physxRigidBody:disableGravity", bool(properties.disable_gravity), camel_case=False
                )
                write_collision_properties(
                    snapshot, [roots[row]], properties.materials, properties.contact_offsets, properties.rest_offsets
                )
                written.add(roots[row])
    check_body_coverage(snapshot, written)
    author_gravity(snapshot, scene.physics_scene_path, gravity)
    physics_scene = snapshot.GetPrimAtPath(scene.physics_scene_path)
    physics_scene.AddAppliedSchema("PhysxSceneAPI")
    frequency = 1.0 / scene.sim.get_physics_dt()
    if not math.isclose(frequency, round(frequency), rel_tol=1e-6):
        raise NotImplementedError("PhysX USD timeStepsPerSecond cannot represent this simulation timestep exactly.")
    physics_scene.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(round(frequency))
    return save_environment_snapshot(snapshot, usd_path)


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
