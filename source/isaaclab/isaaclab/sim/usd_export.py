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
follows that fault line: this module owns the value-to-USD half, and each backend supplies its paths
through :class:`ArticulationPrimPaths`.

Newton is deliberately not built on this. A ``newton.Model`` is the sole description of what it
simulates and the source stage is not retained, so :mod:`isaaclab_newton.sim.usd_export` rebuilds a
stage instead of patching one. The stage-backed backends keep hierarchy, geometry and joint topology
on the stage and correct, so patching leaves strictly less to get wrong.

Ordering
--------

Backends index links and DOFs in *backend* order, while the data arrays are in *public* API order,
which :attr:`ArticulationCfg.body_ordering` may permute. The two are joined by name; joining by
index silently mislabels every body of a reordered articulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pxr import Usd, UsdPhysics

from isaaclab.sim.utils import safe_set_attribute_on_usd_prim

if TYPE_CHECKING:
    from isaaclab.assets import BaseArticulation

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


def _author_joint_state(
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

    ``friction`` is the (static effort, dynamic effort, viscous coefficient) triple of the drive axis.
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

    if not (math.isfinite(lower_limit) and math.isfinite(upper_limit)):
        return
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
    """Author an articulation's simulated state onto the prims it was spawned from.

    Body masses and joint drive gains, armature, friction and limits are read from the simulation
    and written onto the stage, replacing the spawn-time values it still carries. Schemas are applied
    only where absent; existing attributes are overwritten in place.

    Args:
        articulation: The articulation to read. It must be initialized, since the values come from
            the running simulation rather than from its configuration.
        prim_paths: The environment's prim paths, supplied by the backend.
        env_index: Environment whose state to author. Defaults to ``0``.
        stage: Stage to author onto; it must hold the same prim paths as the live stage. Defaults to
            the live stage itself, which the simulation keeps reading: on PhysX, applying a schema to
            a prim that is an articulation root invalidates every articulation view on the stage for
            the rest of the session. Pass a flattened copy, as :func:`export_articulation_to_usd`
            does, to leave the running simulation untouched.

    Returns:
        The prim paths written, bodies first.

    Raises:
        ValueError: If ``env_index`` is not an environment of the articulation.
        RuntimeError: If a path the backend supplied is not a prim on the stage, or a backend-order
            name is absent from the public order. Both are contract violations -- the view is built
            from the stage and the two orders are permutations of one another -- so the stage no
            longer describes what the backend is simulating and a partial export would hide that.
    """
    if not 0 <= env_index < articulation.num_instances:
        raise ValueError(
            f"Environment {env_index} is out of range for an articulation with"
            f" {articulation.num_instances} instance(s)."
        )

    data = articulation.data
    stage = articulation.stage if stage is None else stage

    # Backend order indexes the paths; public order indexes the data. Join by name -- a reordered
    # articulation would otherwise take every value from the wrong row.
    body_row = {name: index for index, name in enumerate(articulation.body_names)}
    joint_row = {name: index for index, name in enumerate(articulation.joint_names)}

    # One host transfer per array; indexing a device tensor per element would sync on every read.
    masses = data.body_mass.torch[env_index].tolist()
    stiffness = data.joint_stiffness.torch[env_index].tolist()
    damping = data.joint_damping.torch[env_index].tolist()
    armature = data.joint_armature.torch[env_index].tolist()
    static_friction = data.joint_friction_coeff.torch[env_index].tolist()
    dynamic_friction = data.joint_dynamic_friction_coeff.torch[env_index].tolist()
    viscous_friction = data.joint_viscous_friction_coeff.torch[env_index].tolist()
    limits = data.joint_pos_limits.torch[env_index].tolist()

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
        mass_api = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr().Set(masses[row])
        written.append(path)
    for (prim, row), path in zip(joint_targets, prim_paths.joints):
        _author_joint_state(
            prim,
            stiffness=stiffness[row],
            damping=damping[row],
            armature=armature[row],
            friction=(static_friction[row], dynamic_friction[row], viscous_friction[row]),
            lower_limit=limits[row][0],
            upper_limit=limits[row][1],
        )
        written.append(path)

    return written


def export_articulation_to_usd(
    articulation: BaseArticulation, prim_paths: ArticulationPrimPaths, usd_path: str, env_index: int = 0
) -> str:
    """Export one environment's articulation, as simulated, to a USD file.

    The live stage is flattened first and the simulated state is authored onto that snapshot, so the
    running simulation never sees the edits. The live stage's own file is not saved.

    Args:
        articulation: The articulation to export.
        prim_paths: The environment's prim paths, supplied by the backend.
        usd_path: Destination path for the USD file.
        env_index: Environment to export. Defaults to ``0``.

    Returns:
        The path the stage was written to.
    """
    snapshot = Usd.Stage.Open(articulation.stage.Flatten())
    write_articulation_state_to_stage(articulation, prim_paths, env_index=env_index, stage=snapshot)
    snapshot.Export(str(usd_path))
    return str(usd_path)
