# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD export in the PhysX dialect, shared by the PhysX-based backends."""

from __future__ import annotations

import logging
import math
from abc import abstractmethod
from collections.abc import Sequence

import numpy as np

from pxr import Usd, UsdPhysics

from .base_usd_exporter import (
    BaseUsdExporter,
    UsdEdit,
    gravity_edits,
    mass_edits,
    material_edits,
    value_edit,
)

logger = logging.getLogger(__name__)

Readings = dict[tuple[str, str | None], dict[str, np.ndarray]]
"""Engine values in SI units and radians, keyed by USD prim path and joint axis (None for bodies).

Joint entries hold the properties of :data:`JOINT_ATTRIBUTES`. Body entries hold ``mass``,
``com`` [m] and ``inertia`` [kg*m^2, row-major 3x3 about the COM in body axes], ``disable_gravity``, the per-shape
``contact_offset`` and ``rest_offset`` [m], ``material`` (static friction, dynamic friction, restitution
per shape, rigid bodies outside articulations only).
Multi-axis joint DOFs have a None axis.
"""

_DEGREES = 180.0 / math.pi
_PER_DEGREE = math.pi / 180.0

JOINT_ATTRIBUTES = {
    "stiffness": ("drive:{axis}:physics:stiffness", "PhysicsDriveAPI:{axis}", _PER_DEGREE),
    "damping": ("drive:{axis}:physics:damping", "PhysicsDriveAPI:{axis}", _PER_DEGREE),
    "max_force": ("drive:{axis}:physics:maxForce", "PhysicsDriveAPI:{axis}", 1.0),
    "lower_limit": ("physics:lowerLimit", None, _DEGREES),
    "upper_limit": ("physics:upperLimit", None, _DEGREES),
    "armature": ("physxJoint:armature", "PhysxJointAPI", 1.0),
    "max_velocity": ("physxJoint:maxJointVelocity", "PhysxJointAPI", _DEGREES),
    "static_friction": ("physxJointAxis:{axis}:staticFrictionEffort", "PhysxJointAxisAPI:{axis}", 1.0),
    "dynamic_friction": ("physxJointAxis:{axis}:dynamicFrictionEffort", "PhysxJointAxisAPI:{axis}", 1.0),
    "viscous_friction": ("physxJointAxis:{axis}:viscousFrictionCoefficient", "PhysxJointAxisAPI:{axis}", _PER_DEGREE),
}
"""Joint property -> USD attribute, required API schema, and scale to USD units for angular DOFs."""

_DRIVE = ("stiffness", "damping", "max_force")

COLLIDER_ATTRIBUTES = {"contact_offset": "physxCollision:contactOffset", "rest_offset": "physxCollision:restOffset"}
"""Per-shape property -> USD attribute on the body's colliders."""

MATERIAL_ATTRIBUTES = ("physics:staticFriction", "physics:dynamicFriction", "physics:restitution")
"""USD attributes of the per-shape ``material`` components, on the colliders' physics material."""


class PhysxUsdExporter(BaseUsdExporter):
    """Base of the PhysX-based backends: writes their engine readings in the PhysX USD dialect."""

    def compute_diff(self, stage: Usd.Stage) -> list[UsdEdit]:
        """Return the edits writing every engine value that ``stage`` does not already state.

        Args:
            stage: The exported stage, reduced to the selected environment.

        Returns:
            Edits in the PhysX USD dialect.
        """
        live, gravity = self.read_engine(stage)
        edits, materials, multi_axis = [], {}, set()
        for (path, axis), values in live.items():
            prim = stage.GetPrimAtPath(path)
            if axis is None and "stiffness" in values:
                multi_axis.add(path.rsplit(":", 1)[0])
                continue
            if axis is not None:
                # A joint without a USD drive and without gains is not driven; writing gains would add a drive.
                driven = prim.HasAPI(UsdPhysics.DriveAPI, axis) or values["stiffness"] or values["damping"]
                for name, (attribute, schema, angular_scale) in JOINT_ATTRIBUTES.items():
                    if name in _DRIVE and not driven:
                        continue
                    scale = angular_scale if axis == "angular" else 1.0
                    schema = schema and schema.format(axis=axis)
                    edits.append(
                        value_edit(prim, attribute.format(axis=axis), float(values[name]) * scale, schema=schema)
                    )
                continue
            edits += mass_edits(prim, float(values["mass"]), values["com"], values["inertia"])
            if "disable_gravity" in values:
                disabled = bool(values["disable_gravity"])
                edits.append(value_edit(prim, "physxRigidBody:disableGravity", disabled, "bool", "PhysxRigidBodyAPI"))
            colliders = _colliders(stage, path)
            for name, attribute in COLLIDER_ATTRIBUTES.items():
                if name in values:
                    # The engine reports shapes per body, without their prim identities.
                    value = _uniform(values[name], path, name)
                    edits += [
                        value_edit(stage.GetPrimAtPath(c), attribute, value, schema="PhysxCollisionAPI")
                        for c in colliders
                    ]
            if "material" in values:
                material = _uniform(values["material"], path, "material")
                materials.update(
                    {collider: dict(zip(MATERIAL_ATTRIBUTES, material.tolist())) for collider in colliders}
                )
        if multi_axis:
            logger.warning(f"Multi-axis joints are not exported: {sorted(multi_axis)}.")
        edits = [edit for edit in edits if edit is not None] + material_edits(stage, materials)
        return edits + gravity_edits(stage, self.scene.physics_scene_path, gravity)

    @abstractmethod
    def read_engine(self, stage: Usd.Stage) -> tuple[Readings, Sequence[float]]:
        """Read the engine values of the stage's physics objects.

        Args:
            stage: The exported stage, reduced to the selected environment.

        Returns:
            The readings, and the gravity vector [m/s^2].
        """
        raise NotImplementedError


def physics_prims(stage: Usd.Stage) -> tuple[list[str], list[str]]:
    """Articulation roots, and rigid bodies that belong to no articulation."""
    prims = list(stage.Traverse())
    articulations = [str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    scopes = [articulation_scope(stage, root).GetPath() for root in articulations]
    bodies = [
        str(prim.GetPath())
        for prim in prims
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not any(prim.GetPath().HasPrefix(scope) for scope in scopes)
    ]
    return articulations, bodies


def articulation_scope(stage: Usd.Stage, root: str) -> Usd.Prim:
    """The prim containing an articulation's links: its root, or the root's parent if the root is a link."""
    prim = stage.GetPrimAtPath(root)
    return prim.GetParent() if prim.HasAPI(UsdPhysics.RigidBodyAPI) else prim


def _uniform(values: np.ndarray, path: str, name: str) -> np.ndarray:
    """The one value of all shapes of a body; different values have no per-shape USD identity."""
    values = np.asarray(values)
    if not np.all(values == values[0]):
        raise NotImplementedError(f"The {name} of body '{path}' differs between its shapes.")
    return values[0]


def _colliders(stage: Usd.Stage, body: str) -> list[str]:
    """Colliders owned by a body: its collision prims, excluding those of nested bodies."""
    colliders = []
    iterator = iter(Usd.PrimRange(stage.GetPrimAtPath(body), Usd.TraverseInstanceProxies()))
    for prim in iterator:
        if str(prim.GetPath()) != body and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            iterator.PruneChildren()
        elif prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(str(prim.GetPath()))
    return colliders
