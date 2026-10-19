# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD export in the PhysX dialect, shared by the PhysX-based backends."""

from __future__ import annotations

import logging
import math
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from pxr import Usd, UsdPhysics

from ..utils.queries import find_matching_prim_paths
from .base_usd_exporter import USD_PHYSICS_JOINT_ATTRIBUTES, USD_PHYSICS_MATERIAL_ATTRIBUTES, BaseUsdExporter, UsdEdit

logger = logging.getLogger(__name__)

Readings = dict[tuple[str, str | None], dict[str, np.ndarray]]
"""Engine values in SI units and radians, keyed by USD prim path and joint axis (None for bodies).

Joint entries hold the properties of :data:`JOINT_ATTRIBUTES`. Body entries hold ``mass``, ``com`` [m]
and ``inertia`` [kg*m^2, row-major 3x3 about the COM in body axes]; rigid bodies outside articulations
also hold the per-shape ``contact_offset`` and ``rest_offset`` [m] and ``material`` (static friction,
dynamic friction, restitution).
"""

JOINT_ATTRIBUTES = {
    **USD_PHYSICS_JOINT_ATTRIBUTES,
    "armature": ("physxJoint:armature", "PhysxJointAPI", 1.0),
    "max_velocity": ("physxJoint:maxJointVelocity", "PhysxJointAPI", math.degrees(1.0)),
    "static_friction": ("physxJointAxis:{axis}:staticFrictionEffort", "PhysxJointAxisAPI:{axis}", 1.0),
    "dynamic_friction": ("physxJointAxis:{axis}:dynamicFrictionEffort", "PhysxJointAxisAPI:{axis}", 1.0),
    "viscous_friction": (
        "physxJointAxis:{axis}:viscousFrictionCoefficient",
        "PhysxJointAxisAPI:{axis}",
        math.radians(1.0),
    ),
}
"""Joint property -> USD attribute, required API schema, and scale to USD units for angular DOFs."""

COLLIDER_ATTRIBUTES = {"contact_offset": "physxCollision:contactOffset", "rest_offset": "physxCollision:restOffset"}
"""Per-shape property -> USD attribute on the body's colliders."""


class PhysxUsdExporter(BaseUsdExporter):
    """Base of the PhysX-based backends: writes their engine values in the PhysX USD dialect.

    Joint and body values come from the scene assets' data, which holds the engine values on every
    backend. A backend only reads the collider values of rigid bodies, which the asset data does not hold.
    """

    def compute_diff(self, stage: Usd.Stage) -> list[UsdEdit]:
        """Return the edits writing every engine value that ``stage`` does not already state.

        Args:
            stage: The exported stage, reduced to the selected environment.

        Returns:
            Edits in the PhysX USD dialect.
        """
        edits, materials = [], {}
        for (path, axis), values in self._read_assets(stage).items():
            prim = stage.GetPrimAtPath(path)
            if axis is not None:
                # A joint without a USD drive and without gains is not driven; writing gains would add a drive.
                driven = prim.HasAPI(UsdPhysics.DriveAPI, axis) or values["stiffness"] or values["damping"]
                for name, (attribute, schema, angular_scale) in JOINT_ATTRIBUTES.items():
                    if attribute.startswith("drive:") and not driven:
                        continue
                    scale = angular_scale if axis == "angular" else 1.0
                    schema = schema and schema.format(axis=axis)
                    value = float(values[name]) * scale
                    edits.append(self._value_edit(prim, attribute.format(axis=axis), value, schema=schema))
                continue
            edits += self._mass_edits(prim, float(values["mass"]), values["com"], values["inertia"])
            colliders = self._colliders(stage, path)
            for name, attribute in COLLIDER_ATTRIBUTES.items():
                if name in values:
                    # The engine reports shapes per body, without their prim identities.
                    value = self._uniform(values[name], path, name)
                    edits += [
                        self._value_edit(stage.GetPrimAtPath(c), attribute, value, schema="PhysxCollisionAPI")
                        for c in colliders
                    ]
            if "material" in values:
                material = self._uniform(values["material"], path, "material")
                materials.update(
                    {
                        collider: dict(zip(USD_PHYSICS_MATERIAL_ATTRIBUTES.values(), material.tolist()))
                        for collider in colliders
                    }
                )
        edits = [edit for edit in edits if edit is not None] + self._material_edits(stage, materials)
        return edits + self._gravity_edits(stage, self.scene.sim.backend_manager.get_gravity())

    @abstractmethod
    def _read_rigid_body(self, path: str) -> dict[str, np.ndarray]:
        """Read the per-shape collider values of a rigid body outside articulations from the engine.

        Args:
            path: Rigid body prim path.

        Returns:
            The body's ``material``, ``contact_offset`` and ``rest_offset`` entries of :data:`Readings`.
        """
        raise NotImplementedError

    """
    Engine values.
    """

    def _read_assets(self, stage: Usd.Stage) -> Readings:
        """Read the engine values of the scene's assets, keyed by their prims in the exported stage."""
        readings = {}
        for articulation in self.scene.articulations.values():
            readings.update(self._read_articulation(stage, articulation))
        for rigid_object in self.scene.rigid_objects.values():
            path = self._rigid_body_path(stage, rigid_object.cfg.prim_path)
            readings[path, None] = {**self._body_values(rigid_object.data, 0), **self._read_rigid_body(path)}
        for collection in self.scene.rigid_object_collections.values():
            # Collection bodies are ordered as the configured objects.
            for index, cfg in enumerate(collection.cfg.rigid_objects.values()):
                path = self._rigid_body_path(stage, cfg.prim_path)
                readings[path, None] = {**self._body_values(collection.data, index), **self._read_rigid_body(path)}
        exported = {path for path, _ in readings}
        missing = [
            str(prim.GetPath())
            for prim in stage.Traverse(Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and str(prim.GetPath()) not in exported
        ]
        if missing:
            logger.warning(f"Physics bodies outside the scene's assets keep their USD values: {missing}.")
        return readings

    def _read_articulation(self, stage: Usd.Stage, articulation) -> Readings:
        """Read an articulation's joint and link values, matching its joint and body names to prims."""
        scope = stage.GetPrimAtPath(self._asset_path(stage, articulation.cfg.prim_path))
        joints = self._named(scope, lambda prim: prim.IsA(UsdPhysics.Joint))
        links = self._named(scope, lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI))
        data = articulation.data
        limits = self._row(data.joint_pos_limits)
        dofs = {
            "stiffness": self._row(data.joint_stiffness),
            "damping": self._row(data.joint_damping),
            "max_force": self._row(data.joint_effort_limits),
            "lower_limit": limits[..., 0],
            "upper_limit": limits[..., 1],
            "armature": self._row(data.joint_armature),
            "max_velocity": self._row(data.joint_vel_limits),
            "static_friction": self._row(data.joint_friction_coeff),
            "dynamic_friction": self._row(data.joint_dynamic_friction_coeff),
            "viscous_friction": self._row(data.joint_viscous_friction_coeff),
        }
        readings, unsupported = {}, set()
        for index, name in enumerate(articulation.joint_names):
            path = self._unique(joints, name, scope)
            prim = stage.GetPrimAtPath(path)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                readings[path, "angular"] = {prop: values[index] for prop, values in dofs.items()}
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                readings[path, "linear"] = {prop: values[index] for prop, values in dofs.items()}
            else:
                unsupported.add(path)
        if unsupported:
            logger.warning(f"Joints other than revolute and prismatic are not exported: {sorted(unsupported)}.")
        for index, name in enumerate(articulation.body_names):
            readings[self._unique(links, name, scope), None] = self._body_values(data, index)
        return readings

    def _body_values(self, data, index: int) -> dict[str, np.ndarray]:
        """Mass properties of one body of an asset in the selected environment."""
        return {
            "mass": float(self._row(data.body_mass)[index]),
            "com": self._row(data.body_com_pose_b)[index, :3],
            "inertia": self._row(data.body_inertia)[index],
        }

    def _row(self, values) -> np.ndarray:
        """The selected environment's entry of an asset data array."""
        return values.torch[self.env_id].detach().cpu().numpy()

    """
    Exported stage.
    """

    @staticmethod
    def _asset_path(stage: Usd.Stage, prim_path: str) -> str:
        """The prim of an asset in the exported stage, which keeps one environment."""
        paths = find_matching_prim_paths(prim_path, stage)
        if len(paths) != 1:
            raise RuntimeError(f"Expected one prim matching '{prim_path}' in the exported stage, found {paths}.")
        return paths[0]

    @classmethod
    def _rigid_body_path(cls, stage: Usd.Stage, prim_path: str) -> str:
        """The rigid body of a rigid object: its prim, or the rigid body below it."""
        root = stage.GetPrimAtPath(cls._asset_path(stage, prim_path))
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(prim.GetPath())
        raise RuntimeError(f"No rigid body below '{root.GetPath()}'.")

    @staticmethod
    def _named(scope: Usd.Prim, predicate: Callable[[Usd.Prim], bool]) -> dict[str, list[str]]:
        """Paths of the prims below ``scope`` that satisfy ``predicate``, grouped by prim name."""
        named = defaultdict(list)
        for prim in Usd.PrimRange(scope, Usd.TraverseInstanceProxies()):
            if predicate(prim):
                named[prim.GetName()].append(str(prim.GetPath()))
        return named

    @staticmethod
    def _unique(named: dict[str, list[str]], name: str, scope: Usd.Prim) -> str:
        """The one prim an asset data name refers to; the data names prims without their paths."""
        paths = named.get(name, [])
        if len(paths) != 1:
            raise RuntimeError(f"Expected one prim named '{name}' below '{scope.GetPath()}', found {paths}.")
        return paths[0]

    @staticmethod
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

    @staticmethod
    def _uniform(values: np.ndarray, path: str, name: str) -> np.ndarray:
        """The one value of all shapes of a body; different values have no per-shape USD identity."""
        values = np.asarray(values)
        if not np.all(values == values[0]):
            raise NotImplementedError(f"The {name} of body '{path}' differs between its shapes.")
        return values[0]
