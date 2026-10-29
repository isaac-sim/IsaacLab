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

from ..utils.queries import find_matching_prim_paths, resolve_matching_prims_from_source
from .base_usd_exporter import (
    USD_PHYSICS_D6_LIMIT_ATTRIBUTES,
    USD_PHYSICS_JOINT_ATTRIBUTES,
    USD_PHYSICS_MATERIAL_ATTRIBUTES,
    BaseUsdExporter,
    JointAttribute,
    UsdEdit,
)

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
    # Per axis: OVPhysX 0.7's PhysX schema has no PhysxJointAPI, and an axis value overrides it in Isaac Sim.
    "armature": JointAttribute("physxJointAxis:{axis}:armature", "PhysxJointAxisAPI:{axis}", 1.0),
    "max_velocity": JointAttribute(
        "physxJointAxis:{axis}:maxJointVelocity", "PhysxJointAxisAPI:{axis}", math.degrees(1.0)
    ),
    "static_friction": JointAttribute("physxJointAxis:{axis}:staticFrictionEffort", "PhysxJointAxisAPI:{axis}", 1.0),
    "dynamic_friction": JointAttribute("physxJointAxis:{axis}:dynamicFrictionEffort", "PhysxJointAxisAPI:{axis}", 1.0),
    "viscous_friction": JointAttribute(
        "physxJointAxis:{axis}:viscousFrictionCoefficient", "PhysxJointAxisAPI:{axis}", math.radians(1.0)
    ),
}
"""Joint property -> USD attribute."""

D6_JOINT_ATTRIBUTES = {
    **JOINT_ATTRIBUTES,
    **USD_PHYSICS_D6_LIMIT_ATTRIBUTES,
}
""":data:`JOINT_ATTRIBUTES` of a rotational D6 joint, whose axes are ``rotX``, ``rotY`` and ``rotZ``."""

D6_AXES = ("rotX", "rotY", "rotZ", "transX", "transY", "transZ")
"""Axes of a D6 joint; PhysX names a rotational DOF ``<joint>:<n>`` after its index here."""

COLLIDER_ATTRIBUTES = {"contact_offset": "physxCollision:contactOffset", "rest_offset": "physxCollision:restOffset"}
"""Per-shape property -> USD attribute on the body's colliders."""

COLLIDER_READINGS = ("material", *COLLIDER_ATTRIBUTES)
"""Per-shape engine values of a rigid body outside articulations, which the asset data does not hold."""


class PhysxUsdExporter(BaseUsdExporter):
    """Base of the PhysX-based backends: writes their engine values in the PhysX USD dialect.

    The base is what the engine loaded from the stage, read when the physics is ready and before any asset
    writes its configuration; a PhysX import of the stage loads the same. A value is written where the live
    engine differs from it, so Isaac Lab's configuration is stated, and nothing the engine derives by itself is.

    Live joint and body values come from the scene assets' data. A backend reads the base and the collider
    values of rigid bodies, which the asset data does not hold.
    """

    def compute_diff(self, stage: Usd.Stage) -> list[UsdEdit]:
        """Return the edits writing every engine value that differs from what the engine loaded from the stage.

        Args:
            stage: The exported stage, reduced to the selected environment.

        Returns:
            Edits in the PhysX USD dialect.
        """
        edits, materials = [], {}
        skipped = defaultdict(list)

        def value_edit(prim: Usd.Prim, attribute: str, value: float, schema: str | None) -> UsdEdit | None:
            """:meth:`_value_edit`, keeping aside a value the USD authors without its schema (PhysX ignores it)."""
            edit = self._value_edit(prim, attribute, value, schema=schema)
            if edit is None and schema and schema not in prim.GetAppliedSchemas():
                if self._stated(prim, attribute) is not None:
                    path = str(prim.GetPath())
                    skipped[path, schema].append(UsdEdit(path, attribute, value, schema=schema))
            return edit

        readings, base = self._read_assets(stage)
        changes = {
            key: {name for name, value in values.items() if not self._close(base[key][name], value)}
            for key, values in readings.items()
        }
        # PhysX ignores an attribute whose API schema the prim lacks; applying the schema for an edit enables it.
        activated = defaultdict(set)
        for (path, axis), changed in changes.items():
            if axis is not None:
                prim, table = stage.GetPrimAtPath(path), self._joint_table(axis)
                changed |= self._not_imported(prim, axis)
                applied = prim.GetAppliedSchemas()
                for name in changed & self._written(prim, axis, table):
                    schema = table[name].schema and table[name].schema.format(axis=axis)
                    if schema and schema not in applied:
                        activated[path].add(schema)
        for (path, axis), values in readings.items():
            prim = stage.GetPrimAtPath(path)
            changed = changes[path, axis]
            if axis is not None:
                table = self._joint_table(axis)
                written = self._written(prim, axis, table)
                changed |= {
                    name
                    for name, (attribute, schema, _) in table.items()
                    if schema
                    and schema.format(axis=axis) in activated[path]
                    and self._stated(prim, attribute.format(axis=axis)) is not None
                }
                for name, (attribute, schema, angular_scale) in table.items():
                    if name not in written:
                        continue
                    scale = 1.0 if axis == "linear" else angular_scale
                    schema = schema and schema.format(axis=axis)
                    if name in changed:
                        edits.append(value_edit(prim, attribute.format(axis=axis), float(values[name]) * scale, schema))
                continue
            edits += self._mass_edits(prim, float(values["mass"]), values["com"], values["inertia"], changed)
            colliders = self._colliders(stage, path)
            # The engine reports shapes per body, without their prim identities.
            uniform = {name: self._uniform(values[name], path, name) for name in COLLIDER_ATTRIBUTES if name in values}
            for name in changed & {*uniform, "material"}:
                if len(values[name]) and self._uniform(values[name], path, name) is None:
                    raise NotImplementedError(
                        f"The {name} of body '{path}' changed and differs between its shapes, which the engine"
                        " reports without their prims."
                    )
            written = {name for name, value in uniform.items() if name in changed and value is not None}
            for collider in map(stage.GetPrimAtPath, colliders):
                names = set(written)
                if written and "PhysxCollisionAPI" not in collider.GetAppliedSchemas():
                    # The edits apply the schema, which enables the collider's other authored offsets.
                    names |= {
                        name
                        for name, attribute in COLLIDER_ATTRIBUTES.items()
                        if uniform.get(name) is not None and self._stated(collider, attribute) is not None
                    }
                edits += [
                    value_edit(collider, COLLIDER_ATTRIBUTES[name], uniform[name], "PhysxCollisionAPI")
                    for name in sorted(names)
                ]
            material = self._uniform(values["material"], path, "material") if "material" in changed else None
            if material is not None:
                materials.update(
                    {
                        collider: dict(zip(USD_PHYSICS_MATERIAL_ATTRIBUTES.values(), material.tolist()))
                        for collider in colliders
                    }
                )
        edits = [edit for edit in edits if edit is not None]
        # A value authored without its schema takes effect only if an edit applies the schema; otherwise state it.
        applied = {(edit.path, edit.schema) for edit in edits}
        edits += [edit for key, group in skipped.items() if key not in applied for edit in group]
        edits += self._material_edits(stage, materials)
        return edits + self._gravity_edits(stage, self.scene.sim.backend_manager.get_gravity())

    def _joint_table(self, axis: str) -> dict[str, JointAttribute]:
        """The USD attributes of a joint axis's properties, as the backend's import reads them."""
        return D6_JOINT_ATTRIBUTES if axis in D6_AXES else JOINT_ATTRIBUTES

    def _not_imported(self, prim: Usd.Prim, axis: str) -> set[str]:
        """Joint properties the engine loaded from the stage that an import of it would not; none by default."""
        return set()

    @staticmethod
    def _written(prim: Usd.Prim, axis: str, table: dict) -> set[str]:
        """Joint properties the export may write on an axis.

        PhysX drives only an axis whose USD states a drive; gains on another axis have no effect.
        """
        driven = prim.HasAPI(UsdPhysics.DriveAPI, axis)
        return {name for name, (attribute, _, _) in table.items() if driven or not attribute.startswith("drive:")}

    def read_engine_base(self) -> dict[str, list[dict[str, np.ndarray]]]:
        """Read what the engine loaded for each scene asset, in every environment.

        Returns:
            Per asset name, the engine values of :meth:`_read_engine` with the limits and friction split as in
            :data:`Readings` and ``com`` the COM position; a rigid object collection holds one entry per object.
        """
        base = {name: [self._read_object(a.cfg.prim_path, a.cfg)] for name, a in self.scene.articulations.items()}
        base |= {name: [self._read_object(obj.cfg.prim_path)] for name, obj in self.scene.rigid_objects.items()}
        for name, collection in self.scene.rigid_object_collections.items():
            base[name] = [self._read_object(cfg.prim_path) for cfg in collection.cfg.rigid_objects.values()]
        return base

    @abstractmethod
    def _read_engine(self, path_expr: str, articulation: bool) -> dict[str, np.ndarray]:
        """Read the engine values of the objects matching a prim path expression, one row per environment.

        Args:
            path_expr: Prim path expression of the articulation roots or rigid bodies, as an asset resolves it.
            articulation: Whether the objects are articulations.

        Returns:
            Raw values: of an articulation, per DOF ``stiffness``, ``damping``, ``max_force``, ``armature``,
            ``max_velocity``, ``limits`` (lower, upper) and ``friction`` (static, dynamic, viscous); per body
            ``mass``, ``com_pose`` and row-major ``inertia``. Of a rigid body, the same body values and per shape
            the ``material`` (static friction, dynamic friction, restitution), ``contact_offset`` and
            ``rest_offset``.
        """
        raise NotImplementedError

    """
    Engine values.
    """

    def _read_object(self, prim_path: str, articulation_cfg=None) -> dict[str, np.ndarray]:
        """Engine values of an asset's articulation or rigid body, resolved as the asset resolves it."""
        if getattr(articulation_cfg, "articulation_root_prim_path", None) is not None:
            path_expr = prim_path + articulation_cfg.articulation_root_prim_path
        else:
            api = UsdPhysics.ArticulationRootAPI if articulation_cfg is not None else UsdPhysics.RigidBodyAPI
            matches = resolve_matching_prims_from_source(
                prim_path, predicate=lambda prim: prim.HasAPI(api), expected_num_matches=1
            )
            path_expr = matches[0][1]
        values = self._read_engine(path_expr, articulation_cfg is not None)
        values["com"] = values.pop("com_pose")[..., :3]
        if articulation_cfg is not None:
            limits, friction = values.pop("limits"), values.pop("friction")
            values.update(lower_limit=limits[..., 0], upper_limit=limits[..., 1])
            values.update(static_friction=friction[..., 0], dynamic_friction=friction[..., 1])
            values.update(viscous_friction=friction[..., 2])
        return values

    def _read_assets(self, stage: Usd.Stage) -> tuple[Readings, Readings]:
        """Read the live values of the scene's assets and their base, keyed by their prims in the exported stage."""
        base = self._engine_base()
        readings, bases = {}, {}
        for name, articulation in self.scene.articulations.items():
            for key, values in self._read_articulation(stage, articulation, base[name][0]):
                readings[key], bases[key] = values
        objects = [(obj.cfg, obj.data, 0, base[name][0]) for name, obj in self.scene.rigid_objects.items()]
        for name, collection in self.scene.rigid_object_collections.items():
            # Collection bodies are ordered as the configured objects.
            for index, cfg in enumerate(collection.cfg.rigid_objects.values()):
                objects.append((cfg, collection.data, index, base[name][index]))
        for cfg, data, index, engine in objects:
            key = self._rigid_body_path(stage, cfg.prim_path), None
            live = self._read_object(cfg.prim_path)
            colliders = {name: self._env_row(live[name]) for name in COLLIDER_READINGS}
            readings[key] = {**self._body_values(data, index), **colliders}
            bases[key] = {
                **self._body_base(engine, 0),
                **{name: self._env_row(engine[name]) for name in COLLIDER_READINGS},
            }
        exported = {path for path, _ in readings}
        missing = [
            str(prim.GetPath())
            for prim in stage.Traverse(Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and str(prim.GetPath()) not in exported
        ]
        if missing:
            # Isaac Lab configures only its assets, so these bodies hold what their USD states.
            logger.info(f"Physics bodies outside the scene's assets are exported as authored: {missing}.")
        return readings, bases

    def _read_articulation(self, stage: Usd.Stage, articulation, engine: dict[str, np.ndarray]):
        """Yield each joint DOF's and link's prim key with its live and base values, matching names to prims."""
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
        # The engine orders DOFs and bodies as the backend does; the asset data may reorder them.
        engine_dof = {name: index for index, name in enumerate(articulation.backend_joint_names)}
        engine_body = {name: index for index, name in enumerate(articulation.backend_body_names)}
        for index, name in enumerate(articulation.joint_names):
            path, axis = self._joint_axis(stage, joints, name, scope)
            live = {prop: values[index] for prop, values in dofs.items()}
            loaded = {prop: self._env_row(engine[prop])[engine_dof[name]] for prop in dofs}
            if axis is None:
                if any(not self._close(loaded[prop], value) for prop, value in live.items()):
                    raise NotImplementedError(
                        f"Joint '{path}' changed; only revolute, prismatic and rotational D6 joints are exported."
                    )
                continue
            yield (path, axis), (live, loaded)
        for index, name in enumerate(articulation.body_names):
            key = self._unique(links, name, scope), None
            yield key, (self._body_values(data, index), self._body_base(engine, engine_body[name]))

    def _body_values(self, data, index: int) -> dict[str, np.ndarray]:
        """Mass properties of one body of an asset in the selected environment."""
        return {
            "mass": float(self._row(data.body_mass)[index]),
            "com": self._row(data.body_com_pose_b)[index, :3],
            "inertia": self._row(data.body_inertia)[index],
        }

    def _body_base(self, engine: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
        """Mass properties of one body as the engine loaded them, in the selected environment."""
        return {
            "mass": float(np.ravel(self._env_row(engine["mass"]))[index]),
            "com": np.reshape(self._env_row(engine["com"]), (-1, 3))[index],
            "inertia": np.reshape(self._env_row(engine["inertia"]), (-1, 9))[index],
        }

    def _env_row(self, rows):
        """The selected environment's row; an asset outside the environments, such as a shared one, has one row."""
        if len(rows) == 1:
            return rows[0]
        if len(rows) != self.scene.num_envs:
            raise NotImplementedError(f"An asset holds {len(rows)} instances for {self.scene.num_envs} environments.")
        return rows[self.env_id]

    def _row(self, values) -> np.ndarray:
        """The selected environment's entry of an asset data array."""
        return self._env_row(values.torch).detach().cpu().numpy()

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

    @classmethod
    def _joint_axis(
        cls, stage: Usd.Stage, joints: dict[str, list[str]], name: str, scope: Usd.Prim
    ) -> tuple[str, str | None]:
        """The prim and USD axis of the joint DOF an asset data name refers to; the axis is None if not exported.

        PhysX names the DOFs of a multi-axis joint ``<joint>:<n>``, ``n`` indexing :data:`D6_AXES`.
        """
        joint, _, suffix = name.rpartition(":")
        numbered = bool(joint) and suffix.isdigit()
        path = cls._unique(joints, joint if numbered else name, scope)
        prim = stage.GetPrimAtPath(path)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            return path, "angular"
        if prim.IsA(UsdPhysics.PrismaticJoint):
            return path, "linear"
        if prim.GetTypeName() == "PhysicsJoint":
            free = [axis for axis in D6_AXES if not cls._locked(prim, axis)]
            if all(axis.startswith("rot") for axis in free):
                if numbered and int(suffix) < 3 and D6_AXES[int(suffix)] in free:
                    return path, D6_AXES[int(suffix)]
                if not numbered and len(free) == 1:
                    return path, free[0]
        return path, None

    @staticmethod
    def _locked(joint: Usd.Prim, axis: str) -> bool:
        """Whether a D6 joint's USD locks an axis: its limit's low end lies above its high end."""
        if not joint.HasAPI(UsdPhysics.LimitAPI, axis):
            return False
        limit = UsdPhysics.LimitAPI(joint, axis)
        low, high = limit.GetLowAttr().Get(), limit.GetHighAttr().Get()
        return low is not None and high is not None and low > high

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
    def _uniform(values: np.ndarray, path: str, name: str) -> np.ndarray | None:
        """The one value of all shapes of a body, or None if its shapes differ.

        Shapes differ when the USD leaves the value to the engine, which derives it per collider, or
        states it per collider; the USD then reproduces it.
        """
        values = np.asarray(values)
        if len(values) == 0:  # a body without colliders
            return None
        if not np.all(values == values[0]):
            logger.debug(f"The {name} of body '{path}' differs between its shapes; keeping its USD values.")
            return None
        return values[0]
