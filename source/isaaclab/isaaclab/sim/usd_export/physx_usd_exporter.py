# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD export in the PhysX dialect, shared by the PhysX-based backends."""

from __future__ import annotations

import contextlib
import logging
import math
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from pxr import Usd, UsdGeom, UsdPhysics, Vt

from ...assets.articulation.articulation_data import ArticulationData
from ...assets.rigid_object.rigid_object_data import RigidObjectData
from ..utils.queries import find_matching_prim_paths, resolve_matching_prims_from_source
from .base_usd_exporter import (
    JOINT_DATA_FIELDS,
    JOINT_STATE_ATTRIBUTES,
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

Joint entries hold the properties of :data:`JOINT_ATTRIBUTES`, named as the asset data names them. Body entries hold
``body_mass``, ``body_com_pos_b`` [m] and ``body_inertia`` [kg*m^2, row-major 3x3 about the COM in body axes];
rigid bodies outside articulations
also hold the per-shape ``contact_offset`` and ``rest_offset`` [m] and ``material`` (static friction,
dynamic friction, restitution).
"""

JOINT_ATTRIBUTES = {
    **USD_PHYSICS_JOINT_ATTRIBUTES,
    # Per axis: OVPhysX 0.7's PhysX schema has no PhysxJointAPI, and an axis value overrides it in Isaac Sim.
    "joint_armature": JointAttribute("physxJointAxis:{axis}:armature", "PhysxJointAxisAPI:{axis}", 1.0),
    "joint_vel_limits": JointAttribute(
        "physxJointAxis:{axis}:maxJointVelocity", "PhysxJointAxisAPI:{axis}", math.degrees(1.0)
    ),
    "joint_friction_coeff": JointAttribute(
        "physxJointAxis:{axis}:staticFrictionEffort", "PhysxJointAxisAPI:{axis}", 1.0
    ),
    "joint_dynamic_friction_coeff": JointAttribute(
        "physxJointAxis:{axis}:dynamicFrictionEffort", "PhysxJointAxisAPI:{axis}", 1.0
    ),
    "joint_viscous_friction_coeff": JointAttribute(
        "physxJointAxis:{axis}:viscousFrictionCoefficient", "PhysxJointAxisAPI:{axis}", math.radians(1.0)
    ),
    **JOINT_STATE_ATTRIBUTES,
}
"""Joint property, named as the asset data names it, -> USD attribute."""

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

DEFORMABLE_SIM_APIS = {"volume": "OmniPhysicsVolumeDeformableSimAPI", "surface": "OmniPhysicsSurfaceDeformableSimAPI"}
"""Deformable kind -> API schema of its simulation mesh, whose points are its nodes."""


class PhysxUsdExporter(BaseUsdExporter):
    """Base of the PhysX-based backends: writes their engine values in the PhysX USD dialect.

    The base is what the engine loaded from the stage, read when the physics is ready and before any asset
    writes its configuration; a PhysX import of the stage loads the same. A value is written where the live
    engine differs from it, so Isaac Lab's configuration is stated, and nothing the engine derives by itself is.

    Live joint and body values come from the scene assets' data, the base from the same asset data bound to a new
    view of the engine. A backend opens that view and reads the collider values of rigid bodies, which the asset
    data does not hold.
    """

    """
    Operations.
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
            """:meth:`_compute_value_edit`, keeping aside a value authored without its schema, which PhysX ignores."""
            edit = self._compute_value_edit(prim, attribute, value, schema=schema)
            if edit is None and schema and schema not in prim.GetAppliedSchemas():
                if self._get_authored_value(prim, attribute) is not None:
                    path = str(prim.GetPath())
                    skipped[path, schema].append(UsdEdit(path, attribute, value, schema=schema))
            return edit

        # 1. Live values and the values the engine loaded from the stage, keyed by prim and joint axis.
        readings, base = self._read_assets(stage)
        # 2. Properties whose live value differs from the loaded one.
        numerical_changes = {
            key: {name for name, value in values.items() if not self._is_close(base[key][name], value)}
            for key, values in readings.items()
        }
        # 3. Plus the joint properties an import of the stage alone cannot restore, though their values are equal.
        required = {}
        for (path, axis), names in numerical_changes.items():
            restate = self._get_properties_to_restate(stage.GetPrimAtPath(path), axis) if axis is not None else set()
            required[path, axis] = names | restate
        # 4. API schemas the joint edits newly apply; PhysX ignores an attribute whose schema the prim lacks.
        activated = defaultdict(set)
        for (path, axis), names in required.items():
            if axis is not None:
                prim, table = stage.GetPrimAtPath(path), self._get_joint_attribute_table(axis)
                applied = prim.GetAppliedSchemas()
                for name in names & self._get_writable_joint_properties(prim, axis, table):
                    schema = table[name].schema and table[name].schema.format(axis=axis)
                    if schema and schema not in applied:
                        activated[path].add(schema)
        # 5. Edits per prim: joint properties, or body mass, collider offsets and material values.
        for (path, axis), values in readings.items():
            prim = stage.GetPrimAtPath(path)
            export_names = set(required[path, axis])
            if axis is not None:
                # Also restate the authored attributes a newly applied schema turns on.
                table = self._get_joint_attribute_table(axis)
                written = self._get_writable_joint_properties(prim, axis, table)
                export_names |= {
                    name
                    for name, (attribute, schema, _) in table.items()
                    if schema
                    and schema.format(axis=axis) in activated[path]
                    and self._get_authored_value(prim, attribute.format(axis=axis)) is not None
                }
                for name, (attribute, schema, angular_scale) in table.items():
                    if name not in written:
                        continue
                    scale = 1.0 if axis == "linear" else angular_scale
                    schema = schema and schema.format(axis=axis)
                    if name in export_names:
                        edits.append(value_edit(prim, attribute.format(axis=axis), float(values[name]) * scale, schema))
                continue
            edits += self._compute_mass_edits(
                prim, float(values["body_mass"]), values["body_com_pos_b"], values["body_inertia"], export_names
            )
            colliders = self._find_owned_colliders(stage, path)
            # The engine reports shapes per body, without their prim identities.
            uniform = {
                name: self._get_uniform_shape_value(values[name], path, name)
                for name in COLLIDER_ATTRIBUTES
                if name in values
            }
            for name in export_names & {*uniform, "material"}:
                if len(values[name]) and self._get_uniform_shape_value(values[name], path, name) is None:
                    raise NotImplementedError(
                        f"The {name} of body '{path}' changed and differs between its shapes, which the engine"
                        " reports without their prims."
                    )
            written = {name for name, value in uniform.items() if name in export_names and value is not None}
            for collider in map(stage.GetPrimAtPath, colliders):
                names = set(written)
                if written and "PhysxCollisionAPI" not in collider.GetAppliedSchemas():
                    # The edits apply the schema, which enables the collider's other authored offsets.
                    names |= {
                        name
                        for name, attribute in COLLIDER_ATTRIBUTES.items()
                        if uniform.get(name) is not None and self._get_authored_value(collider, attribute) is not None
                    }
                edits += [
                    value_edit(collider, COLLIDER_ATTRIBUTES[name], uniform[name], "PhysxCollisionAPI")
                    for name in sorted(names)
                ]
            material = (
                self._get_uniform_shape_value(values["material"], path, "material")
                if "material" in export_names
                else None
            )
            if material is not None:
                materials.update(
                    {
                        collider: dict(zip(USD_PHYSICS_MATERIAL_ATTRIBUTES.values(), material.tolist()))
                        for collider in colliders
                    }
                )
        # 6. A value authored without its schema takes effect only if an edit applies the schema; otherwise state it.
        edits = [edit for edit in edits if edit is not None]
        applied = {(edit.path, edit.schema) for edit in edits}
        edits += [edit for key, group in skipped.items() if key not in applied for edit in group]
        # 7. Materials across the colliders sharing them, deformable nodal states and gravity.
        edits += self._compute_material_edits(stage, materials)
        edits += self._compute_deformable_edits(stage)
        return edits + self._compute_gravity_edits(stage, self.scene.sim.backend_manager.get_gravity())

    def read_engine_base(self) -> dict[str, list[dict[str, np.ndarray]]]:
        """Read what the engine loaded for each scene asset, in every environment.

        Returns:
            Per asset name, the values of :meth:`_read_engine_values` in every environment; a rigid object
            collection holds one entry per object.
        """
        base = {
            name: [self._read_engine_values(a.cfg.prim_path, a.cfg)] for name, a in self.scene.articulations.items()
        }
        base |= {name: [self._read_engine_values(obj.cfg.prim_path)] for name, obj in self.scene.rigid_objects.items()}
        for name, collection in self.scene.rigid_object_collections.items():
            base[name] = [self._read_engine_values(cfg.prim_path) for cfg in collection.cfg.rigid_objects.values()]
        return base

    @abstractmethod
    def _open_engine_view(self, path_expr: str, articulation: bool) -> contextlib.AbstractContextManager:
        """Open the backend's view of the objects matching a prim path expression, one per environment.

        The view is the one the backend's asset data reads, and stays open for the ``with`` block.

        Args:
            path_expr: Prim path expression of the articulation roots or rigid bodies, as an asset resolves it.
            articulation: Whether the objects are articulations.
        """
        raise NotImplementedError

    @abstractmethod
    def _read_collider_values(self, view) -> dict[str, np.ndarray]:
        """Read per environment and shape the ``material`` (static friction, dynamic friction, restitution),
        ``contact_offset`` and ``rest_offset`` of a rigid body view, which no asset data holds."""
        raise NotImplementedError

    """
    Engine values.
    """

    def _read_assets(self, stage: Usd.Stage) -> tuple[Readings, Readings]:
        """Read the live values of the scene's assets and their base, keyed by their prims in the exported stage."""
        base = self._get_engine_base()
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
            key = self._find_rigid_body_path(stage, cfg.prim_path), None
            with self._open_engine_view(self._resolve_engine_path(cfg.prim_path), False) as view:
                live = self._read_collider_values(view)
            colliders = {name: self._select_env_row(live[name]) for name in COLLIDER_READINGS}
            readings[key] = {**self._select_mass_properties(self._read_body_data(data), index), **colliders}
            bases[key] = {
                **self._select_mass_properties(engine, 0),
                **{name: self._select_env_row(engine[name]) for name in COLLIDER_READINGS},
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
        # 1. Live joint and link values from the asset data, and the reset state.
        scope = stage.GetPrimAtPath(self._resolve_asset_path(stage, articulation.cfg.prim_path))
        joints = self._index_prims_by_name(scope, lambda prim: prim.IsA(UsdPhysics.Joint))
        links = self._index_prims_by_name(scope, lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI))
        data = articulation.data
        dofs = {name: self._select_env_row(rows) for name, rows in self._read_joint_data(data).items()}
        bodies = self._read_body_data(data)
        # The joint state Isaac Lab resets the articulation to; the engine holds it only from the first reset on.
        initial = {
            "default_joint_pos": self._select_data_env_row(data.default_joint_pos),
            "default_joint_vel": self._select_data_env_row(data.default_joint_vel),
        }
        # 2. Per joint DOF: its USD prim and axis, the engine base and the USD initial state.
        # The engine orders DOFs and bodies as the backend does; the asset data may reorder them.
        engine_dof = {name: index for index, name in enumerate(articulation.backend_joint_names)}
        engine_body = {name: index for index, name in enumerate(articulation.backend_body_names)}
        unwritten = []
        for index, name in enumerate(articulation.joint_names):
            path, axis = self._resolve_joint_axis(stage, joints, name, scope)
            live = {prop: values[index] for prop, values in dofs.items()}
            loaded = {prop: self._select_env_row(engine[prop])[engine_dof[name]] for prop in dofs}
            state = {prop: values[index] for prop, values in initial.items()}
            if axis is None:
                if any(not self._is_close(loaded[prop], value) for prop, value in live.items()):
                    raise NotImplementedError(
                        f"Joint '{path}' changed; only revolute, prismatic and rotational D6 joints are exported."
                    )
                if any(not self._is_close(value, 0.0) for value in state.values()):
                    unwritten.append(path)
                continue
            yield (path, axis), (live | state, loaded | self._get_imported_joint_state(stage.GetPrimAtPath(path), axis))
        if unwritten:
            logger.warning(
                "The initial state of joints other than revolute, prismatic and rotational D6 ones is not written:"
                f" {sorted(set(unwritten))}."
            )
        # 3. Per link: its body prim with live and base mass properties.
        for index, name in enumerate(articulation.body_names):
            key = self._get_unique_prim_path(links, name, scope), None
            yield (
                key,
                (self._select_mass_properties(bodies, index), self._select_mass_properties(engine, engine_body[name])),
            )

    def _select_data_env_row(self, values) -> np.ndarray:
        """The selected environment's entry of an asset data array."""
        return self._select_env_row(values.torch).detach().cpu().numpy()

    def _select_env_row(self, rows):
        """The selected environment's row; an asset outside the environments, such as a shared one, has one row."""
        if len(rows) == 1:
            return rows[0]
        if len(rows) != self.scene.num_envs:
            raise NotImplementedError(f"An asset holds {len(rows)} instances for {self.scene.num_envs} environments.")
        return rows[self.env_id]

    @classmethod
    def _get_imported_joint_state(cls, prim: Usd.Prim, axis: str) -> dict[str, float]:
        """The state [m or rad] an import starts a joint axis at: the USD's, which PhysX reads only with its schema."""
        scale = 1.0 if axis == "linear" else math.radians(1.0)
        applied = prim.GetAppliedSchemas()
        state = {}
        for name, (attribute, schema, _) in JOINT_STATE_ATTRIBUTES.items():
            value = cls._get_authored_value(prim, attribute.format(axis=axis))
            state[name] = float(value) * scale if value is not None and schema.format(axis=axis) in applied else 0.0
        return state

    def _select_mass_properties(self, bodies: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
        """Mass properties of one body in the selected environment, from :meth:`_read_body_data` of every one."""
        return {
            "body_mass": float(np.ravel(self._select_env_row(bodies["body_mass"]))[index]),
            "body_com_pos_b": np.reshape(self._select_env_row(bodies["body_com_pos_b"]), (-1, 3))[index],
            "body_inertia": np.reshape(self._select_env_row(bodies["body_inertia"]), (-1, 9))[index],
        }

    def _read_engine_values(self, prim_path: str, articulation_cfg=None) -> dict[str, np.ndarray]:
        """Read an asset's engine values in every environment through asset data bound to a new view.

        Args:
            prim_path: The asset's prim path expression.
            articulation_cfg: The asset's articulation configuration, or None for a rigid body.

        Returns:
            Body mass properties, and an articulation's DOF properties or a rigid body's collider values.
        """
        articulation = articulation_cfg is not None
        with self._open_engine_view(self._resolve_engine_path(prim_path, articulation_cfg), articulation) as view:
            data_type = ArticulationData if articulation else RigidObjectData
            # The data only reads the view; the view outlives it.
            data = data_type(view, self.scene.sim.device, backend_manager=self.scene.sim.backend_manager)
            values = self._read_body_data(data)
            values |= self._read_joint_data(data) if articulation else self._read_collider_values(view)
        return values

    @staticmethod
    def _resolve_engine_path(prim_path: str, articulation_cfg=None) -> str:
        """The prim path expression of an asset's articulation root or rigid body, as the asset resolves it."""
        if getattr(articulation_cfg, "articulation_root_prim_path", None) is not None:
            return prim_path + articulation_cfg.articulation_root_prim_path
        api = UsdPhysics.ArticulationRootAPI if articulation_cfg is not None else UsdPhysics.RigidBodyAPI
        matches = resolve_matching_prims_from_source(
            prim_path, predicate=lambda prim: prim.HasAPI(api), expected_num_matches=1
        )
        return matches[0][1]

    @staticmethod
    def _read_body_data(data) -> dict[str, np.ndarray]:
        """Read the mass properties of every body of asset data, in every environment."""
        values = {
            "body_mass": data.body_mass.torch,
            "body_com_pos_b": data.body_com_pose_b.torch[..., :3],  # reads faster than body_com_pos_b
            "body_inertia": data.body_inertia.torch,
        }
        return {name: value.detach().cpu().numpy().copy() for name, value in values.items()}

    @staticmethod
    def _read_joint_data(data) -> dict[str, np.ndarray]:
        """Read the DOF properties of :data:`JOINT_ATTRIBUTES` from articulation data, in every environment."""
        values = {name: getattr(data, name).torch for name in JOINT_DATA_FIELDS}
        limits = data.joint_pos_limits.torch
        values |= {"joint_pos_limits_lower": limits[..., 0], "joint_pos_limits_upper": limits[..., 1]}
        return {name: value.detach().cpu().numpy().copy() for name, value in values.items()}

    """
    Edits.
    """

    def _get_properties_to_restate(self, prim: Usd.Prim, axis: str) -> set[str]:
        """Joint properties the engine loaded from the stage that an import of it would not; none by default."""
        return set()

    def _get_joint_attribute_table(self, axis: str) -> dict[str, JointAttribute]:
        """The USD attributes of a joint axis's properties, as the backend's import reads them."""
        return D6_JOINT_ATTRIBUTES if axis in D6_AXES else JOINT_ATTRIBUTES

    @staticmethod
    def _get_writable_joint_properties(prim: Usd.Prim, axis: str, table: dict) -> set[str]:
        """Joint properties the export may write on an axis.

        PhysX drives only an axis whose USD states a drive; gains on another axis have no effect.
        """
        driven = prim.HasAPI(UsdPhysics.DriveAPI, axis)
        return {name for name, (attribute, _, _) in table.items() if driven or not attribute.startswith("drive:")}

    def _compute_deformable_edits(self, stage: Usd.Stage) -> list[UsdEdit]:
        """Write the nodal state Isaac Lab resets each volume deformable to on its simulation mesh.

        The engine starts a deformable at its simulation mesh's points and velocities; Isaac Lab resets it to the
        nodal positions the engine holds after its warm-up step, at rest. PhysX diverges from the authored points
        of a surface deformable, so its state is not written.
        """
        edits, unwritten = [], []
        for deformable in self.scene.deformable_objects.values():
            # 1. The reset nodal state of the deformable in the exported environment.
            root = stage.GetPrimAtPath(self._resolve_asset_path(stage, deformable.cfg.prim_path))
            state = self._select_data_env_row(deformable.data.default_nodal_state_w)
            # 2. Its simulation meshes, by their OmniPhysics volume or surface API.
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                kind = next((k for k, api in DEFORMABLE_SIM_APIS.items() if api in prim.GetAppliedSchemas()), None)
                if kind is None:
                    continue
                # 3. The reset state in mesh coordinates against the authored points and velocities.
                mesh = UsdGeom.PointBased(prim)
                points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
                velocities = self._get_authored_value(prim, "velocities")
                velocities = np.zeros_like(points) if velocities is None else np.asarray(velocities, dtype=float)
                # The engine holds the nodes in world coordinates, the mesh in its own.
                to_local = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim).GetInverse())
                reset = {
                    "points": state[: len(points), :3] @ to_local[:3, :3] + to_local[3, :3],
                    "velocities": state[: len(points), 3:] @ to_local[:3, :3],
                }
                usd = {"points": points, "velocities": velocities}
                changed = [name for name, value in reset.items() if not self._is_close(usd[name], value)]
                if kind == "surface":
                    unwritten += [str(prim.GetPath())] if changed else []
                    continue
                for name in changed:
                    value = Vt.Vec3fArray.FromNumpy(reset[name].astype(np.float32))
                    edits.append(UsdEdit(str(prim.GetPath()), name, value, "vector3f[]"))
        if unwritten:
            logger.warning(f"The initial nodal state of surface deformables is not written: {unwritten}.")
        return edits

    """
    Exported stage.
    """

    @staticmethod
    def _resolve_asset_path(stage: Usd.Stage, prim_path: str) -> str:
        """The prim of an asset in the exported stage, which keeps one environment."""
        paths = find_matching_prim_paths(prim_path, stage)
        if len(paths) != 1:
            raise RuntimeError(f"Expected one prim matching '{prim_path}' in the exported stage, found {paths}.")
        return paths[0]

    @staticmethod
    def _index_prims_by_name(scope: Usd.Prim, predicate: Callable[[Usd.Prim], bool]) -> dict[str, list[str]]:
        """Paths of the prims below ``scope`` that satisfy ``predicate``, grouped by prim name."""
        named = defaultdict(list)
        for prim in Usd.PrimRange(scope, Usd.TraverseInstanceProxies()):
            if predicate(prim):
                named[prim.GetName()].append(str(prim.GetPath()))
        return named

    @classmethod
    def _resolve_joint_axis(
        cls, stage: Usd.Stage, joints: dict[str, list[str]], name: str, scope: Usd.Prim
    ) -> tuple[str, str | None]:
        """The prim and USD axis of the joint DOF an asset data name refers to; the axis is None if not exported.

        PhysX names the DOFs of a multi-axis joint ``<joint>:<n>``, ``n`` indexing :data:`D6_AXES`.
        """
        joint, _, suffix = name.rpartition(":")
        numbered = bool(joint) and suffix.isdigit()
        path = cls._get_unique_prim_path(joints, joint if numbered else name, scope)
        prim = stage.GetPrimAtPath(path)
        # 1. A revolute or prismatic joint has one axis.
        if prim.IsA(UsdPhysics.RevoluteJoint):
            return path, "angular"
        if prim.IsA(UsdPhysics.PrismaticJoint):
            return path, "linear"
        # 2. A generic joint is exported as a rotational D6: every axis it leaves free must rotate.
        if prim.GetTypeName() == "PhysicsJoint":
            free = [axis for axis in D6_AXES if not cls._is_axis_locked(prim, axis)]
            if all(axis.startswith("rot") for axis in free):
                if numbered and int(suffix) < 3 and D6_AXES[int(suffix)] in free:
                    return path, D6_AXES[int(suffix)]
                if not numbered and len(free) == 1:
                    return path, free[0]
        return path, None

    @staticmethod
    def _get_unique_prim_path(named: dict[str, list[str]], name: str, scope: Usd.Prim) -> str:
        """The one prim an asset data name refers to; the data names prims without their paths."""
        paths = named.get(name, [])
        if len(paths) != 1:
            raise RuntimeError(f"Expected one prim named '{name}' below '{scope.GetPath()}', found {paths}.")
        return paths[0]

    @staticmethod
    def _is_axis_locked(joint: Usd.Prim, axis: str) -> bool:
        """Whether a D6 joint's USD locks an axis: its limit's low end lies above its high end."""
        if not joint.HasAPI(UsdPhysics.LimitAPI, axis):
            return False
        limit = UsdPhysics.LimitAPI(joint, axis)
        low, high = limit.GetLowAttr().Get(), limit.GetHighAttr().Get()
        return low is not None and high is not None and low > high

    @classmethod
    def _find_rigid_body_path(cls, stage: Usd.Stage, prim_path: str) -> str:
        """The rigid body of a rigid object: its prim, or the rigid body below it."""
        root = stage.GetPrimAtPath(cls._resolve_asset_path(stage, prim_path))
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(prim.GetPath())
        raise RuntimeError(f"No rigid body below '{root.GetPath()}'.")

    @staticmethod
    def _find_owned_colliders(stage: Usd.Stage, body: str) -> list[str]:
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
    def _get_uniform_shape_value(values: np.ndarray, path: str, name: str) -> np.ndarray | None:
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
