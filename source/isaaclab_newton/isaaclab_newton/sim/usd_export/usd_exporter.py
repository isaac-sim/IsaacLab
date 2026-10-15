# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton USD export: every live model value the exported stage does not already state."""

from __future__ import annotations

import json
import logging
import math

import newton
import numpy as np
from newton.usd import PrimType, SchemaResolverNewton

from pxr import Usd, UsdPhysics, UsdShade

from isaaclab.sim.usd_export.base_usd_exporter import BaseUsdExporter, UsdEdit
from isaaclab.utils.string import string_to_callable

from isaaclab_newton.physics import NewtonManager

logger = logging.getLogger(__name__)

SETTINGS_KEY = "isaaclab:newton"
"""Layer metadata holding what Newton's importer does not read from USD: the step size and the solver."""

_JOINT = SchemaResolverNewton.mapping[PrimType.JOINT]
_SHAPE = SchemaResolverNewton.mapping[PrimType.SHAPE]
_MATERIAL = SchemaResolverNewton.mapping[PrimType.MATERIAL]
_DEGREES = 180.0 / math.pi
_PER_DEGREE = math.pi / 180.0

DOF_ATTRIBUTES = {
    "joint_target_ke": ("drive:{axis}:physics:stiffness", "PhysicsDriveAPI:{axis}", _PER_DEGREE),
    "joint_target_kd": ("drive:{axis}:physics:damping", "PhysicsDriveAPI:{axis}", _PER_DEGREE),
    "joint_effort_limit": ("drive:{axis}:physics:maxForce", "PhysicsDriveAPI:{axis}", 1.0),
    "joint_limit_lower": ("physics:lowerLimit", None, _DEGREES),
    "joint_limit_upper": ("physics:upperLimit", None, _DEGREES),
    "joint_armature": (_JOINT["armature"].name, None, 1.0),
    "joint_friction": (_JOINT["friction"].name, None, 1.0),
    "joint_damping": (_JOINT["damping"].name, None, _PER_DEGREE),
    "joint_velocity_limit": (_JOINT["velocity_limit"].name, None, _DEGREES),
    "joint_limit_ke": (_JOINT["limit_ke"].name, None, _PER_DEGREE),
    "joint_limit_kd": (_JOINT["limit_kd"].name, None, _PER_DEGREE),
}
"""Model DOF array -> USD joint attribute, required API schema, and scale to USD units for angular DOFs."""

SHAPE_ATTRIBUTES = {
    "shape_margin": _SHAPE["margin"].name,
    "shape_gap": _SHAPE["gap"].name,
}
"""Model shape array -> USD attribute on the collider."""

MATERIAL_ATTRIBUTES = {
    "shape_material_mu": "physics:dynamicFriction",
    "shape_material_restitution": "physics:restitution",
    "shape_material_mu_torsional": _MATERIAL["mu_torsional"].name,
    "shape_material_mu_rolling": _MATERIAL["mu_rolling"].name,
    "shape_material_ke": _MATERIAL["ke"].name,
    "shape_material_kd": _MATERIAL["kd"].name,
    "shape_material_kf": _MATERIAL["kf"].name,
    "shape_material_ka": _MATERIAL["ka"].name,
}
"""Model shape array -> USD attribute on the collider's physics material, which takes precedence."""

SHAPE_CONTACT_ATTRIBUTES = {_MATERIAL[key].name: _SHAPE[key].name for key in ("ke", "kd", "kf", "ka")}
"""Material contact gain -> collider attribute, used by colliders without a physics material."""

_SHAPE_CONTACT_ARRAYS = {f"shape_material_{key}": _SHAPE[key].name for key in ("ke", "kd", "kf", "ka")}


class UsdExporter(BaseUsdExporter):
    """Export a Newton scene environment: its USD plus every live model value the USD does not state."""

    def compute_diff(self, stage: Usd.Stage) -> list[UsdEdit]:
        """Return the live model values that ``stage`` does not already state; record the step size and solver.

        Args:
            stage: The exported stage, reduced to the selected environment.

        Returns:
            Values to author so that :meth:`load_usd` reproduces the live model.
        """
        solver_cfg = self.scene.sim.cfg.physics.solver_cfg
        model = NewtonManager.get_model()
        gravity = model.gravity.numpy()
        gravity = gravity[self.env_id] if gravity.ndim == 2 else gravity
        edits = [
            *self._joint_edits(stage, model),
            *self._body_edits(stage, model),
            *self._shape_edits(stage, model),
            *self._gravity_edits(stage, gravity),
        ]
        settings = {
            "dt": self.scene.sim.get_physics_dt(),
            "force_position_velocity_actuation": self._force_position_velocity,
            "solver_type": f"{type(solver_cfg).__module__}:{type(solver_cfg).__name__}",
            "solver": solver_cfg.to_dict(),
        }
        layer = stage.GetRootLayer()
        layer.customLayerData = {**layer.customLayerData, SETTINGS_KEY: json.dumps(settings)}
        return edits

    @classmethod
    def load_usd(
        cls, path: str, device: str = "cpu"
    ) -> tuple[newton.Model, newton.solvers.SolverBase | None, float | None]:
        """Load a USD file exported from a Newton scene.

        Referenced remote assets are downloaded first, as Isaac Lab does when spawning them.

        Args:
            path: Exported USD file.
            device: Device of the model and solver. Defaults to ``"cpu"``.

        Returns:
            The model, the solver configured as in training (None if the file records none), and the
            physics step size [s] (None if the file records none).
        """
        stage = cls.open_stage(path)
        settings = json.loads(stage.GetRootLayer().customLayerData.get(SETTINGS_KEY, "{}"))
        solver_cfg = None
        if "solver_type" in settings:
            solver_cfg = string_to_callable(settings["solver_type"])()
            solver_cfg.from_dict(settings["solver"])
        force = settings.get("force_position_velocity_actuation", False)
        model = cls.import_model(stage, solver_cfg, device, force)
        solver = cls._manager(solver_cfg)._create_solver(model, solver_cfg) if solver_cfg is not None else None
        return model, solver, settings.get("dt")

    @classmethod
    def import_model(
        cls, stage: Usd.Stage, solver_cfg=None, device: str = "cpu", force_position_velocity_actuation: bool = False
    ) -> newton.Model:
        """Import a stage with Newton's USD importer, as :meth:`load_usd` does.

        Only the solver's custom attributes and schema resolvers are applied; Isaac Lab's builder
        defaults are not, so an exported file carries every value that differs from Newton's.

        Args:
            stage: Stage to import.
            solver_cfg: Newton solver configuration whose attributes the import registers. Defaults to None.
            device: Device of the finalized model. Defaults to ``"cpu"``.
            force_position_velocity_actuation: Actuate joints with both drive gains in position-velocity
                mode. Defaults to False.

        Returns:
            The finalized model.
        """
        manager = cls._manager(solver_cfg)
        builder = newton.ModelBuilder()
        manager._register_builder_attributes(builder)
        # Tagged terrains become heightfields, as in Isaac Lab; the importer does not read their colliders.
        terrains = manager._inject_terrain_heightfields(stage, builder, root_paths=("/",), device=device)
        for shape, label in enumerate(builder.shape_label):
            cls._apply_authored_contacts(stage.GetPrimAtPath(label), builder, shape)
        builder.add_usd(
            stage,
            schema_resolvers=manager._get_usd_import_schema_resolvers(),
            ignore_paths=terrains,
            force_position_velocity_actuation=force_position_velocity_actuation,
        )
        return builder.finalize(device=device)

    def _selected(self, worlds) -> np.ndarray:
        return np.flatnonzero(np.isin(worlds.numpy(), (-1, self.env_id)))

    def _joint_edits(self, stage: Usd.Stage, model: newton.Model) -> list[UsdEdit]:
        values = self._arrays(model, DOF_ATTRIBUTES)
        starts, types, modes = model.joint_qd_start.numpy(), model.joint_type.numpy(), model.joint_target_mode.numpy()
        axes = {int(newton.JointType.REVOLUTE): "angular", int(newton.JointType.PRISMATIC): "linear"}
        edits, dofs, multi_axis = [], [], []
        for joint in self._selected(model.joint_world):
            prim = stage.GetPrimAtPath(model.joint_label[joint])
            # Free joints of floating bodies have no USD prim; their pose is exported by the body.
            if types[joint] == int(newton.JointType.FREE) or not prim:
                continue
            axis = axes.get(int(types[joint]))
            if axis is None:
                if starts[joint + 1] > starts[joint]:
                    multi_axis.append(model.joint_label[joint])
                continue
            dof = starts[joint]
            dofs.append(dof)
            # Newton's importer actuates exactly the joints with a USD drive.
            actuated = modes[dof] != int(newton.JointTargetMode.NONE)
            if not actuated and prim.HasAPI(UsdPhysics.DriveAPI, axis):
                raise NotImplementedError(f"Joint '{prim.GetPath()}' has a USD drive the engine does not actuate.")
            for name, (attribute, schema, angular_scale) in DOF_ATTRIBUTES.items():
                if schema is None or actuated:
                    scale = angular_scale if axis == "angular" else 1.0
                    schema = schema and schema.format(axis=axis)
                    value = float(values[name][dof]) * scale
                    edits.append(self._value_edit(prim, attribute.format(axis=axis), value, schema=schema))
        if multi_axis:
            logger.warning(f"Multi-axis joints are not exported: {multi_axis}.")
        self._force_position_velocity = self._actuation_flag(
            values["joint_target_ke"][dofs], values["joint_target_kd"][dofs], modes[dofs]
        )
        return [edit for edit in edits if edit is not None]

    def _body_edits(self, stage: Usd.Stage, model: newton.Model) -> list[UsdEdit]:
        mass, com, inertia = model.body_mass.numpy(), model.body_com.numpy(), model.body_inertia.numpy()
        edits = []
        for body in self._selected(model.body_world):
            prim = stage.GetPrimAtPath(model.body_label[body])
            if prim:
                edits += self._mass_edits(prim, float(mass[body]), com[body], inertia[body])
        # Root bodies and world anchors are placed where initialization put them; joint state is not exported.
        body_q, anchors = NewtonManager.get_state_0().body_q.numpy(), model.joint_X_p.numpy()
        parents, children, types = model.joint_parent.numpy(), model.joint_child.numpy(), model.joint_type.numpy()
        for joint in self._selected(model.joint_world):
            if parents[joint] != -1:
                continue
            body = stage.GetPrimAtPath(model.body_label[children[joint]])
            if body:
                edits.append(self._pose_edit(body, body_q[children[joint]]))
            prim = stage.GetPrimAtPath(model.joint_label[joint])
            if types[joint] != int(newton.JointType.FREE) and prim:
                edits.append(self._value_edit(prim, "physics:localPos0", anchors[joint][:3], "point3f"))
                edits.append(self._value_edit(prim, "physics:localRot0", anchors[joint][3:7], "quatf"))
        return [edit for edit in edits if edit is not None]

    def _shape_edits(self, stage: Usd.Stage, model: newton.Model) -> list[UsdEdit]:
        values, material = self._arrays(model, SHAPE_ATTRIBUTES), self._arrays(model, MATERIAL_ATTRIBUTES)
        flags, collide = model.shape_flags.numpy(), int(newton.ShapeFlags.COLLIDE_SHAPES)
        edits, materials = [], {}
        for shape in self._selected(model.shape_world):
            prim = stage.GetPrimAtPath(model.shape_label[shape])
            # Sensor sites and visual shapes do not collide; generated shapes have no USD prim.
            if not prim or not int(flags[shape]) & collide:
                continue
            edits += [
                self._value_edit(prim, attribute, float(values[name][shape]))
                for name, attribute in SHAPE_ATTRIBUTES.items()
            ]
            materials[str(prim.GetPath())] = {a: float(material[n][shape]) for n, a in MATERIAL_ATTRIBUTES.items()}
        edits = [edit for edit in edits if edit is not None]
        return edits + self._material_edits(
            stage, materials, lambda c, v: self._collider_contact_edits(stage.GetPrimAtPath(c), v)
        )

    @staticmethod
    def _apply_authored_contacts(prim: Usd.Prim, builder: newton.ModelBuilder, shape: int):
        """Apply the contact values authored on a collider, or its physics material, to a builder shape."""
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        sources = [(prim, SHAPE_ATTRIBUTES), (prim, _SHAPE_CONTACT_ARRAYS)]
        if material:
            sources.append((material.GetPrim(), MATERIAL_ATTRIBUTES))
        for owner, table in sources:
            for array, attribute in table.items():
                value = owner.GetAttribute(attribute).Get()
                if value is not None:
                    getattr(builder, array)[shape] = float(value)

    @staticmethod
    def _manager(solver_cfg) -> type[NewtonManager]:
        if solver_cfg is None:
            return NewtonManager
        manager = solver_cfg.class_type
        return string_to_callable(manager) if isinstance(manager, str) else manager

    @staticmethod
    def _actuation_flag(target_ke, target_kd, modes) -> bool:
        """Return the ``force_position_velocity_actuation`` import option that reproduces all engine modes."""
        for force in (False, True):
            inferred = [
                int(newton.JointTargetMode.from_gains(float(ke), float(kd), force, has_drive=mode != 0))
                for ke, kd, mode in zip(target_ke, target_kd, modes)
            ]
            if inferred == modes.tolist():
                return force
        raise NotImplementedError("Newton's USD import cannot mix position and position-velocity joint actuation.")

    @staticmethod
    def _arrays(model, names) -> dict[str, np.ndarray]:
        return {name: getattr(model, name).numpy() for name in names}

    @classmethod
    def _collider_contact_edits(cls, prim: Usd.Prim, values: dict[str, float]) -> list[UsdEdit]:
        """Contact gains of a collider without a physics material, which Newton reads from the collider."""
        edits = (
            cls._value_edit(prim, SHAPE_CONTACT_ATTRIBUTES[name], value)
            for name, value in values.items()
            if name in SHAPE_CONTACT_ATTRIBUTES
        )
        return [edit for edit in edits if edit is not None]
