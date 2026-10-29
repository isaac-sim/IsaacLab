# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton USD export: every live model value that Isaac Lab set after building the model from the stage."""

from __future__ import annotations

import logging
import math
from collections import Counter

import newton
import numpy as np
from newton.usd import PrimType, SchemaResolverNewton

from pxr import Sdf, Tf, Usd, UsdPhysics, UsdShade

from isaaclab.sim.usd_export.base_usd_exporter import (
    USD_PHYSICS_D6_LIMIT_ATTRIBUTES,
    USD_PHYSICS_JOINT_ATTRIBUTES,
    USD_PHYSICS_MATERIAL_ATTRIBUTES,
    BaseUsdExporter,
    JointAttribute,
    UsdEdit,
)
from isaaclab.utils.string import string_to_callable

from isaaclab_newton.physics import NewtonSolver, NewtonSolverCfg
from isaaclab_newton.physics.newton_builder import (
    register_builder_attributes,
    usd_schema_resolvers,
)

logger = logging.getLogger(__name__)

_SCENE = SchemaResolverNewton.mapping[PrimType.SCENE]
_JOINT = SchemaResolverNewton.mapping[PrimType.JOINT]
_SHAPE = SchemaResolverNewton.mapping[PrimType.SHAPE]
_MATERIAL = SchemaResolverNewton.mapping[PrimType.MATERIAL]

DOF_ATTRIBUTES = {
    "joint_target_ke": USD_PHYSICS_JOINT_ATTRIBUTES["stiffness"],
    "joint_target_kd": USD_PHYSICS_JOINT_ATTRIBUTES["damping"],
    "joint_effort_limit": USD_PHYSICS_JOINT_ATTRIBUTES["max_force"],
    "joint_limit_lower": USD_PHYSICS_JOINT_ATTRIBUTES["lower_limit"],
    "joint_limit_upper": USD_PHYSICS_JOINT_ATTRIBUTES["upper_limit"],
    "joint_armature": JointAttribute(_JOINT["armature"].name, None, 1.0),
    "joint_friction": JointAttribute(_JOINT["friction"].name, None, 1.0),
    "joint_damping": JointAttribute(_JOINT["damping"].name, None, math.radians(1.0)),
    "joint_velocity_limit": JointAttribute(_JOINT["velocity_limit"].name, None, math.degrees(1.0)),
    "joint_limit_ke": JointAttribute(_JOINT["limit_ke"].name, None, math.radians(1.0)),
    "joint_limit_kd": JointAttribute(_JOINT["limit_kd"].name, None, math.radians(1.0)),
}
"""Model DOF array -> USD joint attribute; one without ``{axis}`` holds one value for the whole joint."""

D6_DOF_ATTRIBUTES = {
    **DOF_ATTRIBUTES,
    "joint_limit_lower": USD_PHYSICS_D6_LIMIT_ATTRIBUTES["lower_limit"],
    "joint_limit_upper": USD_PHYSICS_D6_LIMIT_ATTRIBUTES["upper_limit"],
    "joint_limit_ke": JointAttribute(_JOINT["limit_rotX_ke"].name.replace("rotX", "{axis}"), None, math.radians(1.0)),
    "joint_limit_kd": JointAttribute(_JOINT["limit_rotX_kd"].name.replace("rotX", "{axis}"), None, math.radians(1.0)),
}
""":data:`DOF_ATTRIBUTES` of a rotational D6 joint, whose axes are ``rotX``, ``rotY`` and ``rotZ``."""

JOINT_WIDE_LIMIT_GAINS = {"joint_limit_ke": _JOINT["limit_ke"].name, "joint_limit_kd": _JOINT["limit_kd"].name}
"""Joint-wide limit gain attributes, which Newton reads ahead of a D6 joint's per-axis ones."""

SHAPE_ATTRIBUTES = {
    "shape_margin": _SHAPE["margin"].name,
    "shape_gap": _SHAPE["gap"].name,
}
"""Model shape array -> USD attribute on the collider."""

MATERIAL_ATTRIBUTES = {
    "shape_material_mu": USD_PHYSICS_MATERIAL_ATTRIBUTES["dynamic_friction"],
    "shape_material_restitution": USD_PHYSICS_MATERIAL_ATTRIBUTES["restitution"],
    "shape_material_mu_torsional": _MATERIAL["mu_torsional"].name,
    "shape_material_mu_rolling": _MATERIAL["mu_rolling"].name,
    "shape_material_ke": _MATERIAL["ke"].name,
    "shape_material_kd": _MATERIAL["kd"].name,
    "shape_material_kf": _MATERIAL["kf"].name,
    "shape_material_ka": _MATERIAL["ka"].name,
}
"""Model shape array -> USD attribute on the collider's physics material, which takes precedence."""

MUJOCO_FLAG_FIELDS = {
    "enable_multiccd": ("flag_multiccd", False),
    "disable_contacts": ("flag_contact", True),
    "disable_sensors": ("flag_sensor", True),
}
"""MuJoCo solver configuration field -> Newton's flag attribute, and whether the field inverts the flag."""


class UsdExporter(BaseUsdExporter):
    """Export a Newton scene environment: its USD plus every live model value Isaac Lab set after the build.

    The base is Newton's own import of the exported stage, without Isaac Lab: what a consumer of the file gets
    from USD alone. A value is written where Isaac Lab's live model differs from it, so Isaac Lab's builder
    defaults, its asset configuration and its other overrides are all stated, and nothing Newton's import
    derives by itself is.

    The file is meant for Newton's USD import (:meth:`newton.ModelBuilder.add_usd`) with the solver's custom
    attributes registered and the Newton, PhysX and, for MuJoCo, MuJoCo schema resolvers. USD cannot yet
    state Newton's substeps and collision pipeline settings, nor a joint target mode other than the one the
    import infers from the gains; the export warns about these. Terrain that Isaac Lab collides as a
    heightfield is imported as its mesh.
    """

    def compute_diff(self, stage: Usd.Stage) -> list[UsdEdit]:
        """Return the live model values that differ from Newton's import of ``stage``.

        Args:
            stage: The exported stage, reduced to the selected environment.

        Returns:
            Values to author so that Newton's import reproduces the live model.
        """
        physics_cfg = self.scene.sim.cfg.physics
        solver_cfg = physics_cfg.solver_cfg
        model = self.scene.sim.backend_manager.backend.model
        builder = self._import_base(stage, model, solver_cfg)
        gravity = model.gravity.numpy()
        gravity = gravity[self.env_id] if gravity.ndim == 2 else gravity
        joint_edits, drives = self._joint_edits(stage, model, builder)
        edits = [
            *joint_edits,
            *self._mjc_actuator_edits(stage, model, drives),
            *self._body_edits(stage, model, builder),
            *self._shape_edits(stage, model, builder),
            *self._gravity_edits(stage, gravity),
            *self._scene_edits(solver_cfg),
        ]
        # MuJoCo's and Kamino's own contacts are found every solver step; Newton's pipeline once per physics step.
        pipeline = not getattr(solver_cfg, "use_mujoco_contacts", getattr(solver_cfg, "use_collision_detector", False))
        if physics_cfg.num_substeps > 1 and pipeline:
            logger.warning(
                "The file states the solver step rate; an import running Newton's collision pipeline collides every"
                " solver step, while Isaac Lab collides once per physics step."
            )
        return edits

    @staticmethod
    def _import_base(stage: Usd.Stage, model: newton.Model, solver_cfg) -> _AlignedBase:
        """Newton's import of ``stage`` without Isaac Lab, its arrays indexed like ``model``'s entities.

        The import registers the solver's custom attributes and uses Newton's schema resolvers, as a consumer of
        the file does, and is finalized, so what Newton derives on import, such as corrected inertia, is part of
        the base and never written.
        """
        builder = newton.ModelBuilder()
        register_builder_attributes(builder, solver_cfg)
        # The stage's own actuator networks, which Isaac Lab loaded when it built the scene.
        builder.add_usd(stage, schema_resolvers=usd_schema_resolvers(solver_cfg), allow_unsafe_torch_checkpoints=True)
        return _AlignedBase(model, builder.finalize(device="cpu"))

    def _selected(self, worlds) -> np.ndarray:
        return np.flatnonzero(np.isin(worlds.numpy(), (-1, self.env_id)))

    def _joint_edits(
        self, stage: Usd.Stage, model: newton.Model, builder: newton.ModelBuilder
    ) -> tuple[list[UsdEdit], dict[str, tuple[float, float]]]:
        values = self._arrays(model, DOF_ATTRIBUTES)
        changed = self._changed(values, builder)
        starts, types, modes = model.joint_qd_start.numpy(), model.joint_type.numpy(), model.joint_target_mode.numpy()
        reference = self._limit_reference(model)
        edits, dofs, drives = [], {}, {}
        for joint in self._selected(model.joint_world):
            prim = stage.GetPrimAtPath(model.joint_label[joint])
            # Free joints of floating bodies have no USD prim; their pose is exported by the body.
            if types[joint] == int(newton.JointType.FREE) or not prim:
                continue
            units = self._joint_prims(stage, model, joint, prim)
            if units is None:
                if any(changed[name][starts[joint] : starts[joint + 1]].any() for name in changed):
                    raise NotImplementedError(
                        f"Joint '{model.joint_label[joint]}' changed; only revolute, prismatic and rotational D6"
                        " joints are exported."
                    )
                continue
            for prim, axes, table in units:
                path = str(prim.GetPath())
                dofs.update({dof: path for dof, _ in axes})
                # Newton's importer actuates exactly the axes with a USD drive.
                actuated = {dof: modes[dof] != int(newton.JointTargetMode.NONE) for dof, _ in axes}
                for dof, axis in axes:
                    if not actuated[dof] and prim.HasAPI(UsdPhysics.DriveAPI, axis):
                        raise NotImplementedError(f"Joint '{path}' has a USD drive the engine does not actuate.")
                # A changed drive is written whole, since Isaac Lab's build can hold drive values other than the USD's.
                driven = {
                    name for name, (_, schema, _) in table.items() if schema and schema.startswith("PhysicsDriveAPI")
                }
                drive = {dof: actuated[dof] and any(changed[name][dof] for name in driven) for dof, _ in axes}
                if len(axes) == 1 and drive[axes[0][0]]:
                    dof = axes[0][0]
                    drives[path] = (float(values["joint_target_ke"][dof]), float(values["joint_target_kd"][dof]))
                for name, (attribute, schema, angular_scale) in table.items():
                    attribute = self._limit_gain_attribute(prim, name, attribute)
                    live = (
                        values[name] + reference if name in ("joint_limit_lower", "joint_limit_upper") else values[name]
                    )
                    if "{axis}" not in attribute:
                        if any(changed[name][dof] for dof, _ in axes):
                            edits.append(self._joint_value_edit(path, attribute, live, axes, angular_scale))
                        continue
                    for dof, axis in axes:
                        if (changed[name][dof] and (name not in driven or actuated[dof])) or (
                            drive[dof] and name in driven
                        ):
                            scale = 1.0 if axis == "linear" else angular_scale
                            value = float(live[dof]) * scale
                            edits.append(
                                UsdEdit(
                                    path, attribute.format(axis=axis), value, schema=schema and schema.format(axis=axis)
                                )
                            )
        self._target_mode_actuators(stage, model, dofs, values["joint_target_ke"], values["joint_target_kd"], modes)
        return edits + self._force_range_edits(stage, edits), drives

    @staticmethod
    def _joint_prims(
        stage: Usd.Stage, model: newton.Model, joint: int, prim: Usd.Prim
    ) -> list[tuple[Usd.Prim, list[tuple[int, str]], dict]] | None:
        """The USD joints of a model joint, each with its DOFs, their USD axis names and the attribute table.

        Newton merges single-axis USD joints between the same two bodies into one D6 joint named after the
        first; MuJoCo's per-DOF labels name the USD joint of each DOF. None for unsupported joints.
        """
        start, joint_type = int(model.joint_qd_start.numpy()[joint]), int(model.joint_type.numpy()[joint])
        if joint_type == int(newton.JointType.REVOLUTE):
            return [(prim, [(start, "angular")], DOF_ATTRIBUTES)]
        if joint_type == int(newton.JointType.PRISMATIC):
            return [(prim, [(start, "linear")], DOF_ATTRIBUTES)]
        if joint_type != int(newton.JointType.D6):
            return None
        linear, angular = model.joint_dof_dim.numpy()[joint]
        dofs = range(start, start + int(linear + angular))
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            labels = getattr(getattr(model, "mujoco", None), "joint_dof_label", None)
            if labels is None:
                raise NotImplementedError(f"Joint '{prim.GetPath()}' merges USD joints the model does not name.")
            sources = [stage.GetPrimAtPath(labels[dof]) for dof in dofs]
            return [
                (source, [(dof, "angular" if source.IsA(UsdPhysics.RevoluteJoint) else "linear")], DOF_ATTRIBUTES)
                for dof, source in zip(dofs, sources)
            ]
        if linear:
            return None
        axes = model.joint_axis.numpy()
        return [(prim, [(dof, "rot" + "XYZ"[int(np.argmax(np.abs(axes[dof])))]) for dof in dofs], D6_DOF_ATTRIBUTES)]

    @classmethod
    def _joint_value_edit(
        cls, path: str, attribute: str, values: np.ndarray, axes: list[tuple[int, str]], angular_scale: float
    ) -> UsdEdit:
        """Edit writing a value USD states once per joint, which all its axes must share."""
        live = [float(values[dof]) for dof, _ in axes]
        if not cls._close(live, live[0]):
            raise NotImplementedError(f"Joint '{path}' has a different '{attribute}' per axis; USD states one.")
        return UsdEdit(path, attribute, live[0] * (1.0 if axes[0][1] == "linear" else angular_scale))

    def _body_edits(self, stage: Usd.Stage, model: newton.Model, builder: newton.ModelBuilder) -> list[UsdEdit]:
        values = self._arrays(model, ("body_mass", "body_com", "body_inertia"))
        changed = self._changed(values, builder)
        mass, com, inertia = values["body_mass"], values["body_com"], values["body_inertia"]
        edits = []
        for body in self._selected(model.body_world):
            prim = stage.GetPrimAtPath(model.body_label[body])
            properties = {name.removeprefix("body_") for name, rows in changed.items() if rows[body]}
            if prim and properties:
                edits += self._mass_edits(prim, float(mass[body]), com[body], inertia[body], properties)
        # Root bodies and world anchors are placed where initialization put them; joint state is not exported.
        body_q, anchors = self.scene.sim.backend_manager.backend.state_0.body_q.numpy(), model.joint_X_p.numpy()
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

    def _shape_edits(self, stage: Usd.Stage, model: newton.Model, builder) -> list[UsdEdit]:
        values, material = self._arrays(model, SHAPE_ATTRIBUTES), self._arrays(model, MATERIAL_ATTRIBUTES)
        changed = self._changed(values, builder) | self._changed(material, builder)
        flags, collide = model.shape_flags.numpy(), int(newton.ShapeFlags.COLLIDE_SHAPES)
        edits, materials, complete, bound = [], {}, {}, {}
        for shape in self._selected(model.shape_world):
            prim = stage.GetPrimAtPath(model.shape_label[shape])
            # Sensor sites and visual shapes do not collide; generated shapes have no USD prim.
            if not prim or not int(flags[shape]) & collide:
                continue
            edits += [
                UsdEdit(str(prim.GetPath()), attribute, float(values[name][shape]))
                for name, attribute in SHAPE_ATTRIBUTES.items()
                if changed[name][shape]
            ]
            path = str(prim.GetPath())
            bound[path] = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]
            complete[path] = {a: float(material[n][shape]) for n, a in MATERIAL_ATTRIBUTES.items()}
            updated = {a: complete[path][a] for n, a in MATERIAL_ATTRIBUTES.items() if changed[n][shape]}
            if updated:
                materials[path] = updated
        # A changed material value is checked against every collider bound to the material; values no collider
        # changed are left to the import.
        affected: dict[str, set[str]] = {}
        for path, updated in materials.items():
            if bound[path]:
                affected.setdefault(str(bound[path].GetPath()), set()).update(updated)
        materials |= {
            path: {
                name: value for name, value in complete[path].items() if name in affected[str(bound[path].GetPath())]
            }
            for path in complete
            if bound[path] and str(bound[path].GetPath()) in affected
        }
        # A collider without a material gets a new one, which states every value an import reads from it.
        materials |= {path: complete[path] for path in materials if not bound[path]}
        edits = [edit for edit in edits if edit is not None]
        # An unset contact gain of a material falls back to Newton's builder default.
        upstream = newton.ModelBuilder.ShapeConfig()
        fallbacks = {
            attribute: getattr(upstream, array.removeprefix("shape_material_"))
            for array, attribute in MATERIAL_ATTRIBUTES.items()
        }
        return edits + self._material_edits(stage, materials, fallbacks)

    @classmethod
    def _mjc_actuator_edits(
        cls, stage: Usd.Stage, model: newton.Model, drives: dict[str, tuple[float, float]]
    ) -> list[UsdEdit]:
        """Edits giving the MuJoCo actuators of joints whose drive the export writes the joint's live gain.

        Workaround: Newton's import turns a MuJoCo position or velocity actuator into the targeted joint's
        drive, setting the joint's stiffness or damping to the actuator's gain. Only the actuators the import
        converted this way are edited; Newton keeps the others as separate actuators. An actuator whose gain
        becomes zero is deactivated, since the import would keep it as a separate actuator.

        Args:
            stage: The exported stage.
            model: The live model, which records the actuators Newton's import converted.
            drives: Live stiffness and damping of the joints whose drive the export writes, keyed by joint path.

        Returns:
            The edits.
        """
        mujoco = getattr(model, "mujoco", None)
        if mujoco is None or not hasattr(mujoco, "ctrl_source"):
            return []
        joint_target = int(newton.solvers.SolverMuJoCo.CtrlSource.JOINT_TARGET)
        converted = {
            label for label, source in zip(mujoco.actuator_label, mujoco.ctrl_source.numpy()) if source == joint_target
        }
        edits, deactivated = [], []
        for prim in [prim for prim in stage.Traverse() if str(prim.GetPath()) in converted]:
            targets = [str(target) for target in prim.GetRelationship("mjc:target").GetTargets()]
            if len(targets) != 1 or targets[0] not in drives:
                continue
            # Unset parameters take MuJoCo's defaults.
            gain = list(prim.GetAttribute("mjc:gainPrm").Get() or [1.0] + [0.0] * 9)
            bias = list(prim.GetAttribute("mjc:biasPrm").Get() or [0.0] * 10)
            # A position actuator biases by -kp * qpos, a velocity actuator by -kv * qvel.
            index = 1 if bias[1] != 0.0 else 2
            stiffness, damping = drives[targets[0]]
            value = stiffness if index == 1 else damping
            if cls._close(gain[0], value):
                continue
            if value <= 0.0:
                prim.SetActive(False)
                deactivated.append(str(prim.GetPath()))
                continue
            gain[0], bias[index] = value, -value
            path = str(prim.GetPath())
            edits += [UsdEdit(path, "mjc:gainPrm", gain, "double[]"), UsdEdit(path, "mjc:biasPrm", bias, "double[]")]
        if deactivated:
            logger.info(f"MuJoCo actuators replaced by the exported zero-gain joint drives: {deactivated}.")
        return edits

    def _scene_edits(self, solver_cfg) -> list[UsdEdit]:
        """Edits stating the solver step rate and the solver options Newton's importer reads from the physics scene.

        Isaac Lab steps the solver ``num_substeps`` times per physics step; USD states one rate, so the file states
        the solver's, which Newton's import returns as ``physics_dt``. An option equal to the value Newton's import
        uses when the scene does not state it is not written.
        """
        path = self.scene.physics_scene_path
        rate = self.scene.sim.cfg.physics.num_substeps / self.scene.sim.get_physics_dt()
        if not math.isclose(rate, round(rate), rel_tol=1e-9):
            raise ValueError(f"Step size 1/{rate} s is not a whole number of steps per second, as USD states it.")
        edits = [UsdEdit(path, _SCENE["time_steps_per_second"].name, round(rate), "int", "NewtonSceneAPI")]
        types = {bool: "bool", int: "int", float: "float", str: "token"}
        for name, attribute in self._scene_solver_options(solver_cfg).items():
            value = getattr(solver_cfg, name)
            if name in MUJOCO_FLAG_FIELDS and MUJOCO_FLAG_FIELDS[name][1] and value is not None:
                value = not value
            if value is not None and not self._is_import_default(attribute, value):
                edits.append(UsdEdit(path, attribute.usd_attribute_name, value, types[type(value)]))
        if self._runs(solver_cfg, newton.solvers.SolverKamino):
            # Of Kamino's settings Newton reads from USD, only the constraint stabilization gains are written: the
            # PADMM ones change the result only together with settings USD cannot state (rho_0, warm start method).
            builder = newton.ModelBuilder()
            newton.solvers.SolverKamino.register_custom_attributes(builder)
            constraints = solver_cfg.to_solver_config().constraints
            for gain in ("alpha", "beta", "gamma"):
                attribute = builder.custom_attributes[f"kamino:constraints_{gain}"]
                value = float(getattr(constraints, gain))
                if not self._is_import_default(attribute, value):
                    edits.append(UsdEdit(path, attribute.usd_attribute_name, value))
        return edits

    @staticmethod
    def _is_import_default(attribute, value) -> bool:
        """Whether Newton's import resolves ``attribute`` to ``value`` when the USD does not state it."""
        if attribute.default is None:
            return False
        if attribute.usd_value_transformer is not None:
            value = attribute.usd_value_transformer(value, None)
        if isinstance(value, float) or isinstance(attribute.default, float):
            return bool(np.isclose(value, attribute.default, rtol=1e-6, atol=0.0))
        return value == attribute.default

    @classmethod
    def _scene_solver_options(cls, solver_cfg) -> dict[str, newton.ModelBuilder.CustomAttribute]:
        """Solver configuration field -> Newton's physics-scene attribute, for options the solver reads from the model.

        Only MuJoCo reads its options from the model; Newton registers their USD attributes.
        """
        if not cls._runs(solver_cfg, newton.solvers.SolverMuJoCo):
            return {}
        builder = newton.ModelBuilder()
        register_builder_attributes(builder, solver_cfg)
        scene = (newton.Model.AttributeFrequency.ONCE, newton.Model.AttributeFrequency.WORLD)
        attributes = {
            attribute.name: attribute
            for attribute in builder.custom_attributes.values()
            if attribute.namespace == "mujoco" and attribute.frequency in scene and attribute.usd_attribute_name
        }
        fields = solver_cfg.to_dict()
        options = {name: attributes[name] for name in fields if name in attributes}
        options |= {
            name: attributes[flag]
            for name, (flag, _) in MUJOCO_FLAG_FIELDS.items()
            if name in fields and flag in attributes
        }
        return options

    @staticmethod
    def _limit_reference(model: newton.Model) -> np.ndarray:
        """Per DOF, MuJoCo's joint reference, which Newton's import subtracts from the USD limits."""
        mujoco = getattr(model, "mujoco", None)
        if mujoco is not None and hasattr(mujoco, "dof_ref"):
            return mujoco.dof_ref.numpy().astype(float)
        return np.zeros(model.joint_dof_count)

    @classmethod
    def _limit_gain_attribute(cls, prim: Usd.Prim, name: str, attribute: str) -> str:
        """The attribute a limit gain is written to: the joint-wide one when stated, which Newton reads first."""
        joint_wide = JOINT_WIDE_LIMIT_GAINS.get(name)
        if "{axis}" in attribute and joint_wide and cls._stated(prim, joint_wide) is not None:
            return joint_wide
        return attribute

    @classmethod
    def _force_range_edits(cls, stage: Usd.Stage, edits: list[UsdEdit]) -> list[UsdEdit]:
        """Edits widening a MuJoCo joint force range that would clamp an exported drive force limit.

        Newton's import keeps the tighter of a drive's max force and the joint's ``mjc:actuatorfrcrange``.
        """
        widened = []
        for edit in edits:
            if not edit.attribute.endswith("physics:maxForce"):
                continue
            prim = stage.GetPrimAtPath(edit.path)
            bounds = [cls._stated(prim, f"mjc:actuatorfrcrange:{end}") for end in ("min", "max")]
            if any(bound is not None for bound in bounds) and max(abs(b or 0.0) for b in bounds) < edit.value:
                widened += [
                    UsdEdit(edit.path, "mjc:actuatorfrcrange:min", -float(edit.value)),
                    UsdEdit(edit.path, "mjc:actuatorfrcrange:max", float(edit.value)),
                ]
        return widened

    @staticmethod
    def _solver(solver_cfg: NewtonSolverCfg) -> type[NewtonSolver]:
        solver = solver_cfg.class_type
        solver = string_to_callable(solver) if isinstance(solver, str) else solver
        # An exported file sets the class type; accept only Newton solvers.
        if not (isinstance(solver, type) and issubclass(solver, NewtonSolver)):
            raise ValueError(f"'{solver_cfg.class_type}' is not a Newton solver.")
        return solver

    @classmethod
    def _runs(cls, solver_cfg: NewtonSolverCfg | None, solver_class: type) -> bool:
        """Whether the configured solver runs ``solver_class`` (and so reads its custom attributes)."""
        if solver_cfg is None:
            return False
        return any(issubclass(c, solver_class) for c in cls._solver(solver_cfg).solver_classes(solver_cfg))

    def _target_mode_actuators(
        self, stage: Usd.Stage, model: newton.Model, dofs: dict[int, str], target_ke, target_kd, modes
    ):
        """Make Newton's USD import reproduce each joint's target mode, and warn where it cannot.

        Workaround until USD states a target mode: Newton's import infers the mode from the drive gains, so a joint
        with both gains runs in position mode, while Isaac Lab runs it in position-velocity mode. A MuJoCo velocity
        actuator targeting a single-axis joint makes the import use position-velocity mode (gain: the damping).
        """
        position_velocity = int(newton.JointTargetMode.POSITION_VELOCITY)
        mujoco = self._runs(self.scene.sim.cfg.physics.solver_cfg, newton.solvers.SolverMuJoCo)
        converted = self._converted_velocity_dofs(model) if mujoco else set()
        axes = Counter(dofs.values())
        root = self.scene.env_prim_paths[self.env_id]
        differ = set()
        for dof, path in dofs.items():
            inferred = newton.JointTargetMode.from_gains(float(target_ke[dof]), float(target_kd[dof]), has_drive=True)
            if modes[dof] == int(newton.JointTargetMode.NONE) or modes[dof] == int(inferred) or dof in converted:
                continue
            if not (mujoco and modes[dof] == position_velocity and axes[path] == 1):
                differ.add(path)
                continue
            damping = float(target_kd[dof])
            name = Tf.MakeValidIdentifier(path.removeprefix(root + "/"))
            actuator = stage.DefinePrim(f"{root}/VelocityActuators/{name}", "MjcActuator")
            actuator.CreateRelationship("mjc:target", custom=False).SetTargets([path])
            values = {
                "mjc:biasType": (Sdf.ValueTypeNames.Token, "affine"),
                "mjc:gainPrm": (Sdf.ValueTypeNames.DoubleArray, [damping] + [0.0] * 9),
                "mjc:biasPrm": (Sdf.ValueTypeNames.DoubleArray, [0.0, 0.0, -damping] + [0.0] * 7),
            }
            # As the MjcActuator schema defines them, whether or not this process registered it.
            for attribute, (type_name, value) in values.items():
                actuator.CreateAttribute(attribute, type_name, custom=False, variability=Sdf.VariabilityUniform).Set(
                    value
                )
        if differ:
            logger.warning(
                f"USD cannot state the target mode of these joints; an import infers another: {sorted(differ)}."
            )

    @staticmethod
    def _converted_velocity_dofs(model: newton.Model) -> set[int]:
        """DOFs driven by a MuJoCo velocity actuator that Newton's import converted to a joint target."""
        mujoco = getattr(model, "mujoco", None)
        if mujoco is None or not hasattr(mujoco, "ctrl_type"):
            return set()
        solver = newton.solvers.SolverMuJoCo
        rows = zip(mujoco.ctrl_source.numpy(), mujoco.ctrl_type.numpy(), mujoco.actuator_trnid.numpy())
        return {
            int(target[0])
            for source, kind, target in rows
            if source == int(solver.CtrlSource.JOINT_TARGET) and kind == int(solver.CtrlType.VELOCITY)
        }

    @staticmethod
    def _arrays(model, names) -> dict[str, np.ndarray]:
        return {name: getattr(model, name).numpy() for name in names}

    @staticmethod
    def _changed(values: dict[str, np.ndarray], builder: newton.ModelBuilder) -> dict[str, np.ndarray]:
        """Per entity, whether a live model array differs from the value the builder took from the stage."""
        changed = {}
        for name, live in values.items():
            # An explicit width also fits a model without any such entity.
            shape = (len(live), int(np.prod(live.shape[1:])))
            built = np.asarray(getattr(builder, name), dtype=float).reshape(shape)
            changed[name] = ~np.isclose(live.reshape(shape), built, rtol=1e-5, atol=1e-7).all(axis=1)
        return changed


class _AlignedBase:
    """The arrays of a base model indexed like a live model's entities, matched by body, shape and joint label.

    An entity of the live model that the base does not have, such as one of another environment, keeps its
    live value, so it never counts as changed.
    """

    def __init__(self, live: newton.Model, base: newton.Model):
        self._live, self._base, self._arrays = live, base, {}
        joints = self._match(live.joint_label, base.joint_label)
        live_starts, base_starts = live.joint_qd_start.numpy(), base.joint_qd_start.numpy()
        dofs = []
        for live_joint, base_joint in joints:
            count = int(live_starts[live_joint + 1] - live_starts[live_joint])
            if count == int(base_starts[base_joint + 1] - base_starts[base_joint]):
                dofs += [(int(live_starts[live_joint]) + k, int(base_starts[base_joint]) + k) for k in range(count)]
        self._pairs = {
            "body": self._match(live.body_label, base.body_label),
            "shape": self._match(live.shape_label, base.shape_label),
            "joint": dofs,  # the exported joint arrays are per DOF
        }

    @staticmethod
    def _match(live_labels, base_labels) -> list[tuple[int, int]]:
        first = {}
        for index, label in enumerate(base_labels):
            first.setdefault(label, index)
        return [(index, first[label]) for index, label in enumerate(live_labels) if label in first]

    def __getattr__(self, name: str) -> np.ndarray:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._arrays:
            aligned = getattr(self._live, name).numpy().copy()
            base = getattr(self._base, name).numpy()
            for live_index, base_index in self._pairs[name.split("_", 1)[0]]:
                aligned[live_index] = base[base_index]
            self._arrays[name] = aligned
        return self._arrays[name]
