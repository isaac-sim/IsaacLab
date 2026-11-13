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

from pxr import Sdf, Tf, Usd, UsdPhysics

from isaaclab.sim.usd_export.base_usd_exporter import (
    _DEACTIVATE,
    _DEFINE_PRIM,
    JOINT_STATE_ATTRIBUTES,
    USD_PHYSICS_D6_LIMIT_ATTRIBUTES,
    USD_PHYSICS_JOINT_ATTRIBUTES,
    USD_PHYSICS_MATERIAL_ATTRIBUTES,
    BaseUsdExporter,
    JointAttribute,
    PrimDefinition,
    UsdEdit,
)
from isaaclab.utils.string import string_to_callable

from isaaclab_newton.physics import NewtonSolverCfg
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
    heightfield is imported as its mesh. Bodies and colliders added outside the USD, such as by a world builder
    hook, and collision switched there, are refused.
    """

    # Newton's import builds cables from their USD curves, as Isaac Lab does.
    exports_cables = True

    """
    Operations.
    """

    def compute_diff(self, stage: Usd.Stage) -> list[UsdEdit]:
        """Return the live model values that differ from Newton's import of ``stage``.

        Args:
            stage: The exported stage, reduced to the selected environment.

        Returns:
            Values to author so that Newton's import reproduces the live model.
        """
        # 1. Import the stage alone and align that import with the live model by label.
        physics_cfg = self.scene.sim.cfg.physics
        solver_cfg = physics_cfg.solver_cfg
        model = self.scene.sim.backend_manager.backend.model
        base = self._import_base(stage, model, solver_cfg)
        # 2. Refuse what a world builder hook added, or whose collision it switched, where the stage does not.
        flags, collide = model.shape_flags.numpy(), int(newton.ShapeFlags.COLLIDE_SHAPES)
        added = [
            model.body_label[body]
            for body in self._select_env_entities(model.body_world)
            if body in base.unmatched["body"]
        ]
        added += [
            model.shape_label[shape]
            for shape in self._select_env_entities(model.shape_world)
            if (flags[shape] ^ base.shape_flags[shape]) & collide
            or (shape in base.unmatched["shape"] and flags[shape] & collide)
        ]
        # A free root joint is compared through its body: Isaac Lab's cloner names it after the body, the import not.
        types, parents = model.joint_type.numpy(), model.joint_parent.numpy()
        added += [
            model.joint_label[joint]
            for joint in self._select_env_entities(model.joint_world)
            if joint in base.unmatched["joint"]
            and not (types[joint] == int(newton.JointType.FREE) and parents[joint] == -1)
        ]
        if added:
            raise NotImplementedError(
                f"These bodies, joints and colliders were added, or their collision switched, outside the USD: {added}."
            )
        # 3. Edits per category; the MuJoCo actuators follow the drive gains the joint edits write.
        gravity = model.gravity.numpy()
        gravity = gravity[self.env_id] if gravity.ndim == 2 else gravity
        joint_edits, drive_gains = self._compute_joint_edits(stage, model, base)
        edits = [
            *joint_edits,
            *self._compute_mjc_actuator_edits(stage, model, drive_gains),
            *self._compute_body_edits(stage, model, base),
            *self._compute_shape_edits(stage, model, base),
            *self._compute_gravity_edits(stage, gravity),
            *self._compute_scene_edits(solver_cfg),
        ]
        # 4. Warn about solver settings and collision timing that USD cannot state.
        if not self._uses_solver(solver_cfg, newton.solvers.SolverMuJoCo):
            # Newton reads solver settings from USD only for MuJoCo (its options) and Kamino (constraint gains).
            logger.warning(
                f"USD does not state the settings of {type(solver_cfg).__name__} beyond Kamino's constraint gains; an"
                " import runs the solver its consumer builds, with that solver's defaults."
            )
        # MuJoCo's and Kamino's own contacts are found every solver step; Newton's pipeline once per physics step.
        pipeline = not getattr(solver_cfg, "use_mujoco_contacts", getattr(solver_cfg, "use_collision_detector", False))
        if physics_cfg.num_substeps > 1 and pipeline:
            logger.warning(
                "The file states the solver step rate; an import running Newton's collision pipeline collides every"
                " solver step, while Isaac Lab collides once per physics step."
            )
        return edits

    """
    Joint edits.
    """

    def _compute_joint_edits(
        self, stage: Usd.Stage, model: newton.Model, base: _AlignedBase
    ) -> tuple[list[UsdEdit], dict[str, tuple[float, float]]]:
        """Edits making Newton's import of ``stage`` reproduce the live joints.

        Args:
            stage: The exported stage.
            model: The live model.
            base: The import of ``stage``, aligned with ``model``.

        Returns:
            The edits, and the live stiffness and damping of each single-axis joint whose drive they write, keyed by
            joint path.
        """
        # 1. Live DOF values, their import, and the configured reset state.
        values = self._read_model_arrays(model, DOF_ATTRIBUTES)
        changed = self._compute_changed_masks(values, base)
        starts, types, modes = model.joint_qd_start.numpy(), model.joint_type.numpy(), model.joint_target_mode.numpy()
        coords, (positions, velocities) = model.joint_q_start.numpy(), self._compute_reset_joint_state(model)
        reference = self._get_mjc_limit_reference(model)
        edits, dofs, drive_gains = [], {}, {}
        for joint in self._select_env_entities(model.joint_world):
            # 2. The joint's USD prims and DOF axes; a changed joint of an unsupported type is refused.
            prim = stage.GetPrimAtPath(model.joint_label[joint])
            # Free joints of floating bodies have no USD prim; their pose is exported by the body.
            if types[joint] == int(newton.JointType.FREE) or not prim:
                continue
            units = self._map_joint_dofs_to_prims(stage, model, joint, prim)
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
                # 3. The axes whose drive is written whole: changed, or actuated without a USD drive.
                # Newton's importer actuates exactly the axes with a USD drive.
                actuated = {dof: modes[dof] != int(newton.JointTargetMode.NONE) for dof, _ in axes}
                for dof, axis in axes:
                    if not actuated[dof] and prim.HasAPI(UsdPhysics.DriveAPI, axis):
                        raise NotImplementedError(f"Joint '{path}' has a USD drive the engine does not actuate.")
                # A changed drive is written whole, since Isaac Lab's build can hold drive values other than the USD's;
                # so is the drive of an axis Isaac Lab actuates without a USD drive, which the import would not actuate.
                driven = {
                    name for name, (_, schema, _) in table.items() if schema and schema.startswith("PhysicsDriveAPI")
                }
                drive = {
                    dof: actuated[dof]
                    and (not prim.HasAPI(UsdPhysics.DriveAPI, axis) or any(changed[name][dof] for name in driven))
                    for dof, axis in axes
                }
                if len(axes) == 1 and drive[axes[0][0]]:
                    dof = axes[0][0]
                    drive_gains[path] = (float(values["joint_target_ke"][dof]), float(values["joint_target_kd"][dof]))
                # 4. Property edits, following the import's joint-wide or per-axis precedence.
                for name, (attribute, schema, angular_scale) in table.items():
                    attribute = self._select_limit_gain_attribute(prim, name, attribute)
                    live = (
                        values[name] + reference if name in ("joint_limit_lower", "joint_limit_upper") else values[name]
                    )
                    if "{axis}" not in attribute:
                        if any(changed[name][dof] for dof, _ in axes):
                            edits.append(self._compute_joint_value_edit(path, attribute, live, axes, angular_scale))
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
                # 5. Initial position and velocity where the reset state differs from the import.
                for dof, axis in axes:
                    coord = coords[joint] + dof - starts[joint]
                    state = {
                        "position": (positions[coord], base.joint_q[coord]),
                        "velocity": (velocities[dof], base.joint_qd[dof]),
                    }
                    edits += [
                        self._compute_joint_state_edit(prim, axis, name, live)
                        for name, (live, imported) in state.items()
                        if not self._is_close(live, imported)
                    ]
        # 6. MuJoCo: actuators restoring target modes, and force ranges below a written max force.
        actuators = self._compute_target_mode_actuator_edits(
            model, dofs, values["joint_target_ke"], values["joint_target_kd"], modes
        )
        return edits + self._compute_force_range_widening_edits(stage, edits) + actuators, drive_gains

    def _compute_reset_joint_state(self, model: newton.Model) -> tuple[np.ndarray, np.ndarray]:
        """Joint coordinates and velocities with the scene's articulations at the state Isaac Lab resets them to.

        The engine holds that state only from the first reset on, so it is written to a scratch state.
        """
        state, unwritten = model.state(), []
        for name, articulation in self.scene.articulations.items():
            view, data = articulation.root_view, articulation.data
            order = [articulation.joint_names.index(joint) for joint in articulation.backend_joint_names]
            for values, read, write in (
                (data.default_joint_pos, view.get_dof_positions, view.set_dof_positions),
                (data.default_joint_vel, view.get_dof_velocities, view.set_dof_velocities),
            ):
                shape = read(state).shape
                if shape[-1] != len(order):
                    # Ball joints have more coordinates than DOFs.
                    unwritten.append(name)
                    continue
                write(state, values.torch[:, order].detach().cpu().numpy().reshape(shape))
        if unwritten:
            logger.warning(
                f"The initial joint positions of articulations with ball joints are not written: {unwritten}."
            )
        return state.joint_q.numpy(), state.joint_qd.numpy()

    @staticmethod
    def _get_mjc_limit_reference(model: newton.Model) -> np.ndarray:
        """Per DOF, MuJoCo's joint reference, which Newton's import subtracts from the USD limits."""
        mujoco = getattr(model, "mujoco", None)
        if mujoco is not None and hasattr(mujoco, "dof_ref"):
            return mujoco.dof_ref.numpy().astype(float)
        return np.zeros(model.joint_dof_count)

    @staticmethod
    def _map_joint_dofs_to_prims(
        stage: Usd.Stage, model: newton.Model, joint: int, prim: Usd.Prim
    ) -> list[tuple[Usd.Prim, list[tuple[int, str]], dict]] | None:
        """The USD joints of a model joint, each with its DOFs, their USD axis names and the attribute table.

        Newton merges single-axis USD joints between the same two bodies into one D6 joint named after the
        first; MuJoCo's per-DOF labels name the USD joint of each DOF. None for unsupported joints.
        """
        # 1. Revolute or prismatic: the prim itself, with one axis.
        start, joint_type = int(model.joint_qd_start.numpy()[joint]), int(model.joint_type.numpy()[joint])
        if joint_type == int(newton.JointType.REVOLUTE):
            return [(prim, [(start, "angular")], DOF_ATTRIBUTES)]
        if joint_type == int(newton.JointType.PRISMATIC):
            return [(prim, [(start, "linear")], DOF_ATTRIBUTES)]
        if joint_type != int(newton.JointType.D6):
            return None
        linear, angular = model.joint_dof_dim.numpy()[joint]
        dofs = range(start, start + int(linear + angular))
        # 2. A D6 merged from single-axis USD joints: each DOF's source prim, from MuJoCo's DOF labels.
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            labels = getattr(getattr(model, "mujoco", None), "joint_dof_label", None)
            if labels is None:
                raise NotImplementedError(f"Joint '{prim.GetPath()}' merges USD joints the model does not name.")
            sources = [stage.GetPrimAtPath(labels[dof]) for dof in dofs]
            return [
                (source, [(dof, "angular" if source.IsA(UsdPhysics.RevoluteJoint) else "linear")], DOF_ATTRIBUTES)
                for dof, source in zip(dofs, sources)
            ]
        # 3. A rotational D6: rotX, rotY or rotZ from each DOF's joint axis.
        if linear:
            return None
        axes = model.joint_axis.numpy()
        return [(prim, [(dof, "rot" + "XYZ"[int(np.argmax(np.abs(axes[dof])))]) for dof in dofs], D6_DOF_ATTRIBUTES)]

    @classmethod
    def _select_limit_gain_attribute(cls, prim: Usd.Prim, name: str, attribute: str) -> str:
        """The attribute a limit gain is written to: the joint-wide one when stated, which Newton reads first."""
        joint_wide = JOINT_WIDE_LIMIT_GAINS.get(name)
        if "{axis}" in attribute and joint_wide and cls._get_authored_value(prim, joint_wide) is not None:
            return joint_wide
        return attribute

    @classmethod
    def _compute_joint_value_edit(
        cls, path: str, attribute: str, values: np.ndarray, axes: list[tuple[int, str]], angular_scale: float
    ) -> UsdEdit:
        """Edit writing a value USD states once per joint, which all its axes must share."""
        live = [float(values[dof]) for dof, _ in axes]
        if not cls._is_close(live, live[0]):
            raise NotImplementedError(f"Joint '{path}' has a different '{attribute}' per axis; USD states one.")
        return UsdEdit(path, attribute, live[0] * (1.0 if axes[0][1] == "linear" else angular_scale))

    @classmethod
    def _compute_joint_state_edit(cls, prim: Usd.Prim, axis: str, name: str, value: float) -> UsdEdit:
        """Edit stating a joint axis's initial position or velocity, on the attribute Newton's import reads first."""
        attribute, schema, angular_scale = JOINT_STATE_ATTRIBUTES[name]
        value = float(value) * (1.0 if axis == "linear" else angular_scale)
        path, own = str(prim.GetPath()), _JOINT[f"{axis}_{name}"].name
        if cls._get_authored_value(prim, own) is not None:
            return UsdEdit(path, own, value)
        return UsdEdit(path, attribute.format(axis=axis), value, schema=schema.format(axis=axis))

    @classmethod
    def _compute_force_range_widening_edits(cls, stage: Usd.Stage, edits: list[UsdEdit]) -> list[UsdEdit]:
        """Edits widening a MuJoCo joint force range that would clamp an exported drive force limit.

        Newton's import keeps the tighter of a drive's max force and the joint's ``mjc:actuatorfrcrange``.
        """
        widened = []
        for edit in edits:
            if not edit.attribute.endswith("physics:maxForce"):
                continue
            prim = stage.GetPrimAtPath(edit.path)
            bounds = [cls._get_authored_value(prim, f"mjc:actuatorfrcrange:{end}") for end in ("min", "max")]
            if any(bound is not None for bound in bounds) and max(abs(b or 0.0) for b in bounds) < edit.value:
                widened += [
                    UsdEdit(edit.path, "mjc:actuatorfrcrange:min", -float(edit.value)),
                    UsdEdit(edit.path, "mjc:actuatorfrcrange:max", float(edit.value)),
                ]
        return widened

    """
    Actuators.
    """

    def _compute_target_mode_actuator_edits(
        self, model: newton.Model, dofs: dict[int, str], target_ke, target_kd, modes
    ) -> list[UsdEdit]:
        """Edits making Newton's USD import reproduce each joint's target mode, warning where it cannot.

        Workaround until USD states a target mode: Newton's import infers the mode from the drive gains, so a joint
        with both gains runs in position mode, while Isaac Lab runs it in position-velocity mode. A MuJoCo velocity
        actuator targeting a single-axis joint makes the import use position-velocity mode (gain: the damping).
        """
        # 1. The solver, and the DOFs whose velocity actuator the import already converted.
        position_velocity = int(newton.JointTargetMode.POSITION_VELOCITY)
        solver_cfg = self.scene.sim.cfg.physics.solver_cfg
        mujoco = self._uses_solver(solver_cfg, newton.solvers.SolverMuJoCo)
        converted = self._get_converted_velocity_dofs(model) if mujoco else set()
        axes = Counter(dofs.values())
        root = self.scene.env_prim_paths[self.env_id]
        edits, differ = [], set()
        for dof, path in dofs.items():
            # 2. The mode the import infers from the gains, against the live mode.
            inferred = newton.JointTargetMode.from_gains(float(target_ke[dof]), float(target_kd[dof]), has_drive=True)
            if modes[dof] == int(newton.JointTargetMode.NONE) or modes[dof] == int(inferred) or dof in converted:
                continue
            if not (mujoco and modes[dof] == position_velocity and axes[path] == 1):
                differ.add(path)
                continue
            # 3. A single-axis MuJoCo joint in position-velocity mode gets a velocity actuator; others are reported.
            damping = float(target_kd[dof])
            name = Tf.MakeValidIdentifier(path.removeprefix(root + "/"))
            values = {
                "mjc:biasType": (Sdf.ValueTypeNames.Token, "affine"),
                "mjc:gainPrm": (Sdf.ValueTypeNames.DoubleArray, [damping] + [0.0] * 9),
                "mjc:biasPrm": (Sdf.ValueTypeNames.DoubleArray, [0.0, 0.0, -damping] + [0.0] * 7),
            }
            actuator = PrimDefinition("MjcActuator", {"mjc:target": [path]}, values)
            edits.append(UsdEdit(f"{root}/VelocityActuators/{name}", _DEFINE_PRIM, actuator))
        # Of Newton's solvers, only MuJoCo and Kamino apply a target mode.
        if differ and (mujoco or self._uses_solver(solver_cfg, newton.solvers.SolverKamino)):
            logger.warning(
                f"USD cannot state the target mode of these joints; an import infers another: {sorted(differ)}."
            )
        return edits

    @staticmethod
    def _get_converted_velocity_dofs(model: newton.Model) -> set[int]:
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

    @classmethod
    def _compute_mjc_actuator_edits(
        cls, stage: Usd.Stage, model: newton.Model, drive_gains: dict[str, tuple[float, float]]
    ) -> list[UsdEdit]:
        """Edits giving the MuJoCo actuators of joints whose drive the export writes the joint's live gain.

        Workaround: Newton's import turns a MuJoCo position or velocity actuator into the targeted joint's
        drive, setting the joint's stiffness or damping to the actuator's gain. Only the actuators the import
        converted this way are edited; Newton keeps the others as separate actuators. An actuator whose gain
        becomes zero is deactivated, since the import would keep it as a separate actuator.

        Args:
            stage: The exported stage.
            model: The live model, which records the actuators Newton's import converted.
            drive_gains: Live stiffness and damping of the joints whose drive the export writes, keyed by joint path.

        Returns:
            The edits.
        """
        # 1. The actuators Newton's import converted to joint targets.
        mujoco = getattr(model, "mujoco", None)
        if mujoco is None or not hasattr(mujoco, "ctrl_source"):
            return []
        joint_target = int(newton.solvers.SolverMuJoCo.CtrlSource.JOINT_TARGET)
        converted = {
            label for label, source in zip(mujoco.actuator_label, mujoco.ctrl_source.numpy()) if source == joint_target
        }
        edits, deactivated = [], []
        for prim in [prim for prim in stage.Traverse() if str(prim.GetPath()) in converted]:
            # 2. Of those, the ones targeting a joint whose drive the export writes.
            targets = [str(target) for target in prim.GetRelationship("mjc:target").GetTargets()]
            if len(targets) != 1 or targets[0] not in drive_gains:
                continue
            # 3. The live gain of the actuator's kind; unset parameters take MuJoCo's defaults.
            gain = list(prim.GetAttribute("mjc:gainPrm").Get() or [1.0] + [0.0] * 9)
            bias = list(prim.GetAttribute("mjc:biasPrm").Get() or [0.0] * 10)
            # A position actuator biases by -kp * qpos, a velocity actuator by -kv * qvel.
            index = 1 if bias[1] != 0.0 else 2
            stiffness, damping = drive_gains[targets[0]]
            value = stiffness if index == 1 else damping
            if cls._is_close(gain[0], value):
                continue
            # 4. Deactivate an actuator whose gain is not positive; otherwise write the gain and its bias.
            if value <= 0.0:
                edits.append(UsdEdit(str(prim.GetPath()), _DEACTIVATE, None))
                deactivated.append(str(prim.GetPath()))
                continue
            gain[0], bias[index] = value, -value
            path = str(prim.GetPath())
            edits += [UsdEdit(path, "mjc:gainPrm", gain, "double[]"), UsdEdit(path, "mjc:biasPrm", bias, "double[]")]
        if deactivated:
            logger.info(f"MuJoCo actuators replaced by the exported zero-gain joint drives: {deactivated}.")
        return edits

    """
    Body and shape edits.
    """

    def _compute_body_edits(self, stage: Usd.Stage, model: newton.Model, base: _AlignedBase) -> list[UsdEdit]:
        """Edits of body mass properties, root-body poses and the world-side anchors of root joints."""
        # 1. Mass, COM and inertia that differ from the import.
        values = self._read_model_arrays(model, ("body_mass", "body_com", "body_inertia"))
        changed = self._compute_changed_masks(values, base)
        mass, com, inertia = values["body_mass"], values["body_com"], values["body_inertia"]
        edits = []
        for body in self._select_env_entities(model.body_world):
            prim = stage.GetPrimAtPath(model.body_label[body])
            properties = {name.removeprefix("body_") for name, rows in changed.items() if rows[body]}
            if prim and properties:
                edits += self._compute_mass_edits(prim, float(mass[body]), com[body], inertia[body], properties)
        # 2. Root bodies and world anchors where initialization placed them; the joint states place the other bodies.
        body_q, anchors = self.scene.sim.backend_manager.backend.state_0.body_q.numpy(), model.joint_X_p.numpy()
        parents, children, types = model.joint_parent.numpy(), model.joint_child.numpy(), model.joint_type.numpy()
        for joint in self._select_env_entities(model.joint_world):
            if parents[joint] != -1:
                continue
            body = stage.GetPrimAtPath(model.body_label[children[joint]])
            if body:
                edits.append(self._compute_pose_edit(body, body_q[children[joint]]))
            prim = stage.GetPrimAtPath(model.joint_label[joint])
            if types[joint] != int(newton.JointType.FREE) and prim:
                edits.append(self._compute_value_edit(prim, "physics:localPos0", anchors[joint][:3], "point3f"))
                edits.append(self._compute_value_edit(prim, "physics:localRot0", anchors[joint][3:7], "quatf"))
        return [edit for edit in edits if edit is not None]

    def _compute_shape_edits(self, stage: Usd.Stage, model: newton.Model, base: _AlignedBase) -> list[UsdEdit]:
        """Collider margin and gap edits and physics-material edits reproducing the live shapes."""
        # 1. Live shape and material values against the import.
        values, material = (
            self._read_model_arrays(model, SHAPE_ATTRIBUTES),
            self._read_model_arrays(model, MATERIAL_ATTRIBUTES),
        )
        changed = self._compute_changed_masks(values, base) | self._compute_changed_masks(material, base)
        flags, collide = model.shape_flags.numpy(), int(newton.ShapeFlags.COLLIDE_SHAPES)
        edits, changed_materials, complete, generated = [], {}, {}, []
        physics = self._get_collider_physics_materials(stage)
        bound = {}
        # 2. Collider edits; each collider's material binding, live material values and changed ones.
        for shape in self._select_env_entities(model.shape_world):
            # Sensor sites and visual shapes do not collide.
            if not int(flags[shape]) & collide:
                continue
            prim = stage.GetPrimAtPath(model.shape_label[shape])
            if not prim:
                # Shapes Newton's import generates, such as cable segments, have no prim to state a value on.
                if any(rows[shape] for rows in changed.values()):
                    generated.append(model.shape_label[shape])
                continue
            edits += [
                UsdEdit(str(prim.GetPath()), attribute, float(values[name][shape]))
                for name, attribute in SHAPE_ATTRIBUTES.items()
                if changed[name][shape]
            ]
            path = str(prim.GetPath())
            bound[path] = physics.get(path)
            complete[path] = {a: float(material[n][shape]) for n, a in MATERIAL_ATTRIBUTES.items()}
            updated = {a: complete[path][a] for n, a in MATERIAL_ATTRIBUTES.items() if changed[n][shape]}
            if updated:
                changed_materials[path] = updated
        # 3. Every collider sharing a material with a changed value submits it, so the material edits check they agree.
        affected: dict[str, set[str]] = {}
        for path, updated in changed_materials.items():
            if bound[path]:
                affected.setdefault(bound[path], set()).update(updated)
        submitted = changed_materials | {
            path: {name: value for name, value in complete[path].items() if name in affected[bound[path]]}
            for path in complete
            if bound[path] in affected
        }
        # 4. A collider without a material gets a new one, which states every value an import reads from it.
        submitted |= {path: complete[path] for path in submitted if not bound[path]}
        edits = [edit for edit in edits if edit is not None]
        # 5. Shapes Newton's import generates keep its contact values.
        if generated:
            logger.warning(
                "Newton's import gives the shapes it generates its own contact values, which these do not hold:"
                f" {generated}."
            )
        # 6. Material edits; an unset contact gain of a material falls back to Newton's builder default.
        upstream = newton.ModelBuilder.ShapeConfig()
        fallbacks = {
            attribute: getattr(upstream, array.removeprefix("shape_material_"))
            for array, attribute in MATERIAL_ATTRIBUTES.items()
        }
        return edits + self._compute_material_edits(stage, submitted, fallbacks, physics)

    """
    Solver settings.
    """

    def _compute_scene_edits(self, solver_cfg) -> list[UsdEdit]:
        """Edits stating the solver step rate and the solver options Newton's importer reads from the physics scene.

        Isaac Lab steps the solver ``num_substeps`` times per physics step; USD states one rate, so the file states
        the solver's, which Newton's import returns as ``physics_dt``. An option equal to the value Newton's import
        uses when the scene does not state it is not written.
        """
        # 1. The solver step rate.
        path = self.scene.physics_scene_path
        rate = self.scene.sim.cfg.physics.num_substeps / self.scene.sim.get_physics_dt()
        if not math.isclose(rate, round(rate), rel_tol=1e-9):
            raise ValueError(f"Step size 1/{rate} s is not a whole number of steps per second, as USD states it.")
        edits = [UsdEdit(path, _SCENE["time_steps_per_second"].name, round(rate), "int", "NewtonSceneAPI")]
        # 2. MuJoCo options and flags that differ from what the import uses when the scene does not state them.
        types = {bool: "bool", int: "int", float: "float", str: "token"}
        for name, attribute in self._get_mujoco_option_attributes(solver_cfg).items():
            value = getattr(solver_cfg, name)
            if name in MUJOCO_FLAG_FIELDS and MUJOCO_FLAG_FIELDS[name][1] and value is not None:
                value = not value
            if value is not None and not self._is_import_default(attribute, value):
                edits.append(UsdEdit(path, attribute.usd_attribute_name, value, types[type(value)]))
        # 3. Kamino's constraint stabilization gains that differ from the import's.
        if self._uses_solver(solver_cfg, newton.solvers.SolverKamino):
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

    @classmethod
    def _get_mujoco_option_attributes(cls, solver_cfg) -> dict[str, newton.ModelBuilder.CustomAttribute]:
        """Solver configuration field -> Newton's physics-scene attribute, for options the solver reads from the model.

        Only MuJoCo reads its options from the model; Newton registers their USD attributes.
        """
        if not cls._uses_solver(solver_cfg, newton.solvers.SolverMuJoCo):
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
    def _is_import_default(attribute, value) -> bool:
        """Whether Newton's import resolves ``attribute`` to ``value`` when the USD does not state it."""
        if attribute.default is None:
            return False
        if attribute.usd_value_transformer is not None:
            value = attribute.usd_value_transformer(value, None)
        if isinstance(value, float) or isinstance(attribute.default, float):
            return bool(np.isclose(value, attribute.default, rtol=1e-6, atol=0.0))
        return value == attribute.default

    @staticmethod
    def _uses_solver(solver_cfg: NewtonSolverCfg | None, solver_class: type) -> bool:
        """Whether the configured solver runs ``solver_class`` (and so reads its custom attributes)."""
        if solver_cfg is None:
            return False
        # The scene has built this solver, so its class type is a Newton solver.
        solver = solver_cfg.class_type
        solver = string_to_callable(solver) if isinstance(solver, str) else solver
        return any(issubclass(c, solver_class) for c in solver.solver_classes(solver_cfg))

    """
    Model values.
    """

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

    def _select_env_entities(self, worlds) -> np.ndarray:
        return np.flatnonzero(np.isin(worlds.numpy(), (-1, self.env_id)))

    @staticmethod
    def _read_model_arrays(model, names) -> dict[str, np.ndarray]:
        return {name: getattr(model, name).numpy() for name in names}

    @staticmethod
    def _compute_changed_masks(values: dict[str, np.ndarray], base: _AlignedBase) -> dict[str, np.ndarray]:
        """Per entity, whether a live model array differs from the value the import took from the stage."""
        changed = {}
        for name, live in values.items():
            # An explicit width also fits a model without any such entity.
            shape = (len(live), int(np.prod(live.shape[1:])))
            built = np.asarray(getattr(base, name), dtype=float).reshape(shape)
            changed[name] = ~np.isclose(live.reshape(shape), built, rtol=1e-5, atol=1e-7).all(axis=1)
        return changed


class _AlignedBase:
    """The arrays of a base model indexed like a live model's entities, matched by body, shape and joint label.

    The base, an import of the exported environment alone, orders entities differently from the live model even
    with one environment, and holds none of the other environments. An entity of the live model that the base does
    not have keeps its live value, so it never counts as changed; :attr:`unmatched` holds these bodies, shapes and
    joints, a joint also when the base gives it another DOF count.
    """

    def __init__(self, live: newton.Model, base: newton.Model):
        self._live, self._base, self._arrays = live, base, {}
        joints = self._match_labels(live.joint_label, base.joint_label)
        self._pairs = {
            "body": self._match_labels(live.body_label, base.body_label),
            "shape": self._match_labels(live.shape_label, base.shape_label),
            "joint": self._match_entries(joints, live.joint_qd_start, base.joint_qd_start),  # per DOF
            "joint_q": self._match_entries(joints, live.joint_q_start, base.joint_q_start),  # per coordinate
        }
        self.unmatched = {
            kind: set(range(len(getattr(live, f"{kind}_label")))) - {index for index, _ in self._pairs[kind]}
            for kind in ("body", "shape")
        }
        live_starts, base_starts = live.joint_qd_start.numpy(), base.joint_qd_start.numpy()
        alike = {
            live_joint
            for live_joint, base_joint in joints
            if live_starts[live_joint + 1] - live_starts[live_joint]
            == base_starts[base_joint + 1] - base_starts[base_joint]
        }
        self.unmatched["joint"] = set(range(len(live.joint_label))) - alike

    @staticmethod
    def _match_labels(live_labels, base_labels) -> list[tuple[int, int]]:
        indices = {}
        for index, label in enumerate(base_labels):
            indices.setdefault(label, []).append(index)
        pairs = []
        for index, label in enumerate(live_labels):
            if len(indices.get(label, ())) > 1:
                raise ValueError(f"An import of the USD has {len(indices[label])} entities labeled '{label}'.")
            if label in indices:
                pairs.append((index, indices[label][0]))
        return pairs

    @staticmethod
    def _match_entries(joints: list[tuple[int, int]], live_starts, base_starts) -> list[tuple[int, int]]:
        """Pairs of the per-joint entries (DOFs or coordinates) of matched joints with equal entry counts."""
        live_starts, base_starts = live_starts.numpy(), base_starts.numpy()
        pairs = []
        for live_joint, base_joint in joints:
            count = int(live_starts[live_joint + 1] - live_starts[live_joint])
            if count == int(base_starts[base_joint + 1] - base_starts[base_joint]):
                pairs += [(int(live_starts[live_joint]) + k, int(base_starts[base_joint]) + k) for k in range(count)]
        return pairs

    def __getattr__(self, name: str) -> np.ndarray:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._arrays:
            aligned = getattr(self._live, name).numpy().copy()
            base = getattr(self._base, name).numpy()
            for live_index, base_index in self._pairs[name if name in self._pairs else name.split("_", 1)[0]]:
                aligned[live_index] = base[base_index]
            self._arrays[name] = aligned
        return self._arrays[name]
