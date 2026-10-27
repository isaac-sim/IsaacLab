# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton USD export: every live model value that Isaac Lab set after building the model from the stage."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter

import newton
import numpy as np
from newton.usd import PrimType, SchemaResolverNewton

from pxr import Sdf, Tf, Usd, UsdPhysics, UsdShade

from isaaclab.sim.usd_export.base_usd_exporter import (
    USD_PHYSICS_JOINT_ATTRIBUTES,
    USD_PHYSICS_MATERIAL_ATTRIBUTES,
    BaseUsdExporter,
    UsdEdit,
)
from isaaclab.utils.string import string_to_callable

from isaaclab_newton.physics import NewtonBuilderCfg, NewtonCfg, NewtonManager, NewtonShapeCfg, NewtonSolverCfg

logger = logging.getLogger(__name__)

SETTINGS_KEY = "isaaclab:newton"
"""Layer metadata holding what Newton's importer does not read from USD: the step size and the solver."""

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
    "joint_armature": (_JOINT["armature"].name, None, 1.0),
    "joint_friction": (_JOINT["friction"].name, None, 1.0),
    "joint_damping": (_JOINT["damping"].name, None, math.radians(1.0)),
    "joint_velocity_limit": (_JOINT["velocity_limit"].name, None, math.degrees(1.0)),
    "joint_limit_ke": (_JOINT["limit_ke"].name, None, math.radians(1.0)),
    "joint_limit_kd": (_JOINT["limit_kd"].name, None, math.radians(1.0)),
}
"""Model DOF array -> USD joint attribute, required API schema, and scale to USD units for angular DOFs.

An attribute without ``{axis}`` holds one value for the whole joint.
"""

D6_DOF_ATTRIBUTES = {
    **DOF_ATTRIBUTES,
    "joint_limit_lower": ("limit:{axis}:physics:low", "PhysicsLimitAPI:{axis}", math.degrees(1.0)),
    "joint_limit_upper": ("limit:{axis}:physics:high", "PhysicsLimitAPI:{axis}", math.degrees(1.0)),
    "joint_limit_ke": (_JOINT["limit_rotX_ke"].name.replace("rotX", "{axis}"), None, math.radians(1.0)),
    "joint_limit_kd": (_JOINT["limit_rotX_kd"].name.replace("rotX", "{axis}"), None, math.radians(1.0)),
}
""":data:`DOF_ATTRIBUTES` of a rotational D6 joint, whose axes are ``rotX``, ``rotY`` and ``rotZ``."""

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

SHAPE_CONTACT_ATTRIBUTES = {_MATERIAL[key].name: _SHAPE[key].name for key in ("ke", "kd", "kf", "ka")}
"""Material contact gain -> collider attribute, used by colliders without a physics material."""

SHAPE_DEFAULTS = {
    "margin": ("shape_margin", _SHAPE["margin"].name),
    "gap": ("shape_gap", _SHAPE["gap"].name),
    "ke": ("shape_material_ke", _SHAPE["ke"].name),
    "kd": ("shape_material_kd", _SHAPE["kd"].name),
    "mu": ("shape_material_mu", None),
}
""":class:`NewtonShapeCfg` field -> model shape array and the collider attribute that states it.

Workaround until Newton's USD schema has scene-level shape defaults: Isaac Lab applies these defaults
when it builds the model, so an exported collider using one states it explicitly. Friction can only be
stated by a physics material, so a collider without one that uses Isaac Lab's friction is not exported.
"""


class UsdExporter(BaseUsdExporter):
    """Export a Newton scene environment: its USD plus every live model value Isaac Lab set after the build.

    The baseline is Isaac Lab's model builder, which holds the values of the stage before any asset writes
    its configuration to the model. A value is written where the live model differs from it.

    The file is meant for Newton's USD import (:meth:`newton.ModelBuilder.add_usd`) with the solver's custom
    attributes registered and the Newton, PhysX and, for MuJoCo, MuJoCo schema resolvers. USD cannot yet
    state Newton's substeps and collision pipeline settings, nor a joint target mode other than the one the
    import infers from the gains; the export warns about these. Terrain that Isaac Lab collides as a
    heightfield is imported as its mesh.
    """

    def compute_diff(self, stage: Usd.Stage) -> list[UsdEdit]:
        """Return the live model values that differ from the build of ``stage``; record the step size and solver.

        Args:
            stage: The exported stage, reduced to the selected environment.

        Returns:
            Values to author so that :meth:`load_usd` reproduces the live model.
        """
        physics_cfg = self.scene.sim.cfg.physics
        solver_cfg = physics_cfg.solver_cfg
        model = self.scene.sim.backend_manager.backend.model
        builder = self.scene.sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=physics_cfg))
        gravity = model.gravity.numpy()
        gravity = gravity[self.env_id] if gravity.ndim == 2 else gravity
        joint_edits, drives = self._joint_edits(stage, model, builder)
        edits = [
            *joint_edits,
            *self._mjc_actuator_edits(stage, model, drives),
            *self._body_edits(stage, model, builder),
            *self._shape_edits(stage, model, builder, physics_cfg.default_shape_cfg),
            *self._gravity_edits(stage, gravity),
            *self._scene_edits(solver_cfg),
        ]
        settings = {
            "dt": self.scene.sim.get_physics_dt(),
            "solver_type": f"{type(solver_cfg).__module__}:{type(solver_cfg).__name__}",
            "solver": solver_cfg.to_dict(),
        }
        layer = stage.GetRootLayer()
        layer.customLayerData = {**layer.customLayerData, SETTINGS_KEY: json.dumps(settings)}
        # MuJoCo's and Kamino's own contacts are found every solver step; Newton's pipeline once per physics step.
        pipeline = not getattr(solver_cfg, "use_mujoco_contacts", getattr(solver_cfg, "use_collision_detector", False))
        if physics_cfg.num_substeps > 1 and pipeline:
            logger.warning(
                "The file states the solver step rate; an import running Newton's collision pipeline collides every"
                " solver step, while Isaac Lab collides once per physics step."
            )
        return edits

    @classmethod
    def load_usd(
        cls, path: str, device: str = "cpu", allow_unsafe_torch_checkpoints: bool = False
    ) -> tuple[newton.Model, newton.solvers.SolverBase | None, float | None]:
        """Load a USD file exported from a Newton scene.

        Referenced remote assets are downloaded first, as Isaac Lab does when spawning them.

        Args:
            path: Exported USD file.
            device: Device of the model and solver. Defaults to ``"cpu"``.
            allow_unsafe_torch_checkpoints: Load the torch checkpoints the file's actuators reference, such as
                actuator networks. Loading one can run arbitrary code; enable it only for trusted files.
                Defaults to False.

        Returns:
            The model, the solver configured as in training (None if the file records none), and the
            physics step size [s] (None if the file records none).
        """
        stage = cls.open_stage(path)
        settings = json.loads(stage.GetRootLayer().customLayerData.get(SETTINGS_KEY, "{}"))
        solver_cfg = None
        if "solver_type" in settings:
            # The file names the class to call; accept only solver configurations.
            solver_type = string_to_callable(settings["solver_type"])
            if not (isinstance(solver_type, type) and issubclass(solver_type, NewtonSolverCfg)):
                raise ValueError(f"'{settings['solver_type']}' in '{path}' is not a Newton solver configuration.")
            solver_cfg = solver_type()
            solver_cfg.from_dict(settings["solver"])
        force = settings.get("force_position_velocity_actuation", False)
        model = cls.import_model(stage, solver_cfg, device, force, allow_unsafe_torch_checkpoints)
        solver = cls._manager(solver_cfg)._create_solver(model, solver_cfg) if solver_cfg is not None else None
        return model, solver, settings.get("dt")

    @classmethod
    def import_model(
        cls,
        stage: Usd.Stage,
        solver_cfg=None,
        device: str = "cpu",
        force_position_velocity_actuation: bool = False,
        allow_unsafe_torch_checkpoints: bool = False,
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
            allow_unsafe_torch_checkpoints: Load the torch checkpoints the stage's actuators reference; see
                :meth:`load_usd`. Defaults to False.

        Returns:
            The finalized model.
        """
        manager = cls._manager(solver_cfg)(NewtonCfg(solver_cfg=solver_cfg))
        builder = newton.ModelBuilder()
        manager._register_builder_attributes(builder, solver_cfg)
        # Tagged terrains become heightfields, as in Isaac Lab; the importer does not read their colliders.
        terrains = manager._inject_terrain_heightfields(stage, builder, root_paths=("/",), device=device)
        for shape, label in enumerate(builder.shape_label):
            cls._apply_authored_contacts(stage.GetPrimAtPath(label), builder, shape)
        builder.add_usd(
            stage,
            schema_resolvers=manager._get_usd_import_schema_resolvers(),
            ignore_paths=terrains,
            force_position_velocity_actuation=force_position_velocity_actuation,
            allow_unsafe_torch_checkpoints=allow_unsafe_torch_checkpoints,
        )
        return builder.finalize(device=device)

    def _selected(self, worlds) -> np.ndarray:
        return np.flatnonzero(np.isin(worlds.numpy(), (-1, self.env_id)))

    def _joint_edits(
        self, stage: Usd.Stage, model: newton.Model, builder: newton.ModelBuilder
    ) -> tuple[list[UsdEdit], dict[str, tuple[float, float]]]:
        values = self._arrays(model, DOF_ATTRIBUTES)
        changed = self._changed(values, builder)
        starts, types, modes = model.joint_qd_start.numpy(), model.joint_type.numpy(), model.joint_target_mode.numpy()
        edits, dofs, unsupported, drives = [], {}, [], {}
        for joint in self._selected(model.joint_world):
            prim = stage.GetPrimAtPath(model.joint_label[joint])
            # Free joints of floating bodies have no USD prim; their pose is exported by the body.
            if types[joint] == int(newton.JointType.FREE) or not prim:
                continue
            units = self._joint_prims(stage, model, joint, prim)
            if units is None:
                if starts[joint + 1] > starts[joint]:
                    unsupported.append(model.joint_label[joint])
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
                    if "{axis}" not in attribute:
                        if any(changed[name][dof] for dof, _ in axes):
                            edits.append(self._joint_value_edit(path, attribute, values[name], axes, angular_scale))
                        continue
                    for dof, axis in axes:
                        if (changed[name][dof] and (name not in driven or actuated[dof])) or (
                            drive[dof] and name in driven
                        ):
                            scale = 1.0 if axis == "linear" else angular_scale
                            value = float(values[name][dof]) * scale
                            edits.append(
                                UsdEdit(
                                    path, attribute.format(axis=axis), value, schema=schema and schema.format(axis=axis)
                                )
                            )
        if unsupported:
            logger.warning(f"Joints other than revolute, prismatic and rotational D6 are not exported: {unsupported}.")
        self._target_mode_actuators(stage, model, dofs, values["joint_target_ke"], values["joint_target_kd"], modes)
        return edits, drives

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
            if prim and any(rows[body] for rows in changed.values()):
                edits += self._mass_edits(prim, float(mass[body]), com[body], inertia[body])
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

    def _shape_edits(
        self, stage: Usd.Stage, model: newton.Model, builder: newton.ModelBuilder, defaults: NewtonShapeCfg
    ) -> list[UsdEdit]:
        values, material = self._arrays(model, SHAPE_ATTRIBUTES), self._arrays(model, MATERIAL_ATTRIBUTES)
        changed = self._changed(values, builder) | self._changed(material, builder)
        # Isaac Lab's shape defaults that Newton's import does not apply (see SHAPE_DEFAULTS).
        upstream = newton.ModelBuilder.ShapeConfig()
        lab_defaults = {
            name: getattr(defaults, name)
            for name in SHAPE_DEFAULTS
            if getattr(upstream, name) is None or not self._close(getattr(defaults, name), getattr(upstream, name))
        }
        flags, collide = model.shape_flags.numpy(), int(newton.ShapeFlags.COLLIDE_SHAPES)
        edits, materials, complete, bound = [], {}, {}, {}
        for shape in self._selected(model.shape_world):
            prim = stage.GetPrimAtPath(model.shape_label[shape])
            # Sensor sites and visual shapes do not collide; generated shapes have no USD prim.
            if not prim or not int(flags[shape]) & collide:
                continue
            for name, value in lab_defaults.items():
                array, attribute = SHAPE_DEFAULTS[name]
                if changed[array][shape] or not self._close(getattr(builder, array)[shape], value):
                    continue
                if attribute is not None:
                    edits.append(self._value_edit(prim, attribute, float(value)))
                elif not UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]:
                    raise NotImplementedError(f"Collider '{prim.GetPath()}' needs a physics material for its friction.")
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
        # A changed material is checked against every collider bound to it, with complete values.
        affected = {str(bound[path].GetPath()) for path in materials if bound[path]}
        materials |= {
            path: complete[path] for path in complete if bound[path] and str(bound[path].GetPath()) in affected
        }
        edits = [edit for edit in edits if edit is not None]
        return edits + self._material_edits(
            stage, materials, lambda c, v: self._collider_contact_edits(stage.GetPrimAtPath(c), v)
        )

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
        the solver's, which Newton's import returns as ``physics_dt``.
        """
        path = self.scene.physics_scene_path
        rate = self.scene.sim.cfg.physics.num_substeps / self.scene.sim.get_physics_dt()
        if not math.isclose(rate, round(rate), rel_tol=1e-9):
            raise ValueError(f"Step size 1/{rate} s is not a whole number of steps per second, as USD states it.")
        edits = [UsdEdit(path, _SCENE["time_steps_per_second"].name, round(rate), "int", "NewtonSceneAPI")]
        types = {bool: "bool", int: "int", float: "float", str: "token"}
        for name, attribute in self._scene_solver_options(solver_cfg).items():
            value = getattr(solver_cfg, name)
            if value is not None:
                edits.append(UsdEdit(path, attribute, value, types[type(value)]))
        return edits

    @classmethod
    def _scene_solver_options(cls, solver_cfg) -> dict[str, str]:
        """Solver configuration field -> physics-scene attribute, for options the solver reads from the model.

        Only MuJoCo reads its options from the model; Newton registers their USD attributes.
        """
        manager = cls._manager(solver_cfg)
        if not manager._registers_builder_attributes_from_solver(newton.solvers.SolverMuJoCo):
            return {}
        builder = newton.ModelBuilder()
        manager._register_builder_attributes(builder, solver_cfg)
        scene = (newton.Model.AttributeFrequency.ONCE, newton.Model.AttributeFrequency.WORLD)
        attributes = {
            attribute.name: attribute.usd_attribute_name
            for attribute in builder.custom_attributes.values()
            if attribute.namespace == "mujoco" and attribute.frequency in scene and attribute.usd_attribute_name
        }
        return {name: attributes[name] for name in solver_cfg.to_dict() if name in attributes}

    @staticmethod
    def _apply_authored_contacts(prim: Usd.Prim, builder: newton.ModelBuilder, shape: int):
        """Apply the contact values authored on a collider, or its physics material, to a builder shape."""
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        contacts = {
            array: SHAPE_CONTACT_ATTRIBUTES[attribute]
            for array, attribute in MATERIAL_ATTRIBUTES.items()
            if attribute in SHAPE_CONTACT_ATTRIBUTES
        }
        sources = [(prim, SHAPE_ATTRIBUTES), (prim, contacts)]
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
        manager = string_to_callable(manager) if isinstance(manager, str) else manager
        # An exported file sets the class type; accept only Newton managers.
        if not (isinstance(manager, type) and issubclass(manager, NewtonManager)):
            raise ValueError(f"'{solver_cfg.class_type}' is not a Newton physics manager.")
        return manager

    def _target_mode_actuators(
        self, stage: Usd.Stage, model: newton.Model, dofs: dict[int, str], target_ke, target_kd, modes
    ):
        """Make Newton's USD import reproduce each joint's target mode, and warn where it cannot.

        Workaround until USD states a target mode: Newton's import infers the mode from the drive gains, so a joint
        with both gains runs in position mode, while Isaac Lab runs it in position-velocity mode. A MuJoCo velocity
        actuator targeting a single-axis joint makes the import use position-velocity mode (gain: the damping).
        """
        position_velocity = int(newton.JointTargetMode.POSITION_VELOCITY)
        mujoco = self._manager(self.scene.sim.cfg.physics.solver_cfg)._registers_builder_attributes_from_solver(
            newton.solvers.SolverMuJoCo
        )
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
            actuator.CreateRelationship("mjc:target").SetTargets([path])
            actuator.CreateAttribute("mjc:biasType", Sdf.ValueTypeNames.Token).Set("affine")
            actuator.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.DoubleArray).Set([damping] + [0.0] * 9)
            actuator.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.DoubleArray).Set(
                [0.0, 0.0, -damping] + [0.0] * 7
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

    @classmethod
    def _collider_contact_edits(cls, prim: Usd.Prim, values: dict[str, float]) -> list[UsdEdit]:
        """Contact gains of a collider without a physics material, which Newton reads from the collider.

        Its other material values, such as friction, can only be stated by a physics material.
        """
        unstated = sorted(set(values) - set(SHAPE_CONTACT_ATTRIBUTES))
        if unstated:
            raise NotImplementedError(f"Collider '{prim.GetPath()}' needs a physics material for {unstated}.")
        edits = (
            cls._value_edit(prim, SHAPE_CONTACT_ATTRIBUTES[name], value)
            for name, value in values.items()
            if name in SHAPE_CONTACT_ATTRIBUTES
        )
        return [edit for edit in edits if edit is not None]
