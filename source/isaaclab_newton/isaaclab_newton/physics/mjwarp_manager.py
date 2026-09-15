# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp Newton manager."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from newton import Contacts, JointType, Model
from newton.solvers import SolverMuJoCo

from isaaclab.physics import PhysicsManager

from .mjwarp_manager_cfg import MJWarpSolverCfg
from .mjwarp_tendon_control import MjWarpTendonControl
from .newton_manager import NewtonManager

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim.usd_export import UsdWriter

logger = logging.getLogger(__name__)


class NewtonMJWarpManager(NewtonManager):
    """:class:`NewtonManager` specialization for the MuJoCo Warp solver.

    Owns construction of :class:`SolverMuJoCo`, contact-buffer allocation in
    both internal-MuJoCo and Newton-pipeline contact modes, and the debug
    convergence logging emitted from :meth:`_log_solver_debug` when
    :attr:`NewtonCfg.debug_mode` is enabled.
    """

    _builder_attribute_solvers = (SolverMuJoCo,)

    @classmethod
    def author_fixed_configuration(cls, writer: UsdWriter, scene: InteractiveScene) -> None:
        """Export MuJoCo options and native contact properties from the selected solver world."""
        import json

        from pxr import Gf, Sdf, UsdPhysics, UsdShade

        from isaaclab.assets.physics_properties import UsdAttribute
        from isaaclab.sim.usd_export import AssetPaths

        from isaaclab_newton.sim.schemas import apply_mujoco_collision
        from isaaclab_newton.sim.schemas.schemas_cfg import MujocoCollisionCfg

        super().author_fixed_configuration(writer, scene)
        solver = cls._solver
        model = cls.get_model()
        world = writer.env_id if solver.mjc_body_to_newton.shape[0] > 1 else 0
        native = solver.mj_model if solver.use_mujoco_cpu else solver.mjw_model

        def value(owner, name):
            data = getattr(owner, name)
            if isinstance(data, wp.array):
                data = data.numpy()
                template = solver.mj_model.opt if owner is native.opt else solver.mj_model
                if data.ndim > np.asarray(getattr(template, name, 0.0)).ndim:
                    data = data[world if data.shape[0] > 1 else 0]
            return np.asarray(data)

        options = cls._filter_solver_kwargs(SolverMuJoCo, scene.sim.cfg.physics.solver_cfg)
        options.pop("save_to_mjcf", None)
        options.pop("ls_parallel", None)
        for name in tuple(options):
            if hasattr(native.opt, name):
                data = value(native.opt, name)
                options[name] = data.tolist()
        # MJWarp stores the inverse square root, which may differ from the CPU template.
        if hasattr(native.opt, "impratio_invsqrt"):
            options["impratio"] = float(value(native.opt, "impratio_invsqrt")) ** -2
        options["use_mujoco_contacts"] = solver._use_mujoco_contacts
        options["update_data_interval"] = solver.update_data_interval
        options["enable_sleeping"] = solver.enable_sleeping
        import mujoco

        flags = int(value(native.opt, "disableflags"))
        options["disable_contacts"] = bool(flags & mujoco.mjtDisableBit.mjDSBL_CONTACT)
        options["disable_sensors"] = bool(flags & mujoco.mjtDisableBit.mjDSBL_SENSOR)
        options["enable_multiccd"] = not bool(flags & mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        options = {name: option for name, option in options.items() if option is not None}
        writer.stage.GetRootLayer().customLayerData = {
            **writer.stage.GetRootLayer().customLayerData,
            "isaaclab:newtonDriver": {"solver": "mujoco", "options": json.dumps(options)},
        }
        writer.write_gravity(scene.physics_scene_path, value(native.opt, "gravity"))

        def path_for(kind, index):
            path = getattr(model, kind + "_label")[index]
            return writer.resolve_paths(AssetPaths([(path, 0)], [])).bodies[0][0]

        body_values = {name: value(native, "body_" + name) for name in ("mass", "inertia", "ipos", "iquat", "gravcomp")}
        for body, index in enumerate(solver.mjc_body_to_newton.numpy()[world]):
            if index < 0 or model.body_world.numpy()[index] not in (-1, writer.env_id):
                continue
            path = path_for("body", index)
            prim = writer.stage.GetPrimAtPath(path)
            quat = body_values["iquat"][body]
            rotation = np.asarray(Gf.Matrix3d(Gf.Quatd(float(quat[0]), Gf.Vec3d(*map(float, quat[1:]))))).T
            inertia = rotation @ np.diag(body_values["inertia"][body]) @ rotation.T
            com = np.concatenate((body_values["ipos"][body], quat[1:], quat[:1]))
            writer.write_mass_properties(prim, float(body_values["mass"][body]), inertia, com)
            writer.write_attribute(
                path, UsdAttribute("mjc:gravcomp", type_name="float"), float(body_values["gravcomp"][body])
            )

        starts = model.joint_qd_start.numpy()
        shared_joint_values = {}

        def write_joint_value(path, name, data):
            key = (path, name)
            previous = shared_joint_values.setdefault(key, data.copy())
            if not np.array_equal(previous, data):
                raise NotImplementedError(f"MuJoCo USD cannot represent distinct per-axis {name} at {path}.")
            attribute = writer.stage.GetPrimAtPath(path).GetAttribute(name)
            # Existing MJC assets may use double precision for these numeric properties.
            type_name = str(attribute.GetTypeName()) if attribute else ("float[]" if data.ndim else "float")
            writer.write_attribute(path, UsdAttribute(name, type_name=type_name), data)

        dof_values = {
            name: value(native, "dof_" + name) for name in ("armature", "damping", "frictionloss", "solref", "solimp")
        }
        for dof, index in enumerate(solver.mjc_dof_to_newton_dof.numpy()[world]):
            if index < 0:
                continue
            joint = int(np.searchsorted(starts, index, side="right") - 1)
            if model.joint_world.numpy()[joint] not in (-1, writer.env_id) or int(
                model.joint_type.numpy()[joint]
            ) == int(JointType.FREE):
                continue
            path = path_for("joint", joint)
            if not writer.stage.GetPrimAtPath(path):
                continue
            for name, data in dof_values.items():
                target = {"solref": "solreffriction", "solimp": "solimpfriction"}.get(name, name)
                write_joint_value(path, "mjc:" + target, data[dof])
        joint_values = {name: value(native, "jnt_" + name) for name in ("stiffness", "margin", "solref", "solimp")}
        for joint, index in enumerate(solver.mjc_jnt_to_newton_jnt.numpy()[world]):
            if (
                index < 0
                or int(model.joint_type.numpy()[index]) == int(JointType.FREE)
                or model.joint_world.numpy()[index] not in (-1, writer.env_id)
            ):
                continue
            path = path_for("joint", index)
            if not writer.stage.GetPrimAtPath(path):
                continue
            for name, data in joint_values.items():
                target = {"solref": "solreflimit", "solimp": "solimplimit"}.get(name, name)
                write_joint_value(path, "mjc:" + target, data[joint])

        mapping = solver.mjc_geom_to_newton_shape.numpy()[world]
        fields = {
            name: value(native, "geom_" + name)
            for name in (
                "condim",
                "group",
                "priority",
                "solimp",
                "solmix",
                "solref",
                "friction",
                "contype",
                "conaffinity",
            )
        }
        cls._author_collision_filters(
            writer,
            mapping,
            fields,
            value(native, "geom_bodyid"),
            set(map(int, np.asarray(solver.mj_model.exclude_signature))),
        )

        for geom, shape in enumerate(mapping):
            if shape < 0 or model.shape_world.numpy()[shape] not in (-1, writer.env_id):
                continue
            path = writer.resolve_paths(AssetPaths([(model.shape_label[shape], 0)], [])).bodies[0][0]
            prim = writer.stage.GetPrimAtPath(path)
            if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            cfg = MujocoCollisionCfg(
                **{
                    name: data[geom].tolist()
                    for name, data in fields.items()
                    if name not in {"friction", "contype", "conaffinity"}
                }
            )
            apply_mujoco_collision(cfg, path, writer.stage)
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            friction = fields["friction"][geom]
            native_friction = {
                "physics:dynamicFriction": friction[0],
                "newton:torsionalFriction": friction[1],
                "newton:rollingFriction": friction[2],
            }
            if not material or any(
                material.GetPrim().GetAttribute(name).Get() != float(v) for name, v in native_friction.items()
            ):
                material = writer.material_for_override(prim)
                for name, v in native_friction.items():
                    writer.write_attribute(str(material.GetPath()), UsdAttribute(name, type_name="float"), float(v))
                # Explicit MuJoCo attributes take precedence over the Newton material bridge.
                for name, v in (("mjc:torsionalfriction", friction[1]), ("mjc:rollingfriction", friction[2])):
                    writer.write_attribute(str(material.GetPath()), UsdAttribute(name, type_name="float"), float(v))
            for name in ("contype", "conaffinity"):
                prim.CreateAttribute("mjc:" + name, Sdf.ValueTypeNames.Int64).Set(int(fields[name][geom]))

    @classmethod
    def _author_collision_filters(
        cls,
        writer: UsdWriter,
        mapping: np.ndarray,
        fields: dict[str, np.ndarray],
        bodies: np.ndarray,
        excluded: set[int],
    ) -> None:
        """Preserve effective native collision relationships independently of mask bit allocation."""
        from pxr import UsdPhysics

        from isaaclab.sim.usd_export import AssetPaths

        model = cls.get_model()
        # Native masks can have runtime overrides absent from the Newton model and source USD.
        # Encode disabled pairs as standard relationships because USD imports regenerate mask bits.
        for first, shape in enumerate(mapping):
            if shape < 0 or model.shape_world.numpy()[shape] not in (-1, writer.env_id):
                continue
            prim = writer.stage.GetPrimAtPath(
                writer.resolve_paths(AssetPaths([(model.shape_label[shape], 0)], [])).bodies[0][0]
            )
            if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            filtered = []
            for second in range(first + 1, len(mapping)):
                other = mapping[second]
                if other < 0 or model.shape_world.numpy()[other] not in (-1, writer.env_id):
                    continue
                target = writer.stage.GetPrimAtPath(
                    writer.resolve_paths(AssetPaths([(model.shape_label[other], 0)], [])).bodies[0][0]
                )
                if not target or not target.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                body0, body1 = sorted((int(bodies[first]), int(bodies[second])))
                enabled = (int(fields["contype"][first]) & int(fields["conaffinity"][second])) or (
                    int(fields["contype"][second]) & int(fields["conaffinity"][first])
                )
                if not enabled or body0 == body1 or ((body0 << 16) + body1) in excluded:
                    filtered.append(target.GetPath())
            if filtered:
                relationship = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
                for target in filtered:
                    relationship.AddTarget(target)

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: MJWarpSolverCfg) -> SolverMuJoCo:
        """Construct the configured MuJoCo Warp solver."""
        kwargs = cls._filter_solver_kwargs(SolverMuJoCo, solver_cfg)
        # ls_parallel is deprecated in newton; forwarding it (even as False) emits a warning.
        kwargs.pop("ls_parallel", None)
        return SolverMuJoCo(model, **kwargs)

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: MJWarpSolverCfg) -> None:
        """Construct :class:`SolverMuJoCo` and populate the base-class slots.

        Filters cfg fields against the solver's ``__init__`` signature so
        non-constructor metadata (``solver_type``, ``class_type``) and the
        ignored deprecated ``ls_parallel`` field are not forwarded. Sets
        :attr:`NewtonManager._needs_collision_pipeline` to
        ``True`` only when ``use_mujoco_contacts=False``.
        """
        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        NewtonManager._use_single_state = True
        NewtonManager._needs_collision_pipeline = not solver_cfg.use_mujoco_contacts
        NewtonManager._supports_rigid_body_force_input = True

        cfg = PhysicsManager._cfg
        # Cross-config validation that needs both halves.
        if solver_cfg.use_mujoco_contacts and cfg.collision_cfg is not None:
            raise ValueError(
                "NewtonCfg: collision_cfg cannot be set when "
                "solver_cfg.use_mujoco_contacts=True. Either set "
                "use_mujoco_contacts=False or remove collision_cfg."
            )

    @classmethod
    def create_fixed_tendon_control(cls, articulation):
        """Build the MuJoCo tendon adapter for ``articulation``.

        Args:
            articulation: Newton articulation to drive.

        Returns:
            The adapter, or None when no MuJoCo actuator transmits to any of its tendons.
        """
        return MjWarpTendonControl.create(articulation, cls.get_model())

    @classmethod
    def _initialize_contacts(cls) -> None:
        """Allocate contact buffers.

        Delegates to the base implementation when Newton's
        :class:`CollisionPipeline` is active.  When ``use_mujoco_contacts=True``
        the solver runs MuJoCo's internal collision detection, so this method
        instead pre-allocates a :class:`Contacts` buffer sized to the solver's
        maximum contact count; ``solver.update_contacts`` later populates it
        from MuJoCo data for contact-sensor reporting.
        """
        if cls._needs_collision_pipeline:
            super()._initialize_contacts()
            return
        if cls._solver is not None:
            NewtonManager._contacts = Contacts(
                rigid_contact_max=cls._solver.get_max_contact_count(),
                soft_contact_max=0,
                device=PhysicsManager._device,
                requested_attributes=cls._model.get_requested_contact_attributes(),
            )

    @classmethod
    def _reset_solver_internals(cls, world_mask: wp.array | None) -> None:
        """Clear MuJoCo Warp solver-internal state for flagged worlds.

        Specializes the base hook, whose :meth:`SolverBase.reset` call resolves
        to :meth:`SolverMuJoCo.reset` here: with ``flags=0`` it zeroes only the
        solver-owned buffers persisting across steps (``qacc_warmstart``,
        ``qfrc_applied``, ``xfrc_applied``, ``ctrl``, ``act``) for the flagged
        worlds, while the joint state IsaacLab authored during the env reset is
        left untouched.  Without this, a NaN produced in one solve persists
        across :meth:`isaaclab.envs.ManagerBasedEnv.reset` because the next
        solver substep warm-starts from the NaN — the world is then permanently
        dead.  See https://github.com/newton-physics/newton/issues/1266.

        With ``use_mujoco_cpu=True`` the solver owns a single global ``MjData``
        and its reset path is not mask-aware — it clears the buffers for every
        world.  Since this hook fires on every step/forward boundary (usually
        with an all-``False`` mask), the CPU path is gated on at least one
        world actually being flagged so warm-starting is not defeated on every
        step.

        Args:
            world_mask: Per-world bool mask of shape ``(world_count + 1,)``.
                Entries before the last select local worlds; the final entry
                selects global entities in world -1. ``None`` is a no-op.
        """
        if world_mask is None:
            return
        if cls._solver.use_mujoco_cpu and not world_mask.numpy().any():
            return
        # flags=0 skips the joint-state reset to model defaults: IsaacLab owns
        # joint_q/joint_qd and has already written the authored reset pose.
        cls._solver.reset(cls._state_0, world_mask=world_mask, flags=0)

    @classmethod
    def _log_solver_debug(cls) -> None:
        """Optionally log MuJoCo solver convergence at the end of step."""
        cfg = PhysicsManager._cfg
        if cfg is not None and cfg.debug_mode:  # type: ignore[union-attr]
            data = cls._get_solver_convergence_steps()
            logger.info(f"Solver convergence data: {data}")
            if data["max"] == cls._solver.mjw_model.opt.iterations:
                logger.warning(f"Solver didn't converge! max_iter={data['max']}")

    @classmethod
    def _get_solver_convergence_steps(cls) -> dict[str, float | int]:
        """Return MuJoCo Warp solver convergence statistics.

        Reads ``mjw_data.solver_niter`` (only available on
        :class:`SolverMuJoCo`) and summarizes per-environment iteration counts.
        """
        niter = cls._solver.mjw_data.solver_niter.numpy()
        return {
            "max": np.max(niter),
            "mean": np.mean(niter),
            "min": np.min(niter),
            "std": np.std(niter),
        }
