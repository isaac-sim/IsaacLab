# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp Newton manager."""

from __future__ import annotations

import logging

import numpy as np
import warp as wp
from newton import Contacts, Control, Model, State
from newton.solvers import SolverMuJoCo

from isaaclab.physics import PhysicsManager

from .mesh_variants import build_mesh_variants
from .mjwarp_manager_cfg import MJWarpSolverCfg
from .mjwarp_tendon_control import MjWarpTendonControl
from .newton_manager import NewtonManager

logger = logging.getLogger(__name__)


class NewtonMJWarpManager(NewtonManager):
    """:class:`NewtonManager` specialization for the MuJoCo Warp solver.

    Owns construction of :class:`SolverMuJoCo`, contact-buffer allocation in
    both internal-MuJoCo and Newton-pipeline contact modes, and the debug
    convergence logging emitted from :meth:`_log_solver_debug` when
    :attr:`NewtonCfg.debug_mode` is enabled.
    """

    _builder_attribute_solvers = (SolverMuJoCo,)
    _supports_mesh_variants = True
    _contact_observables: SolverMuJoCo.Observables | None = None

    @classmethod
    def _create_solver(cls, model: Model, solver_cfg: MJWarpSolverCfg) -> SolverMuJoCo:
        """Construct the configured MuJoCo Warp solver."""
        kwargs = cls._filter_solver_kwargs(SolverMuJoCo, solver_cfg)
        # ls_parallel is deprecated in newton; forwarding it (even as False) emits a warning.
        kwargs.pop("ls_parallel", None)
        if NewtonManager._mesh_variant_cfgs:
            kwargs.update(separate_worlds=True, enable_sleeping=False)
        solver = SolverMuJoCo(model, **kwargs)
        if NewtonManager._mesh_variant_cfgs and solver.enable_sleeping:
            raise ValueError("Mesh variants require disabled sleeping, including inherited asset settings.")
        return solver

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: MJWarpSolverCfg) -> None:
        """Construct :class:`SolverMuJoCo` and populate the base-class slots.

        Filters cfg fields against the solver's ``__init__`` signature so
        non-constructor metadata (``solver_type``, ``class_type``) and the
        ignored deprecated ``ls_parallel`` field are not forwarded. Sets
        :attr:`NewtonManager._needs_collision_pipeline` to
        ``True`` only when ``use_mujoco_contacts=False``.
        """
        sim = PhysicsManager._sim
        clone_plan = sim.get_clone_plan() if sim is not None else None
        if NewtonManager._mesh_variant_cfgs and solver_cfg.use_mujoco_contacts:
            raise ValueError("Mesh variants require Newton external contacts.")
        variants, render_sets = build_mesh_variants(
            NewtonManager._mesh_variant_cfgs,
            model,
            NewtonManager._cl_protos,
            clone_plan,
            NewtonManager._mesh_variant_visual_sources,
            NewtonManager._mesh_variant_resource_shapes,
        )
        cls.backend.mesh_variants = variants
        NewtonManager._solver = cls._create_solver(model, solver_cfg)
        cls.backend.mesh_variant_render_sets = render_sets
        cls.backend.mesh_variant_ids = {name: selection.variant_ids for name, selection in variants.items()}
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
        cls._contact_observables = None
        if cls._needs_collision_pipeline:
            super()._initialize_contacts()
            cls._contact_observables = (
                cls._solver.observables([SolverMuJoCo.ObservableKind.CONTACT_F], contacts=cls._contacts)
                if cls._report_contacts
                else None
            )
            return
        if cls._solver is not None:
            NewtonManager._contacts = Contacts(
                rigid_contact_max=cls._solver.get_max_contact_count(),
                soft_contact_max=0,
                device=PhysicsManager._device,
                requested_attributes=cls.backend.model.get_requested_contact_attributes(),
            )

    @classmethod
    def _step_solver(
        cls, state_0: State, state_1: State, control: Control, contacts: Contacts | None, substep_dt: float
    ) -> None:
        """Use public force observables to preserve external collision row identities."""
        cls._solver.step(state_0, state_1, control, contacts, substep_dt, observables=cls._contact_observables)

    @classmethod
    def _update_contacts_for_sensors(cls, contacts: Contacts) -> None:
        if cls._contact_observables is None:
            super()._update_contacts_for_sensors(contacts)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        cls._contact_observables = None

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
        world.  Since this hook can fire with an all-``False`` mask (e.g. on
        every forward boundary after a graph-captured write), the CPU path is
        gated on at least one world actually being flagged so warm-starting is
        not defeated on every step.

        Args:
            world_mask: Per-world bool mask of shape ``(world_count + 1,)``.
                Entries before the last select local worlds; the final entry
                selects global entities in world -1. ``None`` is a no-op.
        """
        if world_mask is None:
            return
        if cls._solver.use_mujoco_cpu and not world_mask.numpy().any():
            return
        super()._reset_solver_internals(world_mask)

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
