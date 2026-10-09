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
from newton import Contacts, Model, State
from newton.solvers import SolverMuJoCo

from .mjwarp_manager_cfg import MJWarpSolverCfg
from .mjwarp_tendon_control import MjWarpTendonControl
from .newton_manager import _SENSORS_BY_STATE_ATTRIBUTE, NewtonManager

if TYPE_CHECKING:
    from .newton_backend import NewtonBackend

logger = logging.getLogger(__name__)


class NewtonMJWarpManager(NewtonManager):
    """:class:`NewtonManager` running the MuJoCo Warp solver.

    MuJoCo steps in place on one state. It runs its own collision detection unless
    :attr:`MJWarpSolverCfg.use_mujoco_contacts` is ``False``, in which case Newton's collision pipeline supplies
    contacts.
    """

    builder_attribute_solvers = (SolverMuJoCo,)
    single_state = True
    supports_deterministic = True

    @classmethod
    def create_solver(
        cls,
        model: Model,
        solver_cfg: MJWarpSolverCfg,
        deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED,
    ) -> SolverMuJoCo:
        """Construct the configured MuJoCo Warp solver.

        A determinism guarantee disables MuJoCo Warp's sensor stage, which cannot honor it.
        """
        kwargs = cls.solver_kwargs(SolverMuJoCo, solver_cfg, deterministic_mode)
        # ls_parallel is deprecated in newton; forwarding it (even as False) emits a warning.
        kwargs.pop("ls_parallel", None)
        if deterministic_mode != wp.DeterministicMode.NOT_GUARANTEED:
            kwargs["disable_sensors"] = True
        return SolverMuJoCo(model, **kwargs)

    @classmethod
    def validate_cfg(cls, backend: NewtonBackend) -> None:
        """Reject a collision pipeline configuration while MuJoCo detects contacts internally, and determinism while
        sensors read attributes that only MuJoCo's sensor stage fills."""
        super().validate_cfg(backend)
        cfg = backend.cfg
        if cfg.solver_cfg.use_mujoco_contacts and cfg.collision_cfg is not None:
            raise ValueError(
                "NewtonCfg: collision_cfg cannot be set when solver_cfg.use_mujoco_contacts=True. Either set"
                " use_mujoco_contacts=False or remove collision_cfg."
            )
        if cfg.solver_cfg.use_mujoco_cpu and cfg.deterministic_mode != "not_guaranteed":
            logger.info("MuJoCo CPU backend is already reproducible; Newton's deterministic mode is not applied.")
        mode = backend.deterministic_mode
        blocked = set(backend.model.get_requested_state_attributes()) & _SENSORS_BY_STATE_ATTRIBUTE.keys()
        if mode != wp.DeterministicMode.NOT_GUARANTEED and blocked:
            sensors = sorted({_SENSORS_BY_STATE_ATTRIBUTE[attr] for attr in blocked})
            raise ValueError(
                f"This task does not support deterministic physics: it uses {' and '.join(sensors)},"
                f" reading {sorted(blocked)}. Those attributes come from MuJoCo's post-constraint pass,"
                " which runs inside the sensor stage that a determinism guarantee must disable, so the"
                " values would never be refreshed. Remove the sensors, or drop the determinism request"
                f" (deterministic_mode={mode.name})."
            )

    @classmethod
    def uses_collision_pipeline(cls, backend: NewtonBackend) -> bool:
        return not backend.cfg.solver_cfg.use_mujoco_contacts

    @classmethod
    def create_contacts(cls, backend: NewtonBackend) -> Contacts:
        """Allocate contacts sized to MuJoCo's contact buffer; ``solver.update_contacts`` fills them for sensors."""
        model = backend.model
        return Contacts(
            rigid_contact_max=backend.solver.get_max_contact_count(),
            soft_contact_max=0,
            device=model.device,
            requested_attributes=model.get_requested_contact_attributes(),
        )

    @classmethod
    def reset_solver(cls, backend: NewtonBackend, state: State, world_mask: wp.array) -> None:
        """Clear MuJoCo Warp warm-start and applied-force buffers for masked worlds.

        With ``flags=0`` MuJoCo zeroes only solver-owned buffers that persist across steps (``qacc_warmstart``,
        ``qfrc_applied``, ``xfrc_applied``, ``ctrl``, ``act``), leaving the authored joint state untouched. Without
        this, a NaN from one solve persists across environment resets because the next substep warm-starts from it
        (https://github.com/newton-physics/newton/issues/1266).

        The MuJoCo CPU backend owns one global ``MjData`` whose reset clears every world, so it resets only when at
        least one world is flagged.
        """
        solver = backend.solver
        if solver.use_mujoco_cpu and not world_mask.numpy().any():
            return
        solver.reset(state, world_mask=world_mask, flags=0)

    @classmethod
    def log_debug(cls, backend: NewtonBackend) -> None:
        """Log MuJoCo solver convergence statistics."""
        solver = backend.solver
        niter = solver.mjw_data.solver_niter.numpy()
        data = {"max": np.max(niter), "mean": np.mean(niter), "min": np.min(niter), "std": np.std(niter)}
        logger.info(f"Solver convergence data: {data}")
        if np.max(niter) == solver.mjw_model.opt.iterations:
            logger.warning(f"Solver didn't converge! max_iter={np.max(niter)}")

    @classmethod
    def create_fixed_tendon_control(cls, articulation, model: Model | None = None) -> MjWarpTendonControl | None:
        """Build the MuJoCo tendon adapter for ``articulation``.

        Returns:
            The adapter, or None when no MuJoCo actuator transmits to any of the articulation's tendons.
        """
        return MjWarpTendonControl.create(articulation, model if model is not None else cls.get_model())
