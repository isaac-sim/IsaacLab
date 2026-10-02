# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp Newton manager."""

from __future__ import annotations

import logging

import numpy as np
import warp as wp
from newton import Contacts, Model, State
from newton.solvers import SolverMuJoCo

from .mjwarp_manager_cfg import MJWarpSolverCfg
from .mjwarp_tendon_control import MjWarpTendonControl
from .newton_manager import NewtonManager
from .newton_manager_cfg import NewtonCfg
from .solver_binding import NewtonSolverBinding

logger = logging.getLogger(__name__)


class MJWarpSolverBinding(NewtonSolverBinding):
    """Binding for the MuJoCo Warp solver.

    MuJoCo steps in place on one state. It runs its own collision detection unless
    :attr:`MJWarpSolverCfg.use_mujoco_contacts` is ``False``, in which case Newton's collision pipeline supplies
    contacts.
    """

    builder_attribute_solvers = (SolverMuJoCo,)

    solver: SolverMuJoCo

    def __init__(self, model: Model, solver_cfg: MJWarpSolverCfg, deterministic_mode: wp.DeterministicMode):
        super().__init__(model, solver_cfg, deterministic_mode)
        self.single_state = True
        self.needs_collision_pipeline = not solver_cfg.use_mujoco_contacts

    @classmethod
    def create(
        cls,
        model: Model,
        solver_cfg: MJWarpSolverCfg,
        deterministic_mode: wp.DeterministicMode = wp.DeterministicMode.NOT_GUARANTEED,
    ) -> SolverMuJoCo:
        """Construct the configured MuJoCo Warp solver."""
        kwargs = cls.filter_kwargs(SolverMuJoCo, solver_cfg, deterministic_mode)
        # ls_parallel is deprecated in newton; forwarding it (even as False) emits a warning.
        kwargs.pop("ls_parallel", None)
        return SolverMuJoCo(model, **kwargs)

    @classmethod
    def validate_cfg(cls, cfg: NewtonCfg) -> None:
        """Reject a collision pipeline configuration while MuJoCo detects contacts internally."""
        if cfg.solver_cfg.use_mujoco_contacts and cfg.collision_cfg is not None:
            raise ValueError(
                "NewtonCfg: collision_cfg cannot be set when "
                "solver_cfg.use_mujoco_contacts=True. Either set "
                "use_mujoco_contacts=False or remove collision_cfg."
            )

    def create_contacts(self) -> Contacts:
        """Allocate contacts sized to MuJoCo's contact buffer; ``solver.update_contacts`` fills them for sensors."""
        return Contacts(
            rigid_contact_max=self.solver.get_max_contact_count(),
            soft_contact_max=0,
            device=self.model.device,
            requested_attributes=self.model.get_requested_contact_attributes(),
        )

    def reset(self, state: State, world_mask: wp.array) -> None:
        """Clear MuJoCo Warp warm-start and applied-force buffers for masked worlds.

        With ``flags=0`` MuJoCo zeroes only solver-owned buffers that persist across steps (``qacc_warmstart``,
        ``qfrc_applied``, ``xfrc_applied``, ``ctrl``, ``act``), leaving the authored joint state untouched. Without
        this, a NaN from one solve persists across environment resets because the next substep warm-starts from it
        (https://github.com/newton-physics/newton/issues/1266).

        The MuJoCo CPU backend owns one global ``MjData`` whose reset clears every world, so it resets only when at
        least one world is flagged.
        """
        if self.solver.use_mujoco_cpu and not world_mask.numpy().any():
            return
        self.solver.reset(state, world_mask=world_mask, flags=0)

    def log_debug(self) -> None:
        """Log MuJoCo solver convergence statistics."""
        niter = self.solver.mjw_data.solver_niter.numpy()
        data = {"max": np.max(niter), "mean": np.mean(niter), "min": np.min(niter), "std": np.std(niter)}
        logger.info(f"Solver convergence data: {data}")
        if np.max(niter) == self.solver.mjw_model.opt.iterations:
            logger.warning(f"Solver didn't converge! max_iter={np.max(niter)}")

    @classmethod
    def create_fixed_tendon_control(cls, articulation, model: Model) -> MjWarpTendonControl | None:
        """Build the MuJoCo tendon adapter for ``articulation``.

        Returns:
            The adapter, or None when no MuJoCo actuator transmits to any of the articulation's tendons.
        """
        return MjWarpTendonControl.create(articulation, model)


class NewtonMJWarpManager(NewtonManager):
    """:class:`NewtonManager` running the MuJoCo Warp solver."""

    solver_binding = MJWarpSolverBinding
