# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp solver implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from mujoco_warp import OverflowType
from newton import Contacts, Control, Model, State
from newton.solvers import SolverMuJoCo

from .newton_solver import NewtonSolver, solver_kwargs

if TYPE_CHECKING:
    from .mjwarp_solver_cfg import MJWarpSolverCfg
    from .mjwarp_tendon_control import MjWarpTendonControl

logger = logging.getLogger(__name__)

_SENSORS_BY_STATE_ATTRIBUTE = {"body_qdd": "the IMU or PVA sensor", "body_parent_f": "the joint-wrench sensor"}

_WORLD_OVERFLOWS = int(OverflowType.NEFC | OverflowType.NJMAX_NNZ)
"""MuJoCo Warp overflows of a single world; the others concern buffers shared by all worlds."""


@wp.kernel(enable_backward=False)
def _accumulate_overflow(overflow: wp.array(dtype=wp.int32), world_overflow: wp.array(dtype=wp.int32)):
    world = wp.tid()
    world_overflow[world] = world_overflow[world] | overflow[world]


@wp.kernel(enable_backward=False)
def _clear_overflow(world_mask: wp.array(dtype=wp.bool), world_overflow: wp.array(dtype=wp.int32)):
    world = wp.tid()
    if world_mask[world]:
        world_overflow[world] = 0


@wp.kernel(enable_backward=False)
def _flag_overflowed_worlds(world_overflow: wp.array(dtype=wp.int32), flags: int, faulted: wp.array(dtype=wp.bool)):
    world = wp.tid()
    if world_overflow[world] & flags:
        faulted[world] = True


class MJWarpSolver(NewtonSolver):
    """Newton's MuJoCo Warp solver, which steps in place on one state."""

    solver_class = SolverMuJoCo
    single_state = True
    deterministic = True

    @classmethod
    def applied_deterministic_mode(cls, cfg: MJWarpSolverCfg, mode: wp.DeterministicMode) -> wp.DeterministicMode:
        """MuJoCo on the CPU is reproducible on its own and Warp's mode does not reach it, so none applies."""
        if not cfg.use_mujoco_cpu:
            return mode
        if mode != wp.DeterministicMode.NOT_GUARANTEED:
            logger.info("MuJoCo CPU backend is already reproducible; Newton's deterministic mode is not applied.")
        return wp.DeterministicMode.NOT_GUARANTEED

    @classmethod
    def coupling_error(cls, cfg: MJWarpSolverCfg, in_place: bool) -> str | None:
        """MuJoCo on the CPU resets one global state, which a coupled entry cannot mask per world."""
        if cfg.use_mujoco_cpu:
            return (
                "enables MJWarpSolverCfg.use_mujoco_cpu, whose global reset state cannot yet preserve the per-world"
                " reset-mask lifecycle inside a coupled entry"
            )
        return None

    @classmethod
    def build(
        cls, cfg: MJWarpSolverCfg, model: Model, mode: wp.DeterministicMode, env_template: str | None = None
    ) -> SolverMuJoCo:
        """Construct MuJoCo Warp; a determinism guarantee disables its sensor stage, which cannot honor it."""
        deterministic = mode != wp.DeterministicMode.NOT_GUARANTEED
        blocked = set(model.get_requested_state_attributes()) & _SENSORS_BY_STATE_ATTRIBUTE.keys()
        if deterministic and blocked:
            sensors = sorted({_SENSORS_BY_STATE_ATTRIBUTE[attr] for attr in blocked})
            raise ValueError(
                f"This task does not support deterministic physics: it uses {' and '.join(sensors)},"
                f" reading {sorted(blocked)}. Those attributes come from MuJoCo's post-constraint pass,"
                " which runs inside the sensor stage that a determinism guarantee must disable, so the"
                " values would never be refreshed. Remove the sensors, or drop the determinism request"
                f" (deterministic_mode={mode.name})."
            )
        kwargs = solver_kwargs(cls.solver_class, cfg, mode)
        # ls_parallel is deprecated in newton; forwarding it (even as False) emits a warning.
        kwargs.pop("ls_parallel", None)
        if deterministic:
            kwargs["disable_sensors"] = True
        solver = cls.solver_class(model, **kwargs)
        # MuJoCo Warp overflow flags of each world since its last reset
        solver.isaaclab_world_overflow = wp.zeros(model.world_count, dtype=wp.int32, device=model.device)
        return solver

    @classmethod
    def supports_capture(cls, cfg: MJWarpSolverCfg, solver: SolverMuJoCo) -> bool:
        return not cfg.use_mujoco_cpu

    @classmethod
    def uses_collision_pipeline(cls, cfg: MJWarpSolverCfg, solver: SolverMuJoCo) -> bool:
        return not cfg.use_mujoco_contacts

    @classmethod
    def create_contacts(cls, cfg: MJWarpSolverCfg, solver: SolverMuJoCo) -> Contacts:
        """Allocate contacts sized to MuJoCo's contact buffer; ``update_contacts`` fills them for sensors."""
        model = solver.model
        return Contacts(
            rigid_contact_max=solver.get_max_contact_count(),
            soft_contact_max=0,
            device=model.device,
            requested_attributes=model.get_requested_contact_attributes(),
        )

    @classmethod
    def reset(cls, cfg: MJWarpSolverCfg, solver: SolverMuJoCo, state: State, world_mask: wp.array) -> None:
        """Clear MuJoCo Warp warm-start and applied-force buffers for masked worlds.

        With ``flags=0`` MuJoCo zeroes only solver-owned buffers that persist across steps, leaving the authored joint
        state untouched; otherwise a NaN from one solve persists across resets through the warm start
        (https://github.com/newton-physics/newton/issues/1266). The CPU backend's one global ``MjData`` resets every
        world, so it resets only when a world is flagged.
        """
        if solver.use_mujoco_cpu and not world_mask.numpy().any():
            return
        solver.reset(state, world_mask=world_mask, flags=0)
        overflow = solver.isaaclab_world_overflow
        wp.launch(_clear_overflow, dim=len(overflow), inputs=[world_mask], outputs=[overflow], device=overflow.device)

    @classmethod
    def step(
        cls,
        cfg: MJWarpSolverCfg,
        solver: SolverMuJoCo,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        super().step(cfg, solver, state_in, state_out, control, contacts, dt)
        if not cfg.use_mujoco_cpu:
            # MuJoCo Warp clears its overflow flags every substep; keep them until the world is reset.
            overflow = solver.isaaclab_world_overflow
            inputs, device = [solver.mjw_data.overflow], overflow.device
            wp.launch(_accumulate_overflow, dim=len(overflow), inputs=inputs, outputs=[overflow], device=device)

    @classmethod
    def check_step(cls, cfg: MJWarpSolverCfg, solver: SolverMuJoCo, captured: bool, debug: bool) -> None:
        """Log MuJoCo solver convergence statistics in debug mode."""
        if not debug:
            return
        niter = solver.mjw_data.solver_niter.numpy()
        data = {"max": np.max(niter), "mean": np.mean(niter), "min": np.min(niter), "std": np.std(niter)}
        logger.info(f"Solver convergence data: {data}")
        if np.max(niter) == solver.mjw_model.opt.iterations:
            logger.warning(f"Solver didn't converge! max_iter={np.max(niter)}")

    @classmethod
    def flag_faulted_worlds(cls, cfg: MJWarpSolverCfg, solver: SolverMuJoCo, faulted: wp.array) -> None:
        """Flag the worlds whose constraints overflowed their ``njmax`` capacity since their last reset, which MuJoCo
        Warp drops from the solve of their world."""
        overflow = solver.isaaclab_world_overflow
        inputs = [overflow, _WORLD_OVERFLOWS]
        wp.launch(_flag_overflowed_worlds, dim=len(overflow), inputs=inputs, outputs=[faulted], device=overflow.device)

    @classmethod
    def fixed_tendon_control(cls, cfg: MJWarpSolverCfg, articulation, model: Model) -> MjWarpTendonControl | None:
        """Build the MuJoCo tendon adapter, or ``None`` when no MuJoCo actuator drives the articulation's tendons."""
        from .mjwarp_tendon_control import create_mjwarp_tendon_control

        return create_mjwarp_tendon_control(articulation, model)
