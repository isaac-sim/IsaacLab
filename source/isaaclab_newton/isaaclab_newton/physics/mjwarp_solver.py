# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp solver implementation."""

from __future__ import annotations

import logging
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp
from newton import Contacts, Control, Model, State
from newton.solvers import SolverBase, SolverMuJoCo

from .newton_solver import NewtonSolver, solver_kwargs

if TYPE_CHECKING:
    from .mjwarp_solver_cfg import MJWarpSolverCfg
    from .mjwarp_tendon_control import MjWarpTendonControl

logger = logging.getLogger(__name__)

_SENSORS_BY_STATE_ATTRIBUTE = {"body_qdd": "the IMU or PVA sensor", "body_parent_f": "the joint-wrench sensor"}

_MUJOCO_WARP_OVERFLOW_LABELS = {
    "NEFC": "constraint rows (njmax)",
    "NJMAX_NNZ": "constraint Jacobian nonzeros (njmax_nnz)",
    "BROADPHASE": "broadphase buffer",
    "NARROWPHASE": "narrowphase/contact buffer",
    "CCD": "CCD buffer",
    "HFIELD": "height-field collision buffer",
    "CONTACT_MATCH": "contact-match sensor buffer",
    "NVMAX": "island active-DOF buffer (nvmax)",
    "EPA_HORIZON": "EPA horizon buffer",
    "ITERATIONS": "solver iteration cap (iterations)",
    "LS_ITERATIONS": "line-search iteration cap (ls_iterations)",
    "TACTILE": "tactile sensor collision-pair buffer",
}


@wp.func
def _nonfinite(values: wp.array2d(dtype=Any), world: int, column: int) -> bool:
    return column < values.shape[1] and not wp.isfinite(values[world, column])


@wp.kernel(enable_backward=False)
def _find_nonfinite_solver_worlds(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    qacc: wp.array2d(dtype=wp.float32),
    qfrc_actuator: wp.array2d(dtype=wp.float32),
    qacc_warmstart: wp.array2d(dtype=wp.float32),
    qfrc_applied: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    act: wp.array2d(dtype=wp.float32),
    xfrc_applied: wp.array2d(dtype=wp.spatial_vector),
    invalid_worlds: wp.array(dtype=wp.bool),
):
    world, column = wp.tid()
    if (
        _nonfinite(qpos, world, column)
        or _nonfinite(qvel, world, column)
        or _nonfinite(qacc, world, column)
        or _nonfinite(qfrc_actuator, world, column)
        or _nonfinite(qacc_warmstart, world, column)
        or _nonfinite(qfrc_applied, world, column)
        or _nonfinite(ctrl, world, column)
        or _nonfinite(act, world, column)
        or _nonfinite(xfrc_applied, world, column)
    ):
        invalid_worlds[world] = True


class MJWarpSolver(NewtonSolver):
    """Newton's MuJoCo Warp solver, which steps in place on one state."""

    cfg: MJWarpSolverCfg

    solver_class = SolverMuJoCo
    single_state = True
    max_deterministic_mode = wp.DeterministicMode.RUN_TO_RUN

    def __init__(
        self,
        cfg: MJWarpSolverCfg,
        model: Model,
        mode: wp.DeterministicMode,
        env_template: str | None = None,
    ) -> None:
        super().__init__(cfg, model, mode, env_template)
        self._step_observables: SolverBase.Observables | None = None
        self._warned_overflow_bits: int = 0

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

    def build(self, model: Model, mode: wp.DeterministicMode, env_template: str | None) -> SolverMuJoCo:
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
        kwargs = solver_kwargs(self.solver_class, self.cfg, mode)
        # ls_parallel is deprecated in newton; forwarding it (even as False) emits a warning.
        kwargs.pop("ls_parallel", None)
        if deterministic:
            kwargs["disable_sensors"] = True
        if model._has_body_variants:
            kwargs["separate_worlds"] = True
        return self.solver_class(model, **kwargs)

    def supports_capture(self) -> bool:
        return not self.cfg.use_mujoco_cpu

    def uses_collision_pipeline(self) -> bool:
        return not self.cfg.use_mujoco_contacts

    def create_contacts(self) -> Contacts:
        """Allocate contacts sized to MuJoCo's contact buffer for sensor contact indexing."""
        solver = self.native
        model = solver.model
        return Contacts(
            rigid_contact_max=solver.get_max_contact_count(),
            soft_contact_max=0,
            device=model.device,
            requested_attributes=model.get_requested_contact_attributes(),
        )

    def mark_non_finite_worlds(self, state: State, faults: wp.array) -> None:
        """Include MuJoCo's persistent solver buffers.

        CPU MuJoCo uses one global simulation and clears its history globally on reset, so a fault in
        its private buffers invalidates every environment. MJWarp checks each world's buffers separately.
        """
        super().mark_non_finite_worlds(state, faults)
        data = self.native.mj_data if self.cfg.use_mujoco_cpu else self.native.mjw_data
        arrays = (
            data.qpos,
            data.qvel,
            data.qacc,
            data.qfrc_actuator,
            data.qacc_warmstart,
            data.qfrc_applied,
            data.ctrl,
            data.act,
            data.xfrc_applied,
        )
        if self.cfg.use_mujoco_cpu:
            if any(not np.isfinite(array).all() for array in arrays):
                faults.fill_(True)
        else:
            wp.launch(
                _find_nonfinite_solver_worlds,
                dim=(data.qpos.shape[0], max(array.shape[1] for array in arrays)),
                inputs=arrays,
                outputs=[faults],
                device=faults.device,
            )

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        """Advance one substep and retain the outputs inspected after the step."""
        self._step_observables = observables
        super().step(state_in, state_out, control, contacts, dt, observables=observables)

    def reset(self, state: State, world_mask: wp.array) -> None:
        """Clear solver history and synchronize authored joint state for masked worlds.

        With ``flags=0`` MuJoCo zeroes only solver-owned buffers that persist across steps, leaving the authored joint
        state untouched; otherwise a NaN from one solve persists across resets through the warm start
        (https://github.com/newton-physics/newton/issues/1266). Sparse or disabled imports require an explicit
        synchronization before the next substep can overwrite the authored state. The native CPU backend owns
        one ``MjData`` for template world 0, so other worlds' resets must leave its state and history untouched.
        """
        solver = self.native
        if solver.use_mujoco_cpu and not world_mask.numpy()[0]:
            return
        solver.reset(state, world_mask=world_mask, flags=0)
        if self.cfg.update_data_interval != 1:
            # Newton has no public masked import. With sleeping disabled,
            # reset(flags=0) skips it, so reuse the pinned solver's conversion.
            data = solver.mj_data if solver.use_mujoco_cpu else solver.mjw_data
            solver._update_mjc_data(data, solver.model, state, world_mask=world_mask)

    def check_step(self, captured: bool, debug: bool) -> None:
        """Log MuJoCo capacity-overflow and convergence diagnostics in debug mode."""
        if not debug:
            return
        self._warn_overflow()
        solver = self.native
        niter = solver.mjw_data.solver_niter.numpy()
        data = {"max": np.max(niter), "mean": np.mean(niter), "min": np.min(niter), "std": np.std(niter)}
        logger.info(f"Solver convergence data: {data}")
        if np.max(niter) == solver.mjw_model.opt.iterations:
            logger.warning(f"Solver didn't converge! max_iter={np.max(niter)}")

    @classmethod
    def fixed_tendon_control(cls, cfg: MJWarpSolverCfg, articulation, model: Model) -> MjWarpTendonControl | None:
        """Build the MuJoCo tendon adapter, or ``None`` when no MuJoCo actuator drives the articulation's tendons."""
        from .mjwarp_tendon_control import create_mjwarp_tendon_control

        return create_mjwarp_tendon_control(articulation, model)

    def _diagnostic_observables(self) -> frozenset[str]:
        """Return the overflow observable needed for debug diagnostics, when the backend supports it."""
        if self.cfg.use_mujoco_cpu:
            return frozenset()
        return frozenset({SolverMuJoCo.ObservableKind.OVERFLOW})

    def _warn_overflow(self) -> None:
        """Warn once per newly observed capacity-overflow buffer."""
        if self._step_observables is None or self._step_observables.overflow is None:
            return
        masks = self._step_observables.overflow.numpy()
        if masks.size == 0:
            return
        all_bits = int(np.bitwise_or.reduce(masks))
        new_bits = all_bits & ~self._warned_overflow_bits
        if new_bits == 0:
            return
        self._warned_overflow_bits |= new_bits

        world_masks = masks[:-1]
        global_mask = int(masks[-1])
        affected_worlds = np.flatnonzero(world_masks & new_bits)
        global_affected = bool(global_mask & new_bits)

        labels = _overflow_labels()
        names, known_bits = [], 0
        for bit in sorted(labels):
            if new_bits & bit:
                names.append(labels[bit])
                known_bits |= bit
        unknown_bits = new_bits & ~known_bits
        if unknown_bits:
            names.append(f"unknown overflow bits 0x{unknown_bits:x}")

        worlds = [str(int(index)) for index in affected_worlds[:8]]
        if affected_worlds.size > len(worlds):
            worlds.append(f"and {affected_worlds.size - len(worlds)} more")
        scope = f"{affected_worlds.size} environment(s) ({', '.join(worlds)})" if affected_worlds.size else ""
        if global_affected:
            scope = f"{scope} and the global world" if scope else "the global world"
        logger.warning(
            "MJWarp capacity overflow in %s: %s. "
            "Increase the named capacity; dropped contacts or constraints can invalidate the simulation.",
            scope,
            ", ".join(names),
        )


@cache
def _overflow_labels() -> dict[int, str]:
    """Return user-facing labels for Newton and MuJoCo Warp overflow bits."""
    labels = {
        int(SolverMuJoCo.OverflowBits.CONTACT_PIPELINE): "collision pipeline contact buffer (rigid_contact_max)",
        int(SolverMuJoCo.OverflowBits.SOLVER_NCONMAX): "solver contact buffer (nconmax)",
    }
    _, mujoco_warp = SolverMuJoCo.import_mujoco()
    for name, label in _MUJOCO_WARP_OVERFLOW_LABELS.items():
        flag = getattr(mujoco_warp.OverflowType, name, None)
        if flag is not None:
            labels[int(flag)] = label
    return labels
