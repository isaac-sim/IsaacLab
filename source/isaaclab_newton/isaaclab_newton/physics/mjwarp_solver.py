# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo Warp solver specification."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from newton import Contacts, Model, State
from newton.solvers import SolverMuJoCo

from .mjwarp_tendon_control import MjWarpTendonControl, create_mjwarp_tendon_control
from .solver_specs import SolverSpec, solver_kwargs

if TYPE_CHECKING:
    from .mjwarp_solver_cfg import MJWarpSolverCfg
    from .newton_manager_cfg import NewtonCfg

logger = logging.getLogger(__name__)

_SENSORS_BY_STATE_ATTRIBUTE = {"body_qdd": "the IMU or PVA sensor", "body_parent_f": "the joint-wrench sensor"}


def build(model: Model, cfg: MJWarpSolverCfg, deterministic_mode: wp.DeterministicMode) -> SolverMuJoCo:
    """Construct the configured MuJoCo Warp solver.

    A determinism guarantee disables MuJoCo Warp's sensor stage, which cannot honor it.
    """
    kwargs = solver_kwargs(SolverMuJoCo, cfg, deterministic_mode)
    # ls_parallel is deprecated in newton; forwarding it (even as False) emits a warning.
    kwargs.pop("ls_parallel", None)
    if deterministic_mode != wp.DeterministicMode.NOT_GUARANTEED:
        kwargs["disable_sensors"] = True
    return SolverMuJoCo(model, **kwargs)


def validate(model: Model, cfg: MJWarpSolverCfg, physics_cfg: NewtonCfg, mode: wp.DeterministicMode) -> None:
    """Reject a collision pipeline configuration while MuJoCo detects contacts internally, and determinism while
    sensors read attributes that only MuJoCo's sensor stage fills."""
    if cfg.use_mujoco_contacts and physics_cfg.collision_cfg is not None:
        raise ValueError(
            "NewtonCfg: collision_cfg cannot be set when solver_cfg.use_mujoco_contacts=True. Either set"
            " use_mujoco_contacts=False or remove collision_cfg."
        )
    if cfg.use_mujoco_cpu and physics_cfg.deterministic_mode != "not_guaranteed":
        logger.info("MuJoCo CPU backend is already reproducible; Newton's deterministic mode is not applied.")
    blocked = set(model.get_requested_state_attributes()) & _SENSORS_BY_STATE_ATTRIBUTE.keys()
    if mode != wp.DeterministicMode.NOT_GUARANTEED and blocked:
        sensors = sorted({_SENSORS_BY_STATE_ATTRIBUTE[attr] for attr in blocked})
        raise ValueError(
            f"This task does not support deterministic physics: it uses {' and '.join(sensors)},"
            f" reading {sorted(blocked)}. Those attributes come from MuJoCo's post-constraint pass,"
            " which runs inside the sensor stage that a determinism guarantee must disable, so the"
            " values would never be refreshed. Remove the sensors, or drop the determinism request"
            f" (deterministic_mode={mode.name})."
        )


def create_contacts(solver: SolverMuJoCo, cfg: MJWarpSolverCfg) -> Contacts:
    """Allocate contacts sized to MuJoCo's contact buffer; ``solver.update_contacts`` fills them for sensors."""
    model = solver.model
    return Contacts(
        rigid_contact_max=solver.get_max_contact_count(),
        soft_contact_max=0,
        device=model.device,
        requested_attributes=model.get_requested_contact_attributes(),
    )


def reset(solver: SolverMuJoCo, cfg: MJWarpSolverCfg, state: State, world_mask: wp.array) -> None:
    """Clear MuJoCo Warp warm-start and applied-force buffers for masked worlds.

    With ``flags=0`` MuJoCo zeroes only solver-owned buffers that persist across steps (``qacc_warmstart``,
    ``qfrc_applied``, ``xfrc_applied``, ``ctrl``, ``act``), leaving the authored joint state untouched. Without
    this, a NaN from one solve persists across environment resets because the next substep warm-starts from it
    (https://github.com/newton-physics/newton/issues/1266).

    The MuJoCo CPU backend owns one global ``MjData`` whose reset clears every world, so it resets only when at
    least one world is flagged.
    """
    if solver.use_mujoco_cpu and not world_mask.numpy().any():
        return
    solver.reset(state, world_mask=world_mask, flags=0)


def log_debug(solver: SolverMuJoCo, cfg: MJWarpSolverCfg) -> None:
    """Log MuJoCo solver convergence statistics."""
    niter = solver.mjw_data.solver_niter.numpy()
    data = {"max": np.max(niter), "mean": np.mean(niter), "min": np.min(niter), "std": np.std(niter)}
    logger.info(f"Solver convergence data: {data}")
    if np.max(niter) == solver.mjw_model.opt.iterations:
        logger.warning(f"Solver didn't converge! max_iter={np.max(niter)}")


def coupling_error(cfg: MJWarpSolverCfg, in_place: bool) -> str | None:
    """MuJoCo on the CPU resets one global state, which a coupled entry cannot mask per world."""
    if cfg.use_mujoco_cpu:
        return (
            "enables MJWarpSolverCfg.use_mujoco_cpu, whose global reset state cannot yet preserve the per-world"
            " reset-mask lifecycle inside a coupled entry"
        )
    return None


def fixed_tendon_control(articulation, model: Model) -> MjWarpTendonControl | None:
    """Build the MuJoCo tendon adapter, or None when no MuJoCo actuator drives the articulation's tendons."""
    return create_mjwarp_tendon_control(articulation, model)


MJWARP_SPEC = SolverSpec(
    attribute_solvers=lambda cfg: (SolverMuJoCo,),
    validate=validate,
    build=build,
    single_state=True,
    deterministic=True,
    captures_model_changes=True,
    supports_capture=lambda solver, cfg: not cfg.use_mujoco_cpu,
    coupling_error=coupling_error,
    external_contacts=lambda solver, cfg: not cfg.use_mujoco_contacts,
    create_contacts=create_contacts,
    reset=reset,
    log_debug=log_debug,
    fixed_tendon_control=fixed_tendon_control,
)
"""Specification of the MuJoCo Warp solver.

MuJoCo steps in place on one state. It runs its own collision detection unless
:attr:`MJWarpSolverCfg.use_mujoco_contacts` is ``False``, in which case Newton's collision pipeline supplies contacts.
"""
