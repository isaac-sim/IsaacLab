# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment checks for an Isaac Lab environment advanced by a learned-dynamics artifact."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import newton
import torch
import warp as wp
from isaaclab_newton.physics import NewtonManager
from physicsnemo_robotics.dynamics import DynamicsArtifact, LearnedDynamicsSolver

from isaaclab.utils.string import string_to_callable

from .binding import NewtonBinding
from .physics import (
    NewtonPhysicsNeMoManager,
    PhysicsNeMoSolverCfg,
    validate_effort_control,
    validate_state_ownership,
)


@dataclass(frozen=True)
class PreflightCheck:
    """Outcome of one deployment check."""

    name: str
    """Short name of the checked contract."""

    passed: bool
    """Whether the contract holds."""

    detail: str
    """What was verified, or why the check failed."""


@dataclass(frozen=True)
class PreflightReport:
    """Outcome of :func:`preflight_learned_physics`."""

    checks: tuple[PreflightCheck, ...]
    """Checks in evaluation order."""

    def __str__(self) -> str:
        lines = [f"[{'pass' if check.passed else 'FAIL'}] {check.name}: {check.detail}" for check in self.checks]
        lines.append(f"deployment {'ready' if self.passed else 'not ready'}")
        return "\n".join(lines)

    @property
    def passed(self) -> bool:
        """Whether every check passed."""
        return all(check.passed for check in self.checks)


def preflight_learned_physics(manager: type[NewtonManager], artifact: DynamicsArtifact) -> PreflightReport:
    """Check that an environment is ready to advance with a learned-dynamics artifact.

    Run it after the environment's first reset. The checks never step or modify the simulation.

    Args:
        manager: Newton physics manager of the environment, such as ``env.sim.physics_manager``.
        artifact: Artifact the environment was configured with, as returned by
            :func:`~isaaclab_physicsnemo.apply_learned_physics`.

    Returns:
        One result per deployment contract. The report passes when every check passes.
    """
    # ``SimulationContext.physics_manager`` is a lazily resolved import string of the manager class.
    manager = string_to_callable(manager) if isinstance(manager, str) else manager
    checks: list[PreflightCheck] = []

    def check(name: str, predicate: Callable[[], str | None], success: str) -> None:
        """Record a check whose predicate returns a failure reason or ``None``, or raises."""
        try:
            failure = predicate()
        except Exception as error:  # a failed contract is reported, not raised
            failure = f"{type(error).__name__}: {error}"
        checks.append(PreflightCheck(name, failure is None, success if failure is None else failure))

    def manager_dispatch() -> str | None:
        if not (isinstance(manager, type) and issubclass(manager, NewtonPhysicsNeMoManager)):
            return f"the active physics manager is {manager}"
        if not isinstance(manager._cfg.solver_cfg, PhysicsNeMoSolverCfg):
            return "the active solver configuration is not a PhysicsNeMoSolverCfg"
        if not isinstance(manager._solver, LearnedDynamicsSolver):
            return "the environment has not built the learned solver"
        return None

    check("manager dispatch", manager_dispatch, "NewtonCfg selects the PhysicsNeMo learned-dynamics manager")
    if not checks[-1].passed:
        return PreflightReport(tuple(checks))
    solver = manager._solver
    solver_cfg = manager._cfg.solver_cfg

    def artifact_identity() -> str | None:
        configured = Path(solver_cfg.artifact_path).expanduser().resolve()
        if configured != artifact.source_path or solver.artifact_content_identity != artifact.content_identity:
            return f"the installed solver was loaded from {configured}, not from the checked artifact"
        return None

    def simulation_identity() -> str | None:
        binding = NewtonBinding(manager)
        trained = artifact.manifest.schema
        if binding.spec.model.fingerprint != trained.simulation.model.fingerprint:
            return "the live Newton model differs from the artifact's training scene"
        trained.execution.assert_environment_compatible(binding.execution_spec)
        return None

    def cadence() -> str | None:
        frame_dt = artifact.manifest.schema.simulation.time.frame_dt
        if manager._num_substeps != 1 or not math.isclose(manager.get_dt(), frame_dt, rel_tol=1e-9):
            return (
                f"the physics step {manager.get_dt()} s in {manager._num_substeps} substeps differs from {frame_dt} s"
            )
        return None

    def eager_execution() -> str | None:
        if manager._graph is not None or manager._graph_capture_pending:
            return "a CUDA graph is captured or pending"
        return None

    def capabilities() -> str | None:
        if not manager._use_single_state or manager._supports_rigid_body_force_input:
            return "the manager does not use single-state stepping without external body forces"
        if manager._supports_contact_sensors or manager._report_contacts:
            return "contact sensors are enabled"
        model = manager.get_model()
        requested = set(model.get_requested_state_attributes()) & newton.State.EXTENDED_ATTRIBUTES
        return f"sensors request unsupported state attributes {sorted(requested)}" if requested else None

    def collision_inputs() -> str | None:
        requires_contacts = solver.operator.adapter.requires_contacts
        if (
            manager._needs_collision_pipeline != requires_contacts
            or (manager.get_contacts() is None) == requires_contacts
        ):
            return f"the collision pipeline does not match an artifact with requires_contacts={requires_contacts}"
        return None

    def effort_control() -> str | None:
        validate_effort_control(manager.get_model())
        return None

    def pending_model_changes() -> str | None:
        return f"model changes {sorted(manager._model_changes)} are queued" if manager._model_changes else None

    def reset_wiring() -> str | None:
        if NewtonManager._reset_solver_internals_delegate != manager._reset_solver_internals:
            return "environment resets are not forwarded to the learned solver"
        mask = manager._world_reset_mask
        if mask is None or mask.dtype != wp.bool or mask.shape != (manager.get_model().world_count + 1,):
            return "the per-world reset mask is not allocated"
        return None

    def state_ownership() -> str | None:
        validate_state_ownership(manager.get_model(), solver.operator.adapter.codec)
        return None

    def finite_state() -> str | None:
        state = solver.operator.adapter.codec.read(manager.get_state_0())
        return None if bool(torch.isfinite(state).all()) else "the current Newton state is not finite"

    check("artifact identity", artifact_identity, f"the solver runs artifact {artifact.content_identity}")
    check("simulation identity", simulation_identity, "model and execution match the training scene")
    check("frame cadence", cadence, "one learned step per physics step of the artifact frame")
    check("eager execution", eager_execution, "learned inference runs without CUDA graph capture")
    check("solver capabilities", capabilities, "no contact sensor, external force, or extended output")
    check("collision inputs", collision_inputs, "the collision pipeline matches the artifact contact inputs")
    check("effort control", effort_control, "no implicit position drive acts on targets the model does not see")
    check("pending model changes", pending_model_changes, "no physics property change is queued")
    check("reset wiring", reset_wiring, "environment resets restart the affected learned histories")
    check("state ownership", state_ownership, "the codec owns every dynamic Newton state row")
    check("finite state", finite_state, "the current state is finite")
    return PreflightReport(tuple(checks))
