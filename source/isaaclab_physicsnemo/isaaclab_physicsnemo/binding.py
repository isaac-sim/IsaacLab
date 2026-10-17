# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Identity and live buffers of the Newton simulation stepped by an Isaac Lab environment."""

from __future__ import annotations

import math

import newton
import torch
from isaaclab_newton.physics import NewtonManager
from physicsnemo_robotics.newton import (
    FrameSpec,
    NewtonExecutionSpec,
    NewtonSimulationSpec,
    NewtonSolverSignature,
    NewtonTimeSpec,
    collision_pipeline_configuration,
)

from isaaclab.utils.string import string_to_callable

from .signature import isaaclab_model_signature


class NewtonBinding:
    """Checked view of the Newton simulation an initialized Isaac Lab environment steps.

    The binding derives the simulation and execution identity that PhysicsNeMo Robotics trajectory
    schemas and dynamics artifacts embed, and reacquires the live state, control, and contact buffers
    from the physics manager on every access. It never steps or resets the simulation.

    One frame is one Isaac Lab physics step, made of the manager's solver substeps.
    """

    def __init__(self, manager: type[NewtonManager]) -> None:
        """Bind the active Newton physics manager.

        Args:
            manager: Newton physics manager of an environment after construction, such as
                ``env.sim.physics_manager``.

        Raises:
            RuntimeError: If the manager has no active solver.
            ValueError: If the Newton model does not hold one world per environment, or the physics step is
                not a whole number of solver steps.
        """
        # ``SimulationContext.physics_manager`` is a lazily resolved import string of the manager class.
        manager = string_to_callable(manager) if isinstance(manager, str) else manager
        if manager._solver is None:
            raise RuntimeError("The Newton manager has no active solver; construct the environment first.")
        model = manager.get_model()
        if manager.get_num_envs() != model.world_count:
            raise ValueError(
                f"The Newton model holds {model.world_count} worlds for {manager.get_num_envs()} environments."
            )
        self._manager = manager
        self.execution_spec: NewtonExecutionSpec = execution_spec_from_manager(manager)
        """Stepping, collision, and gradient semantics of the manager."""
        self.spec = NewtonSimulationSpec(
            model=isaaclab_model_signature(model),
            solver=NewtonSolverSignature.from_solver(manager._solver),
            time=_frame_time(manager),
            frame=FrameSpec(),
        )
        """Model, solver, timing, and frame identity of the simulation."""

    @property
    def model(self) -> newton.Model:
        """Finalized Newton model of the scene."""
        return self._manager.get_model()

    @property
    def worlds(self) -> int:
        """Number of Newton worlds, one per environment."""
        return self.model.world_count

    @property
    def device(self) -> torch.device:
        """Device of the simulation."""
        return torch.device(str(self.model.device))

    @property
    def state(self) -> newton.State:
        """Current Newton state."""
        return self._manager.get_state_0()

    @property
    def control(self) -> newton.Control:
        """Newton control buffer holding the efforts applied during the next step."""
        return self._manager.get_control()

    @property
    def contacts(self) -> newton.Contacts | None:
        """Newton contact buffer, or ``None`` when the manager runs no collision pipeline."""
        return self._manager.get_contacts()


def execution_spec_from_manager(manager: type[NewtonManager]) -> NewtonExecutionSpec:
    """Derive the stepping and collision semantics of an initialized Newton physics manager.

    Collision results are representable only when the manager detects contacts before every solver
    substep. Collision cadences that skip substeps are rejected instead of being declared inexactly.

    Args:
        manager: Newton physics manager with an active solver.

    Returns:
        The execution declaration that trajectory schemas and artifacts compare.

    Raises:
        ValueError: If the collision cadence cannot be represented exactly.
    """
    manager = string_to_callable(manager) if isinstance(manager, str) else manager
    model = manager.get_model()
    stepper = f"{manager.__module__}.{manager.__qualname__}"
    if not manager._needs_collision_pipeline:
        return NewtonExecutionSpec(
            stepper_implementation=stepper,
            collision_pipeline_implementation=None,
            resolved_collision_configuration={},
            collide_on_reset=False,
            collide_each_substep=False,
            clear_forces=True,
            requires_grad=model.requires_grad,
        )
    if manager._num_substeps > 1 and manager._collision_decimation != 1:
        raise ValueError(
            "Contacts must be refreshed before every solver substep to be represented exactly; set"
            " collision_decimation=1."
        )
    pipeline = manager._collision_pipeline
    return NewtonExecutionSpec(
        stepper_implementation=stepper,
        collision_pipeline_implementation=f"{type(pipeline).__module__}.{type(pipeline).__qualname__}",
        resolved_collision_configuration=collision_pipeline_configuration(pipeline),
        collide_on_reset=False,
        collide_each_substep=True,
        clear_forces=True,
        requires_grad=model.requires_grad,
    )


def _frame_time(manager: type[NewtonManager]) -> NewtonTimeSpec:
    """Return one physics step as a whole number of solver steps."""
    frame_dt, solver_dt = manager.get_dt(), manager.get_solver_dt()
    substeps = round(frame_dt / solver_dt)
    if substeps < 1 or not math.isclose(frame_dt, substeps * solver_dt, rel_tol=1e-12):
        raise ValueError(f"The physics step {frame_dt} s is not a whole number of solver steps of {solver_dt} s.")
    return NewtonTimeSpec(solver_dt=solver_dt, substeps=substeps)
