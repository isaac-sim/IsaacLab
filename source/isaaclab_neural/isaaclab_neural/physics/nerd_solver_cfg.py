# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Solver configuration (:class:`NeRDSolverCfg`) for standalone ``newton-neural`` state-increment models."""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_newton.physics import NewtonSolverCfg

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .nerd_solver import NeRDSolver


@configclass
class NeRDSolverCfg(NewtonSolverCfg):
    """Configuration for stepping a robot with ``newton-neural``'s standalone ``SolverNeRD``.

    Use it as the solver of a :class:`~isaaclab_newton.physics.NewtonCfg` whose ``class_type`` is the
    :class:`~isaaclab_neural.physics.NewtonNeRDManager` and whose ``num_substeps`` is 1. That manager takes one solver
    step per environment step, the frame the model was trained on. Newton's collision pipeline runs only if a
    contact sensor or the contact input of the model reads its contacts. In a task with neither, set ``collision_cfg``
    of the physics to ``None``, since Isaac Lab refuses a collision configuration that no pipeline uses.

    Example:
        .. code-block:: python

            replace(
                newton_mjwarp,
                class_type="isaaclab_neural.physics.newton_nerd_manager:NewtonNeRDManager",
                solver_cfg=NeRDSolverCfg(checkpoint="anymal_c.pt", articulation="{ENV_REGEX_NS}/Robot"),
                num_substeps=1,
            )
    """

    class_type: type[NeRDSolver] | str = "{DIR}.nerd_solver:NeRDSolver"
    """Solver adapter class wrapping ``newton_neural.solvers.SolverNeRD``."""

    checkpoint: str = MISSING
    """Path to the saved state-increment model checkpoint."""

    articulation: str = MISSING
    """Robot prim path regex (e.g., ``"{ENV_REGEX_NS}/Robot"``) matching Newton articulation labels from start."""

    precision: str | None = None
    """Optional inference precision override (``"fp32"``, ``"tf32"``, ``"fp16"``, ``"bf16"``; checkpoint default)."""

    compile: bool | None = None
    """Optional ``torch.compile`` toggle (defaults to ``True`` on CUDA)."""

    def validate_config(self) -> None:
        """Raise ``ValueError`` with training instructions if :attr:`checkpoint` does not exist."""
        require_checkpoint(self.checkpoint)


def require_checkpoint(checkpoint: str) -> None:
    """Validate that ``checkpoint`` exists on disk; raises ``ValueError`` so preset listing skips untrained presets."""
    path = Path(checkpoint)
    if path.is_file():
        return
    raise ValueError(
        f"the task runs on the model saved at {path.resolve()}, and that file does not exist. Record its data and "
        "train it first, with scripts/neural/record.py and scripts/neural/train.py for this task, run from the same "
        "working directory. To run the task on another model file, set env.sim.physics.solver_cfg.checkpoint=<path>"
    )
