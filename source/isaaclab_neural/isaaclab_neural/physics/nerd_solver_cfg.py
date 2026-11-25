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

    Use it as the solver of an ordinary :class:`~isaaclab_newton.physics.NewtonCfg`. Each solver step must
    match the checkpoint's frame duration. For a model trained on one task control period, declare that period
    as ``sim.dt``, set ``decimation=1`` and ``num_substeps=1``. The native solver rejects a mismatched timestep.

    Example:
        .. code-block:: python

            replace(
                newton_mjwarp,
                solver_cfg=NeRDSolverCfg(checkpoint="anymal_c.pt", articulation="{ENV_REGEX_NS}/Robot"),
                num_substeps=1,
            )
    """

    class_type: type[NeRDSolver] | str = "{DIR}.nerd_solver:NeRDSolver"
    """Solver adapter class wrapping ``newton_neural.solvers.SolverNeRD``."""

    checkpoint: str = ""
    """Path to the saved state-increment model checkpoint.

    A shared physics declaration leaves it empty, and a task that has a model sets it. Validation rejects an
    empty path, so a task without a model does not offer the preset as runnable.
    """

    articulation: str = MISSING
    """Robot prim path regex (e.g., ``"{ENV_REGEX_NS}/Robot"``) matching Newton articulation labels from start."""

    precision: str | None = None
    """Optional inference precision override (``"fp32"``, ``"tf32"``, ``"fp16"``, ``"bf16"``; checkpoint default)."""

    compile: bool | None = None
    """Optional ``torch.compile`` toggle (defaults to ``True`` on CUDA)."""

    def validate_config(self) -> None:
        """Raise ``ValueError`` if :attr:`checkpoint` is empty or does not exist, with training instructions."""
        require_checkpoint(self.checkpoint)


def require_checkpoint(checkpoint: str) -> None:
    """Validate that a checkpoint is configured and exists on disk."""
    if not checkpoint:
        raise ValueError(
            "the task declares no neural dynamics model for this physics preset. Record and train one with "
            "scripts/neural/record.py and scripts/neural/train.py for a task that registers a model, or name a "
            "compatible model file with env.sim.physics.solver_cfg.checkpoint=<path>"
        )
    path = Path(checkpoint)
    if path.is_file():
        return
    raise ValueError(
        f"the task runs on the model saved at {path.resolve()}, and that file does not exist. Record its data and "
        "train it first, with scripts/neural/record.py and scripts/neural/train.py for this task, run from the same "
        "working directory. To run the task on another model file, set env.sim.physics.solver_cfg.checkpoint=<path>"
    )
