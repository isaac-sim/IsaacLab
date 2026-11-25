# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Solver configuration (:class:`MJWarpNeRDSolverCfg`) for two-way coupled free-motion models in MuJoCo Warp."""

from __future__ import annotations

import dataclasses
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any, Self

from isaaclab_newton.physics import MJWarpSolverCfg

from isaaclab.utils import configclass

from .nerd_solver_cfg import require_checkpoint

if TYPE_CHECKING:
    from .mjwarp_nerd_solver import MJWarpNeRDSolver


@configclass
class MJWarpNeRDSolverCfg(MJWarpSolverCfg):
    """Configuration for two-way coupling a free-motion neural model inside MuJoCo Warp (``SolverNeRDMuJoCo``).

    Inherits all :class:`~isaaclab_newton.physics.MJWarpSolverCfg` fields; ``sim.dt / num_substeps`` must match the
    checkpoint's ``frame_dt``.
    """

    class_type: type[MJWarpNeRDSolver] | str = "{DIR}.mjwarp_nerd_solver:MJWarpNeRDSolver"
    """Solver adapter class wrapping ``newton_neural.solvers.SolverNeRDMuJoCo``."""

    checkpoint: str = ""
    """Path to the saved free-motion model checkpoint.

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

    @classmethod
    def from_mjwarp(cls, mujoco: MJWarpSolverCfg, **settings: Any) -> Self:
        """Copy all settings from ``mujoco`` (excluding ``class_type``) and apply ``settings`` overrides."""
        # Omit class_type so the constructed config keeps MJWarpNeRDSolver.
        fields = (field.name for field in dataclasses.fields(MJWarpSolverCfg) if field.name != "class_type")
        return cls(**{name: getattr(mujoco, name) for name in fields} | settings)
