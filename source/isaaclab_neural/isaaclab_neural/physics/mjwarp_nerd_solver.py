# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton solver adapter (:class:`MJWarpNeRDSolver`) wrapping ``SolverNeRDMuJoCo`` for two-way coupling."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import warp as wp
from isaaclab_newton.physics import MJWarpSolver
from isaaclab_newton.physics.newton_solver import solver_kwargs
from newton import Model
from newton_neural.solvers import SolverNeRDMuJoCo, load_nerd

from ..nerd import default_compile
from .nerd_solver import (
    COUPLING_ERROR,
    articulation_pattern,
    check_model,
    require_articulation,
)

if TYPE_CHECKING:
    from .mjwarp_nerd_solver_cfg import MJWarpNeRDSolverCfg

logger = logging.getLogger(__name__)


class MJWarpNeRDSolver(MJWarpSolver):
    """Isaac Lab ``MJWarpSolver`` subclass injecting neural free accelerations via ``SolverNeRDMuJoCo``.

    Preserves all MuJoCo Warp contacts, sensors, resets, and solver options. Post-startup edits to robot masses or
    gains reach MuJoCo Warp but not the neural weights (each kind emits a ``NewtonNeuralWarning`` once). On CUDA,
    Isaac Lab captures the combined step into its physics-step graph (internal solver graphing disabled).
    """

    cfg: MJWarpNeRDSolverCfg

    solver_class = SolverNeRDMuJoCo
    max_deterministic_mode = wp.DeterministicMode.NOT_GUARANTEED

    @classmethod
    def coupling_error(cls, cfg: MJWarpNeRDSolverCfg, in_place: bool) -> str:
        """Return why neural solvers cannot run inside Isaac Lab's coupled-solver wrapper."""
        return COUPLING_ERROR

    def build(self, model: Model, mode: wp.DeterministicMode, env_template: str | None) -> SolverNeRDMuJoCo:
        """Build ``SolverNeRDMuJoCo`` from ``self.cfg`` and validate the Newton model."""
        device = wp.device_to_torch(model.device)
        nerd = load_nerd(self.cfg.checkpoint, device)
        require_articulation(nerd)
        # Collect matching MJWarpSolverCfg constructor kwargs (no determinism overrides since NOT_GUARANTEED).
        options = solver_kwargs(self.solver_class, self.cfg, mode)
        options.pop("ls_parallel", None)  # Deprecated in Newton; omit to avoid deprecation warning
        if options.get("compile") is None:
            options["compile"] = default_compile(device)
        if model._has_body_variants:
            options["separate_worlds"] = True
        options["articulation"] = articulation_pattern(model, self.cfg.articulation, env_template)
        # Disable internal solver graphs because Isaac Lab captures the full physics step itself.
        solver = self.solver_class(model, nerd, use_cuda_graph=False, **options)
        check_model(solver)
        logger.info(f"Built {solver}")
        return solver

    def supports_capture(self) -> bool:
        """Return ``True`` on CUDA only (Warp CPU capture omits PyTorch ops)."""
        return super().supports_capture() and self.native.model.device.is_cuda
