# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the custom MJWarp and VBD coupled solver."""

from __future__ import annotations

from typing import Literal

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonSolverCfg, VBDSolverCfg

from isaaclab.utils import configclass


@configclass
class CoupledMJWarpVBDSolverCfg(NewtonSolverCfg):
    """Configuration for the custom MJWarp and VBD coupled solver."""

    class_type: type | str = "{DIR}.coupled_mjwarp_vbd_solver:CoupledMJWarpVBDSolver"

    rigid_solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    """MJWarp rigid-body solver configuration."""

    soft_solver_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD deformable solver configuration."""

    coupling_mode: Literal["one_way", "two_way"] = "two_way"
    """Coupling direction between the rigid and deformable solvers."""
