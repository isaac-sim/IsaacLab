# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the custom MJWarp and VBD coupling manager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab_newton.physics import MJWarpSolverCfg

from isaaclab.utils.configclass import configclass

from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelSolverCfg, VBDSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class CoupledMJWarpVBDSolverCfg(NewtonModelSolverCfg):
    """Configuration for the custom MJWarp and VBD coupling manager."""

    class_type: type[NewtonManager] | str = "{DIR}.coupled_mjwarp_vbd_manager:NewtonCoupledMJWarpVBDManager"
    """Manager class for the coupled solver."""

    rigid_solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    """MJWarp rigid-body solver configuration."""

    soft_solver_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD deformable solver configuration."""

    coupling_mode: Literal["one_way", "two_way"] = "two_way"
    """Coupling direction between the rigid and deformable solvers."""
