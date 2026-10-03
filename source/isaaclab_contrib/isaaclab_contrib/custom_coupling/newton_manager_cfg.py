# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the custom MJWarp and VBD coupling manager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonSolverCfg, VBDSolverCfg

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class CoupledMJWarpVBDSolverCfg(NewtonSolverCfg):
    """Configuration for the custom MJWarp and VBD coupling manager."""

    class_type: type[NewtonManager] | str = "{DIR}.coupled_mjwarp_vbd_manager:NewtonCoupledMJWarpVBDManager"
    """Manager class for the coupled solver."""

    rigid_solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    """MJWarp rigid-body solver configuration."""

    soft_solver_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD deformable solver configuration."""

    coupling_mode: Literal["one_way", "two_way"] = "two_way"
    """Coupling direction between the rigid and deformable solvers."""

    @property
    def physics_solvers(self) -> tuple[str, ...]:
        """Return the active rigid and deformable solver identifiers."""
        return tuple(sorted(set(self.rigid_solver_cfg.physics_solvers + self.soft_solver_cfg.physics_solvers)))

    @property
    def physics_coupling(self) -> str:
        """Return the custom coupling method and direction."""
        return f"custom_{self.coupling_mode}"
