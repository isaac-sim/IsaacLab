# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for VBD and global Newton model parameters."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    NewtonSolverCfg,
    VBDSolverCfg,
)
from isaaclab_newton.physics import NewtonSoftContactCfg as NewtonModelCfg

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class NewtonModelSolverCfg(NewtonSolverCfg):
    """Compatibility base for coupled solver configs with legacy model parameters."""

    model_cfg: NewtonModelCfg | None = None
    """Deprecated global soft-contact configuration.

    Use :attr:`~isaaclab_newton.physics.NewtonCfg.soft_contact_cfg` instead.
    """


@configclass
class CoupledMJWarpVBDSolverCfg(NewtonModelSolverCfg):
    """Deprecated configuration for the coupled MJWarp and VBD solver.

    .. deprecated:: 0.5.0
        Use :class:`isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolverCfg`.
    """

    class_type: type[NewtonManager] | str = (
        "isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager:NewtonCoupledMJWarpVBDManager"
    )
    """Manager class for the coupled MJWarp and VBD solver."""

    rigid_solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    """Rigid-body sub-solver configuration."""

    soft_solver_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD sub-solver configuration."""

    coupling_mode: Literal["one_way", "two_way"] = "two_way"
    """Coupling direction between the rigid and VBD solvers."""

    def __post_init__(self) -> None:
        warnings.warn(
            "isaaclab_contrib.deformable.CoupledMJWarpVBDSolverCfg is deprecated. "
            "Use isaaclab_contrib.custom_coupling.CoupledMJWarpVBDSolverCfg.",
            DeprecationWarning,
            stacklevel=2,
        )
