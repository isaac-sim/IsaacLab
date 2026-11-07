# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated MJWarp and VBD configuration compatible with the generic proxy coupler."""

from __future__ import annotations

import warnings
from typing import Literal

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonSolverCfg, VBDSolverCfg
from isaaclab_newton.physics.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

from isaaclab.utils import configclass, replace


@configclass
class CoupledMJWarpVBDSolverCfg(NewtonSolverCfg):
    """Deprecated configuration for MJWarp and VBD proxy coupling.

    Use :class:`~isaaclab_newton.physics.coupling.CouplerProxyCfg` with named rigid and soft entries. This adapter
    exposes every collidable rigid body to VBD as a proxy. Proxy feedback replaces the former shared-model
    penalty reactions; existing simulations require retuning.
    """

    class_type: type | str = "{DIR}.coupled_mjwarp_vbd_solver:CoupledMJWarpVBDSolver"

    rigid_solver_cfg: MJWarpSolverCfg = MJWarpSolverCfg()
    """MJWarp rigid-body solver configuration."""

    soft_solver_cfg: VBDSolverCfg = VBDSolverCfg(integrate_with_external_rigid_solver=True)
    """VBD configuration, copied with external rigid integration disabled for proxy dynamics."""

    coupling_mode: Literal["one_way", "two_way"] = "two_way"
    """Whether proxy contacts feed reaction forces back to the rigid solver."""

    def __post_init__(self) -> None:
        warnings.warn(
            "CoupledMJWarpVBDSolverCfg is deprecated; use isaaclab_newton.physics.coupling.CouplerProxyCfg. "
            "This compatibility configuration now uses proxy feedback instead of shared-model penalty reactions; "
            "retune existing simulations.",
            FutureWarning,
            stacklevel=2,
        )

    def _to_proxy_cfg(self) -> CouplerProxyCfg:
        """Translate the legacy fields without mutating either nested solver configuration."""
        if self.coupling_mode not in ("one_way", "two_way"):
            raise ValueError("coupling_mode must be 'one_way' or 'two_way'.")
        return CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid", solver_cfg=self.rigid_solver_cfg, bodies=[".*"], include_static_shapes=True
                ),
                CouplerEntryCfg(
                    name="soft",
                    solver_cfg=replace(self.soft_solver_cfg, integrate_with_external_rigid_solver=False),
                    all_particles=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="soft",
                    bodies=[".*"],
                    mode="staggered",
                    collision_pipeline=None,
                    proxy_relaxation=0.0 if self.coupling_mode == "one_way" else 1.0,
                )
            ],
        )
