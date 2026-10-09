# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Solver configuration for the physical Rizon--Sharpa teapot demonstration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg

from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

if TYPE_CHECKING:
    from rizon_sharpa_teapot import NewtonTeapotCouplerManager


@configclass
class TeapotSolverCfg(CouplerProxyCfg):
    """Select convex rigid contacts and the exact MPM shell after simulation launch."""

    class_type: type[NewtonTeapotCouplerManager] | str = "rizon_sharpa_teapot:NewtonTeapotCouplerManager"


def make_solver_cfg(
    mpm_cfg: MPMSolverCfg, *, fluid_coupling: str = "one_way", rigid_substeps: int = 1
) -> TeapotSolverCfg:
    """Resolve hand--pot contacts and select measured sync or liquid feedback."""
    return TeapotSolverCfg(
        entries=[
            CouplerEntryCfg(
                name="robot",
                solver_cfg=MJWarpSolverCfg(
                    integrator="implicitfast",
                    cone="elliptic",
                    impratio=20.0,
                    iterations=16,
                    ls_iterations=32,
                    # Keep headroom for hand self-contacts and the pot without
                    # oversized dense constraint buffers.
                    njmax=96,
                    nconmax=256,
                ),
                bodies=[r"/World/envs/env_.*/Robot", r"/World/envs/env_.*/PourContainer"],
                substeps=rigid_substeps,
            ),
            CouplerEntryCfg(
                name="fluid",
                solver_cfg=mpm_cfg,
                all_particles=True,
                include_static_shapes=True,
                in_place=True,
            ),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source="robot",
                destination="fluid",
                bodies=[r"/World/envs/env_.*/PourContainer"],
                # Match the supported-cup profile in the coupled Franka pouring task.
                # Scale only the MPM proxy mass/inertia; MJWarp retains the configured pot.
                mass_scale=1000.0,
                mode="staggered",
                collision_pipeline=None,
            )
        ]
        if fluid_coupling == "two_way"
        else [],
    )
