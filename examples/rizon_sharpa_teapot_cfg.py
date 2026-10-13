# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Solver configuration for the physical Rizon--Sharpa teapot example."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg

from isaaclab.utils import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg

if TYPE_CHECKING:
    from _rizon_sharpa_teapot import NewtonTeapotCouplerManager


@configclass
class TeapotSolverCfg(CouplerProxyCfg):
    """Select convex rigid contacts and the exact MPM shell after simulation launch."""

    class_type: type[NewtonTeapotCouplerManager] | str = "_rizon_sharpa_teapot:NewtonTeapotCouplerManager"


def make_solver_cfg(mpm_cfg: MPMSolverCfg) -> TeapotSolverCfg:
    """Resolve rigid hand--pot contacts and synchronize the measured fluid boundary."""
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
                    # Distributed mesh contacts support the handle against the finger pads.
                    enable_multiccd=True,
                    # Keep headroom for hand self-contacts and the pot without
                    # oversized dense constraint buffers.
                    njmax=128,
                    nconmax=256,
                ),
                bodies=[r"/World/envs/env_.*/Robot", r"/World/envs/env_.*/PourContainer"],
                substeps=1,
            ),
            CouplerEntryCfg(
                name="fluid",
                solver_cfg=mpm_cfg,
                all_particles=True,
                include_static_shapes=True,
                in_place=True,
            ),
        ],
    )
