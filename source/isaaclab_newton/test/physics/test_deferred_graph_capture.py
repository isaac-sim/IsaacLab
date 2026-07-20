# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression test for the deferred CUDA-graph capture policy.

Decision sites (``initialize_solver``, ``set_decimation``, hard reset) never
capture the CUDA graph — they invalidate it and arm a pending flag that the
first ``step()`` consumes by capturing. Callers that never set a decimation —
plain :class:`~isaaclab.sim.SimulationContext` loops — must still get their
graph from that first step.

Regression: on the Newton-actuator path the initial capture was skipped
outright instead of deferred, so such scenes ran eager forever (measured
5.06 ms vs 0.91 ms per tick for the identical workload).
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import warp as wp
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationCfg, build_simulation_context

from isaaclab_assets import ANYMAL_C_CFG

NEWTON_CFG = NewtonCfg(solver_cfg=MJWarpSolverCfg(), use_cuda_graph=True)

IMPLICIT_ACTUATORS = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        stiffness=80.0,
        damping=2.0,
        effort_limit_sim=80.0,
        velocity_limit_sim=7.5,
    )
}


def test_first_step_captures_deferred_graph():
    """A fast-path scene that never sets a decimation gets its graph on the first step."""
    sim_cfg = SimulationCfg(dt=1.0 / 120.0, physics=NEWTON_CFG)
    with build_simulation_context(
        device="cuda:0",
        gravity_enabled=True,
        add_ground_plane=True,
        sim_cfg=sim_cfg,
    ) as sim:
        sim._app_control_on_stop_handle = None
        sim_utils.create_prim("/World/Env_0", "Xform", translation=(0.0, 0.0, 0.0))
        art_cfg = ANYMAL_C_CFG.replace(actuators=IMPLICIT_ACTUATORS, prim_path="/World/Env_0/Robot")
        articulation = Articulation(art_cfg)
        sim.reset()
        assert articulation.is_initialized

        # No set_decimation call anywhere: the flag armed by initialize_solver
        # must survive untouched until the first step consumes it.
        assert SimulationManager._graph_capture_pending, "initialize_solver must arm the deferred capture"
        assert SimulationManager._graph is None

        articulation.write_data_to_sim()
        sim.step()

        assert not SimulationManager._graph_capture_pending, "first step must consume the pending capture"
        assert SimulationManager._graph is not None, "first step must capture the CUDA graph (not run eager forever)"

        # The captured graph must actually be the step vehicle: stepping again
        # keeps the state finite and does not re-arm the flag.
        articulation.write_data_to_sim()
        sim.step()
        assert not SimulationManager._graph_capture_pending
        joint_pos = wp.to_torch(articulation.data.joint_pos)
        assert bool(joint_pos.isfinite().all()), "post-capture step produced non-finite joint state"
