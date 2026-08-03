# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the solver-internal reset performed when env-reset masks are consumed."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

from unittest.mock import patch

import pytest
import torch
import warp as wp
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager
from newton.solvers import SolverMuJoCo

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def _generate_single_joint_articulations(num_articulations: int, device: str) -> Articulation:
    """Spawn ``num_articulations`` copies of the simple revolute articulation, one per env prim."""
    for i in range(num_articulations):
        sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 2.5, 0.0, 0.0))
    articulation_cfg = ArticulationCfg(
        prim_path="/World/Env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_force=80.0, max_joint_velocity=5.0),
        ),
        actuators={
            "joint": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=400.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=10.0,
            ),
        },
    )
    return Articulation(articulation_cfg)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_env_reset_clears_selected_mjwarp_solver_internals(device):
    """An env reset clears the flagged world's MuJoCo warm-start history and keeps the others.

    Regression test for NaN values persisting across env reset: MJWarp warm-starts each solve
    from ``qacc_warmstart``, so solver-internal buffers must be cleared for reset worlds at both
    boundaries that consume the accumulated reset masks (``forward()`` and ``step()``), while the
    authored joint state and the untouched worlds' history are preserved.
    """
    sim_cfg = SimulationCfg(
        dt=1 / 120,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=20,
                nconmax=20,
                integrator="implicitfast",
            ),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )
    with build_simulation_context(sim_cfg=sim_cfg, device=device) as sim:
        sim._app_control_on_stop_handle = None
        articulation = _generate_single_joint_articulations(num_articulations=2, device=device)
        sim.reset()

        solver = SimulationManager._solver
        assert isinstance(solver, SolverMuJoCo)
        warm_start = wp.to_torch(solver.mjw_data.qacc_warmstart)
        assert warm_start.shape[0] == 2
        assert warm_start.shape[1] > 0

        env_ids = torch.tensor([0], dtype=torch.int32, device=device)
        joint_pos = articulation.data.default_joint_pos.torch[:1].clone() + 0.25
        joint_vel = torch.full_like(articulation.data.default_joint_vel.torch[:1], 0.5)
        articulation.write_joint_state_to_sim_index(
            position=joint_pos,
            velocity=joint_vel,
            env_ids=env_ids,
        )

        state = SimulationManager._state_0
        joint_q_before = wp.to_torch(state.joint_q).clone()
        joint_qd_before = wp.to_torch(state.joint_qd).clone()
        warm_start[0].fill_(13.0)
        warm_start[1].fill_(17.0)
        wp.synchronize_device(device)

        # The public, non-integrating forward boundary consumes the authored-write mask.
        sim.forward()
        wp.synchronize_device(device)

        torch.testing.assert_close(wp.to_torch(state.joint_q), joint_q_before)
        torch.testing.assert_close(wp.to_torch(state.joint_qd), joint_qd_before)
        assert torch.count_nonzero(warm_start[0]).item() == 0
        torch.testing.assert_close(warm_start[1], torch.full_like(warm_start[1], 17.0))

        # The physics-step boundary also resets when no forward boundary consumed the mask first.
        articulation.write_joint_state_to_sim_index(
            position=joint_pos,
            velocity=joint_vel,
            env_ids=env_ids,
        )
        warm_start[0].fill_(23.0)
        warm_start[1].fill_(29.0)
        wp.synchronize_device(device)

        with (
            patch.object(SimulationManager, "_simulate_full", classmethod(lambda cls: None)),
            patch.object(SimulationManager, "_simulate_physics_only", classmethod(lambda cls: None)),
        ):
            sim.step(render=False)
        wp.synchronize_device(device)

        torch.testing.assert_close(wp.to_torch(state.joint_q), joint_q_before)
        torch.testing.assert_close(wp.to_torch(state.joint_qd), joint_qd_before)
        assert torch.count_nonzero(warm_start[0]).item() == 0
        torch.testing.assert_close(warm_start[1], torch.full_like(warm_start[1], 29.0))
