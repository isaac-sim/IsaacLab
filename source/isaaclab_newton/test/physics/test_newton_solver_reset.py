# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the solver-internal reset performed when env-reset masks are consumed."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation()

from unittest.mock import patch

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonBackend, NewtonCfg
from isaaclab_physx.sim.schemas import PhysxJointCfg
from newton.solvers import SolverMuJoCo

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.warp.capture import GraphedCallable


def _generate_single_joint_articulations(num_articulations: int, device: str) -> Articulation:
    """Spawn ``num_articulations`` copies of the simple revolute articulation, one per env prim."""
    sim_utils.create_prim("/World/Env_0", "Xform")
    articulation_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
            joint_drive_props=[sim_utils.UsdPhysicsDriveCfg(max_force=80.0), PhysxJointCfg(max_joint_velocity=5.0)],
        ),
        actuators={
            "joint": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                actuator_effort_limit=400.0,
                actuator_velocity_limit=100.0,
                stiffness=0.0,
                damping=10.0,
            ),
        },
    )
    positions = np.zeros((num_articulations, 3), dtype=np.float32)
    positions[:, 0] = np.arange(num_articulations) * 2.5
    plan = clone_plan_from_env_0(
        CloneCfg(clone_template="/World/Env_{}"), (articulation_cfg,), num_articulations, 2.5, positions=positions
    )
    articulation = Articulation(articulation_cfg)
    replicate(plan)
    return articulation


def _no_substeps(self, contacts, *, observables):
    """Stand-in for :meth:`NewtonBackend._run_solver_substeps` that leaves the state and solver history as they are."""


def _sim_cfg(use_cuda_graph: bool) -> SimulationCfg:
    return SimulationCfg(
        dt=1 / 120,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=20, nconmax=20, integrator="implicitfast"),
            num_substeps=1,
            use_cuda_graph=use_cuda_graph,
        ),
    )


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
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

        solver = sim.backend_manager.backend.solver
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

        state = sim.backend_manager.backend.state_0
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

        backend = sim.backend_manager.backend
        backend.invalidate_graph()
        with patch.object(NewtonBackend, "_run_solver_substeps", _no_substeps):
            sim.step(render=False)
        backend.invalidate_graph()
        wp.synchronize_device(device)

        torch.testing.assert_close(wp.to_torch(state.joint_q), joint_q_before)
        torch.testing.assert_close(wp.to_torch(state.joint_qd), joint_qd_before)
        assert torch.count_nonzero(warm_start[0]).item() == 0
        torch.testing.assert_close(warm_start[1], torch.full_like(warm_start[1], 29.0))


@pytest.mark.parametrize("use_cuda_graph", [False, True], ids=["eager_step", "captured_step"])
@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_the_step_after_a_replayed_state_write_resolves_it_as_after_an_eager_write(device, use_cuda_graph):
    """A joint write recorded by a :class:`GraphedCallable` flags its world on the device only. On every replay, the
    next physics step, eager or replayed from its own graph, resets that world's solver history, keeps the other
    worlds' history and evaluates its forward kinematics, as after the same write made eagerly."""
    with build_simulation_context(sim_cfg=_sim_cfg(use_cuda_graph), device=device) as sim:
        sim._app_control_on_stop_handle = None
        articulation = _generate_single_joint_articulations(num_articulations=2, device=device)
        sim.reset()
        backend = sim.backend_manager.backend
        warm_start = wp.to_torch(backend.solver.mjw_data.qacc_warmstart)
        env_mask = wp.array([True, False], dtype=wp.bool, device=device)
        default_joint_pos = articulation.data.default_joint_pos.torch.clone()
        joint_pos = wp.from_torch(default_joint_pos.clone())

        def reset_env_0():
            articulation.write_joint_position_to_sim_mask(position=joint_pos, env_mask=env_mask)

        reset = GraphedCallable(reset_env_0, device)
        try:
            with patch.object(NewtonBackend, "_run_solver_substeps", _no_substeps):
                backend.invalidate_graph()
                for offset in (0.25, 0.5, 1.0):  # the first write and step record, the others replay
                    wp.to_torch(joint_pos).copy_(default_joint_pos + offset)
                    warm_start[0].fill_(13.0)
                    warm_start[1].fill_(17.0)
                    reset()
                    sim.step(render=False)
                    wp.synchronize_device(device)
                    assert torch.count_nonzero(warm_start[0]).item() == 0, f"offset {offset}"
                    torch.testing.assert_close(warm_start[1], torch.full_like(warm_start[1], 17.0))
                    body_q = wp.to_torch(backend.state_0.body_q).clone()
                    # the same write made eagerly, resolved by the forward boundary
                    articulation.write_joint_position_to_sim_mask(position=joint_pos, env_mask=env_mask)
                    backend.forward()
                    wp.synchronize_device(device)
                    torch.testing.assert_close(body_q, wp.to_torch(backend.state_0.body_q), msg=f"offset {offset}")
                assert bool(backend.graphs) == use_cuda_graph
        finally:
            reset.reset()


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_a_recorded_read_after_a_recorded_write_sees_its_forward_kinematics_on_every_replay(device):
    """A read recorded after an eager read consumed the pending FK of a recorded write still evaluates FK on every
    replay, so it reads the body poses of each replayed write."""
    with build_simulation_context(sim_cfg=_sim_cfg(use_cuda_graph=False), device=device) as sim:
        sim._app_control_on_stop_handle = None
        articulation = _generate_single_joint_articulations(num_articulations=2, device=device)
        sim.reset()
        env_mask = wp.array([True, False], dtype=wp.bool, device=device)
        default_joint_pos = articulation.data.default_joint_pos.torch.clone()
        joint_pos = wp.from_torch(default_joint_pos.clone())
        poses = wp.zeros_like(articulation.data.body_link_pose_w.warp)

        def reset_env_0():
            articulation.write_joint_position_to_sim_mask(position=joint_pos, env_mask=env_mask)

        def read_body_poses():
            wp.copy(poses, articulation.data.body_link_pose_w.warp)

        reset, read = GraphedCallable(reset_env_0, device), GraphedCallable(read_body_poses, device)
        try:
            for offset in (0.25, 0.5, 1.0):  # the first calls record, the others replay
                wp.to_torch(joint_pos).copy_(default_joint_pos + offset)
                reset()
                if offset == 0.25:
                    _ = articulation.data.body_link_pose_w  # an eager read consumes the pending FK
                read()
                wp.synchronize_device(device)
                # the same write and read without a recording
                articulation.write_joint_position_to_sim_mask(position=joint_pos, env_mask=env_mask)
                expected = articulation.data.body_link_pose_w.torch
                torch.testing.assert_close(wp.to_torch(poses), expected, msg=f"offset {offset}")
        finally:
            reset.reset()
            read.reset()
