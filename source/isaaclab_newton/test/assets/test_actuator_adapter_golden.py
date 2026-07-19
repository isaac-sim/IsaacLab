# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bitwise golden captures of the Newton actuator adapter.

Freezes the adapter's engine-channel outputs on a fixed heterogeneous
scenario so that adapter refactors (which keep the same Newton kernels) are
refereed bitwise, independent of the Lab actuator models. The tolerance-based
Lab-vs-Newton equivalence lives in ``test_newton_actuators_newton.py``; this
file answers a different question: did a refactor change ANY bit of what the
adapter writes?

Scenario: ANYmal-C (IdealPD) + Cartpole (explicit PD) per env — mixed DOF
counts and base types. Each recorded tick starts from an EXPLICITLY WRITTEN
joint state, so the recorded channels are the actuator stages' output on a
deterministic input: the adapter writes ``joint_f`` before the solver
integrates, which keeps the capture bitwise-stable even though multi-step
trajectories are not (MuJoCo-Warp's FP-atomic solve is not run-to-run
deterministic). Tick 2 follows a partial reset to pin the reset-mask path.

The golden is machine-generated: when the file is absent the test writes it
and skips. Regenerate by deleting the file and re-running. Goldens pin the
kernel outputs of this device class; a Warp or driver upgrade that changes
codegen legitimately regenerates them.
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import os

import numpy as np
import torch
import warp as wp
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.sim import SimulationCfg, build_simulation_context

from isaaclab_assets import ANYMAL_C_CFG, CARTPOLE_CFG

NUM_ENVS = 2
DT = 1.0 / 120.0
TARGET_OFFSET = 0.1

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "data", "actuator_adapter_golden.npz")

NEWTON_CFG = NewtonCfg(solver_cfg=MJWarpSolverCfg(), use_cuda_graph=True)

ANYMAL_ACTUATORS = {
    "legs": IdealPDActuatorCfg(
        joint_names_expr=[".*"],
        stiffness=60.0,
        damping=4.0,
        effort_limit=80.0,
        velocity_limit=7.5,
    )
}
CARTPOLE_ACTUATORS = {
    "cart": IdealPDActuatorCfg(
        joint_names_expr=["slider_to_cart"],
        stiffness=10.0,
        damping=2.0,
        effort_limit=400.0,
        velocity_limit=100.0,
    ),
    "pole": IdealPDActuatorCfg(
        joint_names_expr=["cart_to_pole"],
        stiffness=0.0,
        damping=0.0,
        effort_limit=400.0,
        velocity_limit=100.0,
    ),
}


def _write_state(art, offset: float) -> None:
    """Write a deterministic joint state derived from the default state."""
    default_pos = wp.to_torch(art.data.default_joint_pos).clone()
    pattern = torch.arange(default_pos.shape[1], device=default_pos.device, dtype=default_pos.dtype)
    pos = default_pos + offset + 0.01 * pattern.unsqueeze(0)
    vel = torch.full_like(pos, 0.05) + 0.002 * pattern.unsqueeze(0)
    art.write_joint_state_to_sim(pos, vel)


def _record(channels: dict, anymal, cartpole) -> None:
    channels.setdefault("anymal_applied", []).append(
        wp.to_torch(anymal.data._sim_bind_joint_effort).cpu().numpy().copy()
    )
    channels.setdefault("anymal_computed", []).append(
        wp.to_torch(anymal.data._sim_bind_joint_computed_effort).cpu().numpy().copy()
    )
    channels.setdefault("cartpole_applied", []).append(
        wp.to_torch(cartpole.data._sim_bind_joint_effort).cpu().numpy().copy()
    )


def _run_golden_scenario() -> dict[str, np.ndarray]:
    """Run the fixed scenario and return the per-tick engine channels."""
    sim_cfg = SimulationCfg(dt=DT, physics=NEWTON_CFG, use_newton_actuators=True)
    with build_simulation_context(
        device="cuda:0",
        gravity_enabled=True,
        add_ground_plane=True,
        sim_cfg=sim_cfg,
    ) as sim:
        sim._app_control_on_stop_handle = None
        for i in range(NUM_ENVS):
            sim_utils.create_prim(f"/World/Env_{i}", "Xform", translation=(i * 6.0, 0, 0))
        anymal_cfg = ANYMAL_C_CFG.replace(actuators=ANYMAL_ACTUATORS, prim_path="/World/Env_.*/Anymal")
        cartpole_cfg = CARTPOLE_CFG.replace(actuators=CARTPOLE_ACTUATORS, prim_path="/World/Env_.*/Cartpole")
        cartpole_cfg.init_state = cartpole_cfg.init_state.replace(pos=(0.0, 3.0, 2.0))
        anymal = Articulation(anymal_cfg)
        cartpole = Articulation(cartpole_cfg)
        sim.reset()
        assert anymal.is_initialized and cartpole.is_initialized

        for art in (anymal, cartpole):
            init_pos = wp.to_torch(art.data.joint_pos).clone()
            art.set_joint_position_target_index(target=init_pos + TARGET_OFFSET)
            art.set_joint_velocity_target_index(target=torch.zeros_like(init_pos))

        channels: dict[str, list[np.ndarray]] = {}

        # tick 1: actuator output on an explicitly written deterministic state
        _write_state(anymal, offset=0.02)
        _write_state(cartpole, offset=-0.03)
        anymal.write_data_to_sim()
        cartpole.write_data_to_sim()
        sim.step()
        _record(channels, anymal, cartpole)

        # tick 2: after a partial reset (pins the reset-mask path), again from
        # a written deterministic state
        anymal.reset(env_ids=torch.tensor([0], device="cuda:0"))
        _write_state(anymal, offset=-0.01)
        _write_state(cartpole, offset=0.04)
        anymal.write_data_to_sim()
        cartpole.write_data_to_sim()
        sim.step()
        _record(channels, anymal, cartpole)

    return {name: np.stack(steps) for name, steps in channels.items()}


def test_adapter_matches_golden():
    """The adapter's engine-channel outputs are bit-identical to the stored golden."""
    import pytest  # noqa: PLC0415

    recorded = _run_golden_scenario()
    if not os.path.exists(GOLDEN_PATH):
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        np.savez(GOLDEN_PATH, **recorded)
        pytest.skip(f"golden generated at {GOLDEN_PATH}; commit it and re-run")
    golden = np.load(GOLDEN_PATH)
    for name, values in recorded.items():
        assert np.array_equal(values, golden[name]), (
            f"channel '{name}' diverged bitwise from the golden capture; "
            f"max abs diff {np.abs(values - golden[name]).max():.3e}. If the change is "
            "intentional (kernel or contract change), delete the golden and regenerate."
        )
