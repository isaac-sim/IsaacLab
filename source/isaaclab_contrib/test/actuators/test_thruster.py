# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher

HEADLESS = True

# if not AppLauncher.instance():
simulation_app = AppLauncher(headless=HEADLESS).app

"""Rest of imports follows"""

import math
from types import SimpleNamespace

import pytest
import torch

# The thruster model is pure torch bookkeeping; env/motor counts and device select no distinct branch.
NUM_ENVS = 2
NUM_MOTORS = 4
DEVICE = "cpu"


def make_thruster_cfg(num_motors: int):
    """Create a minimal Thruster-like config object for tests."""
    return SimpleNamespace(
        dt=0.01,
        num_motors=num_motors,
        thrust_range=(0.0, 10.0),
        max_thrust_rate=100.0,
        thrust_const_range=(0.05, 0.1),
        tau_inc_range=(0.01, 0.02),
        tau_dec_range=(0.01, 0.02),
        torque_to_thrust_ratio=0.0,
        use_discrete_approximation=True,
        use_rps=True,
        integration_scheme="euler",
    )


def test_zero_thrust_const_is_handled():
    """When thrust_const_range contains zeros, Thruster clamps values and compute returns finite outputs."""
    from isaaclab_contrib.actuators import Thruster

    cfg = make_thruster_cfg(NUM_MOTORS)
    cfg.thrust_const_range = (0.0, 0.0)

    thruster_names = [f"t{i}" for i in range(NUM_MOTORS)]
    thruster_ids = slice(None)
    init_rps = torch.ones(NUM_ENVS, NUM_MOTORS, device=DEVICE)

    thr = Thruster(cfg, thruster_names, thruster_ids, NUM_ENVS, DEVICE, init_rps)  # type: ignore[arg-type]

    command = torch.full((NUM_ENVS, NUM_MOTORS), 1.0, device=DEVICE)
    action = SimpleNamespace(thrusts=command.clone(), thruster_indices=thruster_ids)

    thr.compute(action)  # type: ignore[arg-type]

    assert torch.isfinite(action.thrusts).all()


def test_negative_thrust_range_results_finite():
    """Negative configured thrust ranges are clamped and yield finite outputs after hardening."""
    from isaaclab_contrib.actuators import Thruster

    cfg = make_thruster_cfg(NUM_MOTORS)
    cfg.thrust_range = (-5.0, -1.0)
    cfg.thrust_const_range = (0.05, 0.05)

    thruster_names = [f"t{i}" for i in range(NUM_MOTORS)]
    thruster_ids = slice(None)
    init_rps = torch.ones(NUM_ENVS, NUM_MOTORS, device=DEVICE)

    thr = Thruster(cfg, thruster_names, thruster_ids, NUM_ENVS, DEVICE, init_rps)  # type: ignore[arg-type]

    command = torch.full((NUM_ENVS, NUM_MOTORS), -2.0, device=DEVICE)
    action = SimpleNamespace(thrusts=command.clone(), thruster_indices=thruster_ids)

    thr.compute(action)  # type: ignore[arg-type]

    assert torch.isfinite(action.thrusts).all()


def test_reset_idx_resamples_only_selected_envs():
    """reset_idx re-samples parameters and re-initializes thrust only for the selected env."""
    from isaaclab_contrib.actuators import Thruster

    num_envs = 3
    cfg = make_thruster_cfg(NUM_MOTORS)
    thruster_names = [f"t{i}" for i in range(NUM_MOTORS)]
    init_rps = torch.arange(1, num_envs + 1, dtype=torch.float32, device=DEVICE)[:, None].repeat(1, NUM_MOTORS)

    thr = Thruster(cfg, thruster_names, slice(None), num_envs, DEVICE, init_rps)  # type: ignore[arg-type]

    # Move the thrust state away from its initial value so re-initialization is observable.
    command = torch.full((num_envs, NUM_MOTORS), cfg.thrust_range[1] * 0.5, device=DEVICE)
    thr.compute(SimpleNamespace(thrusts=command, thruster_indices=slice(None)))  # type: ignore[arg-type]
    # Mutate a sampled parameter so re-sampling produces a measurable change.
    thr.tau_inc_s[0, 0] = thr.tau_inc_s[0, 0] + 1.0
    prev_val = thr.tau_inc_s[0, 0].item()
    prev_thrust = thr.curr_thrust.clone()
    prev_tau_inc = thr.tau_inc_s.clone()

    thr.reset_idx(torch.tensor([0], dtype=torch.int64, device=DEVICE))

    assert not torch.isclose(torch.tensor(prev_val, device=DEVICE), thr.tau_inc_s[0, 0])
    torch.testing.assert_close(thr.curr_thrust[0], thr.thrust_const[0] * init_rps[0] ** 2)
    torch.testing.assert_close(thr.curr_thrust[1:], prev_thrust[1:])
    torch.testing.assert_close(thr.tau_inc_s[1:], prev_tau_inc[1:])


@pytest.mark.parametrize(("use_discrete_approximation", "integration_scheme"), [(True, "euler"), (False, "rk4")])
def test_mixing_and_integration_modes(use_discrete_approximation, integration_scheme):
    """One compute step matches a hand-derived reference for each mixing and integration mode."""
    from isaaclab_contrib.actuators import Thruster

    cfg = make_thruster_cfg(NUM_MOTORS)
    cfg.use_discrete_approximation = use_discrete_approximation
    cfg.integration_scheme = integration_scheme
    cfg.max_thrust_rate = 1.0e6  # keep the rate clamp inactive
    cfg.thrust_const_range = (0.1, 0.1)
    cfg.tau_inc_range = (0.02, 0.02)
    cfg.tau_dec_range = (0.02, 0.02)
    thruster_names = [f"t{i}" for i in range(NUM_MOTORS)]
    init_rps = torch.ones(NUM_ENVS, NUM_MOTORS, device=DEVICE)

    thr = Thruster(cfg, thruster_names, slice(None), NUM_ENVS, DEVICE, init_rps)  # type: ignore[arg-type]
    command = torch.full((NUM_ENVS, NUM_MOTORS), 2.5, device=DEVICE)
    out = thr.compute(SimpleNamespace(thrusts=command, thruster_indices=slice(None)))  # type: ignore[arg-type]

    # rpm error decays as d(rpm)/dt = k * (rpm_des - rpm), with k = 1/(dt + tau) (discrete) or 1/tau (continuous).
    rpm, rpm_des = 1.0, 5.0  # sqrt(0.1 / 0.1), sqrt(2.5 / 0.1)
    k = 1.0 / (0.01 + 0.02) if use_discrete_approximation else 1.0 / 0.02
    h = 0.01 * k
    if integration_scheme == "euler":
        gain = h
    else:
        # RK4 on a linear ODE reproduces the 4th-order Taylor expansion of exp(-h).
        gain = 1.0 - sum((-h) ** n / math.factorial(n) for n in range(5))
    expected = 0.1 * (rpm + gain * (rpm_des - rpm)) ** 2
    torch.testing.assert_close(out.thrusts, torch.full_like(out.thrusts, expected))

    cfg.integration_scheme = "bad"
    with pytest.raises(ValueError, match="integration scheme unknown"):
        Thruster(cfg, thruster_names, slice(None), NUM_ENVS, DEVICE, init_rps)  # type: ignore[arg-type]


def test_thruster_compute_clamps_and_shapes():
    """Thruster.compute should return thrusts with correct shape and within clamp bounds."""
    from isaaclab_contrib.actuators import Thruster

    cfg = make_thruster_cfg(NUM_MOTORS)

    thruster_names = [f"t{i}" for i in range(NUM_MOTORS)]
    thruster_ids = slice(None)
    init_rps = torch.ones(NUM_ENVS, NUM_MOTORS, device=DEVICE)

    thr = Thruster(cfg, thruster_names, thruster_ids, NUM_ENVS, DEVICE, init_rps)  # type: ignore[arg-type]

    # command above max to check clamping
    command = torch.full((NUM_ENVS, NUM_MOTORS), cfg.thrust_range[1] * 2.0, device=DEVICE)
    action = SimpleNamespace(thrusts=command.clone(), thruster_indices=thruster_ids)

    out = thr.compute(action)  # type: ignore[arg-type]

    assert out.thrusts.shape == (NUM_ENVS, NUM_MOTORS)
    # values must be clipped to configured range
    assert torch.all(out.thrusts <= cfg.thrust_range[1] + 1e-6)
    assert torch.all(out.thrusts >= cfg.thrust_range[0] - 1e-6)
