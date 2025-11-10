# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher

HEADLESS = True

# if not AppLauncher.instance():
simulation_app = AppLauncher(headless=HEADLESS).app

"""Rest of imports follows"""

import torch

import pytest
import isaaclab.utils.math as math_utils
from types import SimpleNamespace


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
    from isaaclab.actuators import Thruster

    num_envs = 2
    num_motors = 2
    cfg = make_thruster_cfg(num_motors)
    cfg.thrust_const_range = (0.0, 0.0)

    thruster_names = [f"t{i}" for i in range(num_motors)]
    thruster_ids = slice(None)
    init_rps = torch.ones(num_envs, num_motors)

    thr = Thruster(cfg, thruster_names, thruster_ids, num_envs, "cpu", init_rps)  # type: ignore[arg-type]

    command = torch.full((num_envs, num_motors), 1.0)
    action = SimpleNamespace(thrusts=command.clone(), thruster_indices=thruster_ids)

    thr.compute(action)  # type: ignore[arg-type]

    assert torch.isfinite(action.thrusts).all()


def test_negative_thrust_range_results_finite():
    """Negative configured thrust ranges are clamped and yield finite outputs after hardening."""
    from isaaclab.actuators import Thruster

    num_envs = 2
    num_motors = 2
    cfg = make_thruster_cfg(num_motors)
    cfg.thrust_range = (-5.0, -1.0)
    cfg.thrust_const_range = (0.05, 0.05)

    thruster_names = [f"t{i}" for i in range(num_motors)]
    thruster_ids = slice(None)
    init_rps = torch.ones(num_envs, num_motors)

    thr = Thruster(cfg, thruster_names, thruster_ids, num_envs, "cpu", init_rps)  # type: ignore[arg-type]

    command = torch.full((num_envs, num_motors), -2.0)
    action = SimpleNamespace(thrusts=command.clone(), thruster_indices=thruster_ids)

    thr.compute(action)  # type: ignore[arg-type]

    assert torch.isfinite(action.thrusts).all()


def test_tensor_vs_slice_indices_and_subset_reset():
    """Compute should accept tensor or slice thruster indices, and reset_idx should affect only specified envs."""
    from isaaclab.actuators import Thruster

    num_envs = 3
    num_motors = 4
    cfg = make_thruster_cfg(num_motors)

    thruster_names = [f"t{i}" for i in range(num_motors)]
    thruster_ids_tensor = torch.tensor([0, 2], dtype=torch.int64)
    thruster_ids_slice = slice(None)
    init_rps = torch.ones(num_envs, num_motors)

    thr_tensor = Thruster(cfg, thruster_names, thruster_ids_tensor, num_envs, "cpu", init_rps)  # type: ignore[arg-type]
    thr_slice = Thruster(cfg, thruster_names, thruster_ids_slice, num_envs, "cpu", init_rps)  # type: ignore[arg-type]

    command = torch.full((num_envs, num_motors), cfg.thrust_range[1] * 0.5)
    action_tensor = SimpleNamespace(thrusts=command.clone(), thruster_indices=thruster_ids_tensor)
    action_slice = SimpleNamespace(thrusts=command.clone(), thruster_indices=thruster_ids_slice)

    thr_tensor.compute(action_tensor)  # type: ignore[arg-type]
    thr_slice.compute(action_slice)  # type: ignore[arg-type]

    assert action_tensor.thrusts.shape == (num_envs, num_motors)
    assert action_slice.thrusts.shape == (num_envs, num_motors)

    prev = thr_tensor.curr_thrust.clone()
    thr_tensor.reset_idx(torch.tensor([2], dtype=torch.int64))
    assert not torch.allclose(prev[2], thr_tensor.curr_thrust[2])


def test_mixing_and_integration_modes():
    """Verify mixing factor selection and integration kernel choice reflect the config."""
    from isaaclab.actuators import Thruster

    num_envs = 1
    num_motors = 1
    cfg = make_thruster_cfg(num_motors)

    # discrete mixing
    cfg.use_discrete_approximation = True
    cfg.integration_scheme = "euler"
    thr_d = Thruster(cfg, ["t0"], slice(None), num_envs, "cpu", torch.ones(num_envs, num_motors))  # type: ignore[arg-type]
    # bound method objects are recreated on access; compare underlying functions instead
    assert getattr(thr_d.mixing_factor_function, "__func__", None) is Thruster.discrete_mixing_factor
    assert getattr(thr_d._step_thrust, "__func__", None) is Thruster.compute_thrust_with_rpm_time_constant

    # continuous mixing and RK4
    cfg.use_discrete_approximation = False
    cfg.integration_scheme = "rk4"
    thr_c = Thruster(cfg, ["t0"], slice(None), num_envs, "cpu", torch.ones(num_envs, num_motors))  # type: ignore[arg-type]
    assert getattr(thr_c.mixing_factor_function, "__func__", None) is Thruster.continuous_mixing_factor
    assert getattr(thr_c._step_thrust, "__func__", None) is Thruster.compute_thrust_with_rpm_time_constant_rk4


def test_thruster_compute_clamps_and_shapes():
    """Thruster.compute should return thrusts with correct shape and within clamp bounds."""
    from isaaclab.actuators import Thruster

    num_envs = 4
    num_motors = 3
    cfg = make_thruster_cfg(num_motors)

    thruster_names = [f"t{i}" for i in range(num_motors)]
    thruster_ids = slice(None)
    init_rps = torch.ones(num_envs, num_motors)

    thr = Thruster(cfg, thruster_names, thruster_ids, num_envs, "cpu", init_rps)  # type: ignore[arg-type]

    # command above max to check clamping
    command = torch.full((num_envs, num_motors), cfg.thrust_range[1] * 2.0)
    action = SimpleNamespace(thrusts=command.clone(), thruster_indices=thruster_ids)

    out = thr.compute(action)  # type: ignore[arg-type]

    assert out.thrusts.shape == (num_envs, num_motors)
    # values must be clipped to configured range
    assert torch.all(out.thrusts <= cfg.thrust_range[1] + 1e-6)
    assert torch.all(out.thrusts >= cfg.thrust_range[0] - 1e-6)


def test_thruster_reset_idx_changes_state():
    """reset_idx should re-sample parameters for specific env indices."""
    from isaaclab.actuators import Thruster

    num_envs = 3
    num_motors = 2
    cfg = make_thruster_cfg(num_motors)

    thruster_names = [f"t{i}" for i in range(num_motors)]
    thruster_ids = slice(None)
    init_rps = torch.ones(num_envs, num_motors)

    thr = Thruster(cfg, thruster_names, thruster_ids, num_envs, "cpu", init_rps)  # type: ignore[arg-type]

    # Mutate an internal sampled parameter so reset produces a measurable change.
    thr.tau_inc_s[0, 0] = thr.tau_inc_s[0, 0] + 1.0
    prev_val = thr.tau_inc_s[0, 0].item()

    # reset only environment 0
    thr.reset_idx(torch.tensor([0], dtype=torch.int64))

    # at least the first tau_inc value for env 0 should differ from the mutated value
    assert not torch.isclose(torch.tensor(prev_val), thr.tau_inc_s[0, 0])