# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline reproduction of the franka_fr3 ros2_control target pipeline.

The real interface (isaac_ros_robots/franka_fr3, ``FrankaFr3SystemInterface``)
processes incoming position commands on the 1 kHz FCI thread:

    latest 200 Hz command (ZOH) -> clamp to joint limits - 0.05 rad margin
    -> EMA low-pass (alpha = 0.02, tau ~ 50 ms)
    -> Ruckig OTG (vel/acc/jerk limits scaled by ``relative_dynamics``)
    -> libfranka ControllerMode::kJointImpedance

The steady-state pipeline is deterministic and independent of the measured
robot state, so the impedance targets the firmware tracked can be reconstructed
from the recorded ``des_dof_pos``. The reconstruction is APPROXIMATE at the
start: the hardware shaper's internal state at recording t=0 (persisted through
homing + dwell) is unobserved. We seed it settled at ``init_target`` (the last
pre-recording command, i.e. ``des_dof_pos[0]``); the residual decays with the
EMA time constant (~50 ms) and the fitting loss masks a burn-in window sized by
:func:`data_contract.compute_burn_in_s`. Exact reconstruction requires
driver-exported applied targets. Constants mirror
``franka_fr3_system_interface.cpp``.
"""

from __future__ import annotations

import numpy as np

FR3_JOINT_MIN = np.array([-2.74, -1.79, -2.90, -3.04, -2.81, 0.54, -3.02])
FR3_JOINT_MAX = np.array([2.74, 1.79, 2.90, -0.15, 2.81, 4.52, 3.02])
FR3_VELOCITY_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
FR3_ACCELERATION_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
FR3_JERK_MAX = np.array([7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0])
JOINT_LIMIT_MARGIN = 0.05
TARGET_EMA_ALPHA = 0.02
DEFAULT_RELATIVE_DYNAMICS = 0.3


def shape_targets(
    des: np.ndarray,
    init_target: np.ndarray,
    substeps: int,
    joint_indices: list[int] | None = None,
    relative_dynamics: float = DEFAULT_RELATIVE_DYNAMICS,
    ema_alpha: float = TARGET_EMA_ALPHA,
    ctrl_dt: float = 0.001,
) -> np.ndarray:
    """Reconstruct the 1 kHz impedance targets from recorded commands.

    Args:
        des: recorded commands at the command rate, shape (T, 7), in DATA
            column order.
        init_target: settled shaper state at t=0 — the last pre-recording
            command (use ``des[0]``), shape (7,).
        substeps: controller ticks per command sample (e.g. 1000/200 = 5).
        joint_indices: canonical FR3 joint index of each data column (from
            :func:`data_contract.canonical_indices`). Required whenever the
            data columns are not already in fr3_joint1..7 order — the
            per-joint clamp and Ruckig limits are permuted accordingly.
        relative_dynamics: Ruckig limit scale — MUST match the deployed
            hardware parameter.
        ema_alpha: target low-pass coefficient per controller tick.
        ctrl_dt: controller tick period.

    Returns:
        shaped targets, shape (T, substeps, 7), in the same column order as
        ``des``.
    """
    try:
        from ruckig import InputParameter, OutputParameter, Result, Ruckig
    except ImportError as e:
        raise ImportError(
            "the 'ruckig' python package is required to reproduce the franka_fr3 target shaping (pip install ruckig)"
        ) from e

    T, n = des.shape
    if n != 7:
        raise ValueError(f"franka_fr3 shaping expects 7 joints, got {n}")
    idx = np.arange(n) if joint_indices is None else np.asarray(joint_indices)
    if sorted(idx.tolist()) != list(range(n)):
        raise ValueError(f"joint_indices must be a permutation of 0..{n - 1}, got {idx.tolist()}")

    otg = Ruckig(n, ctrl_dt)
    inp = InputParameter(n)
    out = OutputParameter(n)
    init = np.asarray(init_target, dtype=float)
    inp.current_position = list(init)
    inp.current_velocity = [0.0] * n
    inp.current_acceleration = [0.0] * n
    inp.max_velocity = list(FR3_VELOCITY_MAX[idx] * relative_dynamics)
    inp.max_acceleration = list(FR3_ACCELERATION_MAX[idx] * relative_dynamics)
    inp.max_jerk = list(FR3_JERK_MAX[idx] * relative_dynamics)
    inp.target_velocity = [0.0] * n
    inp.target_acceleration = [0.0] * n

    lo = FR3_JOINT_MIN[idx] + JOINT_LIMIT_MARGIN
    hi = FR3_JOINT_MAX[idx] - JOINT_LIMIT_MARGIN

    filtered = init.copy()
    shaped = np.zeros((T, substeps, n))
    for i in range(T):
        raw = np.clip(des[i], lo, hi)  # latest command, zero-order-held below
        for s in range(substeps):
            filtered += ema_alpha * (raw - filtered)
            inp.target_position = list(filtered)
            result = otg.update(inp, out)
            if result in (Result.Working, Result.Finished):
                shaped[i, s] = out.new_position
                out.pass_to_input(inp)
            else:
                # The real interface holds the previous trajectory state on error.
                shaped[i, s] = inp.current_position
    return shaped
