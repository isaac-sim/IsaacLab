# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the franka_fr3 command-shaping reconstruction."""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("ruckig")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fr3_target_shaping import (  # noqa: E402
    FR3_JOINT_MAX,
    FR3_JOINT_MIN,
    FR3_VELOCITY_MAX,
    JOINT_LIMIT_MARGIN,
    shape_targets,
)

READY = np.array([0.0, -0.7853981634, 0.0, -2.3561944902, 0.0, 1.5707963268, 0.7853981634])


def test_settled_constant_command_is_identity():
    des = np.tile(READY, (50, 1))
    shaped = shape_targets(des, init_target=READY, substeps=5)
    assert np.abs(shaped - READY).max() < 1e-12


def test_converges_to_step_command():
    target = READY.copy()
    target[5] += 0.1
    des = np.tile(target, (400, 1))  # 2 s at 200 Hz
    shaped = shape_targets(des, init_target=READY, substeps=5)
    assert np.abs(shaped[-1, -1] - target).max() < 1e-4


def test_velocity_limit_respected():
    target = READY.copy()
    target[0] += 1.0  # large step -> Ruckig must ramp
    des = np.tile(target, (200, 1))
    shaped = shape_targets(des, init_target=READY, substeps=5, relative_dynamics=0.3)
    flat = shaped.reshape(-1, 7)
    vel = np.abs(np.diff(flat[:, 0])) / 0.001
    assert vel.max() <= 0.3 * FR3_VELOCITY_MAX[0] * 1.01


def test_clamp_applied_per_canonical_joint_under_permutation():
    # Column 0 holds fr3_joint6 (canonical index 5, limits [0.54, 4.52]); a
    # command below its lower limit must clamp at min+margin — the j6 limit,
    # not the j1 limit that column 0 would get without joint_indices.
    names_idx = [5, 0, 1, 2, 3, 4, 6]  # data col -> canonical joint
    init = READY[names_idx]
    des = np.tile(init, (700, 1))  # 3.5 s: ~1 rad travel at the 0.78 rad/s cap + EMA lag
    des[:, 0] = 0.0  # below fr3_joint6 lower bound 0.54
    shaped = shape_targets(des, init_target=init, substeps=5, joint_indices=names_idx)
    expected = FR3_JOINT_MIN[5] + JOINT_LIMIT_MARGIN
    assert shaped[-1, -1, 0] == pytest.approx(expected, abs=1e-3)
    assert shaped[-1, -1, 0] > 0.5  # definitely not the j1-style clamp


def test_state_persists_across_samples():
    # A ramp processed in one call equals the same ramp processed sample by
    # sample only if EMA/Ruckig state carries across command samples — guard
    # against slice-and-reinitialize regressions.
    T = 100
    des = np.tile(READY, (T, 1))
    des[:, 2] = READY[2] + np.linspace(0.0, 0.05, T)
    full = shape_targets(des, init_target=READY, substeps=5)
    assert np.all(np.diff(full[:, :, 2].reshape(-1)) >= -1e-9)  # monotone ramp, no resets


def test_rejects_bad_joint_indices():
    des = np.tile(READY, (10, 1))
    with pytest.raises(ValueError, match="permutation"):
        shape_targets(des, init_target=READY, substeps=5, joint_indices=[0, 0, 1, 2, 3, 4, 5])


def test_rejects_wrong_joint_count():
    with pytest.raises(ValueError, match="7 joints"):
        shape_targets(np.zeros((10, 6)), init_target=np.zeros(6), substeps=5)


def test_targets_stay_inside_limits():
    rng = np.random.default_rng(0)
    des = READY + rng.uniform(-4.0, 4.0, size=(200, 7))  # wildly out of range
    shaped = shape_targets(des, init_target=READY, substeps=5)
    lo = FR3_JOINT_MIN + JOINT_LIMIT_MARGIN - 1e-9
    hi = FR3_JOINT_MAX - JOINT_LIMIT_MARGIN + 1e-9
    flat = shaped.reshape(-1, 7)
    assert (flat >= lo).all() and (flat <= hi).all()
