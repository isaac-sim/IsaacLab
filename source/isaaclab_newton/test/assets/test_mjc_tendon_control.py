# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets.articulation.mjc_tendon_control import (
    MjcTendonControl,
    resolve_fixed_tendon_actuator_columns,
)
from newton.solvers import SolverMuJoCo

_TENDON = int(SolverMuJoCo.TrnType.TENDON)
_JOINT = 0


def _make_view_and_model(trntypes=(_JOINT, _TENDON, _TENDON), trn_targets=(0, 0, 2), tendon_count=3):
    """Build a view whose middle tendon is transmitted to by no actuator.

    Args:
        trntypes: ``mujoco:actuator_trntype`` per actuator column.
        trn_targets: ``mujoco:actuator_trnid`` target per actuator column.
        tendon_count: Number of fixed tendons the view reports.
    """
    actuator_count = len(trntypes)
    attrs = {
        "mujoco.actuator_trntype": np.array([[list(trntypes)]], dtype=np.int32),
        "mujoco.actuator_trnid": np.array([[[[t, 0] for t in trn_targets]]], dtype=np.int32),
    }
    view = SimpleNamespace(
        device=wp.get_device("cpu"),
        custom_frequency_counts={"mujoco:actuator": actuator_count, "mujoco:tendon": tendon_count},
        custom_frequency_labels={"mujoco:actuator": [f"act{i}" for i in range(actuator_count)]},
        get_attribute=lambda name, source: wp.array(attrs[name], dtype=wp.int32, device="cpu"),
    )
    return view, SimpleNamespace()


def test_each_tendon_maps_to_the_actuator_that_transmits_to_it():
    view, model = _make_view_and_model()

    columns = resolve_fixed_tendon_actuator_columns(view, model)

    # actuator 1 transmits to tendon 0 and actuator 2 to tendon 2; tendon 1 has none.
    np.testing.assert_array_equal(columns, [1, -1, 2])


def test_a_model_with_no_tendon_actuators_resolves_to_nothing():
    view, model = _make_view_and_model(trntypes=(_JOINT, _JOINT), trn_targets=(0, 1))

    assert resolve_fixed_tendon_actuator_columns(view, model) is None


def test_a_view_without_the_actuator_frequency_resolves_to_nothing():
    view, model = _make_view_and_model()
    view.custom_frequency_counts["mujoco:actuator"] = 0

    assert resolve_fixed_tendon_actuator_columns(view, model) is None


def test_a_target_outside_the_tendon_range_is_ignored():
    view, model = _make_view_and_model(trn_targets=(0, 0, 99))

    np.testing.assert_array_equal(resolve_fixed_tendon_actuator_columns(view, model), [1, -1, -1])


def test_tendons_no_actuator_transmits_to_are_named_once(caplog):
    articulation = SimpleNamespace(
        device=wp.get_device("cpu"),
        fixed_tendon_names=["rh_FFJ0", "passive", "rh_LFJ0"],
    )

    with caplog.at_level(logging.WARNING):
        MjcTendonControl(articulation, np.array([1, -1, 2], dtype=np.int32))

    assert "passive" in caplog.text
    assert "rh_FFJ0" not in caplog.text


@pytest.mark.parametrize("columns", [[1, 2, 3], [0, 0, 0]])
def test_a_fully_actuated_model_warns_about_nothing(caplog, columns):
    articulation = SimpleNamespace(
        device=wp.get_device("cpu"),
        fixed_tendon_names=["a", "b", "c"],
    )

    with caplog.at_level(logging.WARNING):
        MjcTendonControl(articulation, np.array(columns, dtype=np.int32))

    assert caplog.text == ""
