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


def _make_view_and_model(trntypes=(_JOINT, _TENDON, _TENDON), tendon_count=2, targets=None):
    """Build a view with ``trntypes`` actuators over ``tendon_count`` tendons named ``t0, t1, ...``.

    ``mujoco:actuator_trnid`` is deliberately absent: the USD importer never writes it for a
    tendon actuator, so the resolver must pair by target label, as the solver does.

    Args:
        trntypes: ``mujoco:actuator_trntype`` per actuator column.
        tendon_count: Number of fixed tendons the view reports.
        targets: Tendon name each tendon actuator targets, in column order. Defaults to the tendons
            in declaration order.
    """
    actuator_count = len(trntypes)
    tendon_columns = [i for i, trntype in enumerate(trntypes) if trntype == _TENDON]
    if targets is None:
        targets = [f"t{k}" for k in range(len(tendon_columns))]
    labels = [f"/robot/joints/j{i}" for i in range(actuator_count)]
    for column, target in zip(tendon_columns, targets, strict=True):
        labels[column] = f"/robot/tendons/{target}"
    attrs = {"mujoco.actuator_trntype": np.array([[list(trntypes)]], dtype=np.int32)}
    view = SimpleNamespace(
        device=wp.get_device("cpu"),
        custom_frequency_counts={"mujoco:actuator": actuator_count, "mujoco:tendon": tendon_count},
        custom_frequency_labels={"mujoco:actuator": labels},
        tendon_names=[f"t{k}" for k in range(tendon_count)],
        get_attribute=lambda name, source: wp.array(attrs[name], dtype=wp.int32, device="cpu"),
    )
    return view, SimpleNamespace()


def _make_articulation(fixed_tendon_names, actuator_count=3, count_per_world=1):
    """Build the minimal articulation ``MjcTendonControl`` reads: names, device, and a root view.

    Args:
        fixed_tendon_names: Tendon names the warning path reports.
        actuator_count: ``mujoco:actuator`` count the command buffer is sized from.
        count_per_world: Instances of this articulation the view holds per world.
    """
    return SimpleNamespace(
        device=wp.get_device("cpu"),
        fixed_tendon_names=list(fixed_tendon_names),
        root_view=SimpleNamespace(
            world_count=1,
            count_per_world=count_per_world,
            custom_frequency_counts={"mujoco:actuator": actuator_count},
        ),
    )


def test_tendon_actuators_pair_with_tendons_by_target_label():
    # Actuator 0 drives a joint; column 1 targets t1 and column 2 targets t0, so declaration order
    # would pair them backwards. The label decides.
    view, model = _make_view_and_model(targets=["t1", "t0"])

    columns = resolve_fixed_tendon_actuator_columns(view, model)

    np.testing.assert_array_equal(columns, [2, 1])


def test_a_model_with_no_tendon_actuators_resolves_to_nothing():
    view, model = _make_view_and_model(trntypes=(_JOINT, _JOINT))

    assert resolve_fixed_tendon_actuator_columns(view, model) is None


def test_a_view_without_the_actuator_frequency_resolves_to_nothing():
    view, model = _make_view_and_model()
    view.custom_frequency_counts["mujoco:actuator"] = 0

    assert resolve_fixed_tendon_actuator_columns(view, model) is None


def test_a_tendon_no_actuator_drives_gets_no_column():
    """A passive tendon keeps column -1; the other tendons still find their actuators."""
    view, model = _make_view_and_model(tendon_count=3)

    np.testing.assert_array_equal(resolve_fixed_tendon_actuator_columns(view, model), [1, 2, -1])


def test_an_actuator_targeting_an_unknown_tendon_is_refused():
    view, model = _make_view_and_model(targets=["t0", "nope"])

    with pytest.raises(ValueError, match="not one of this articulation's fixed tendons"):
        resolve_fixed_tendon_actuator_columns(view, model)


def test_two_actuators_on_one_tendon_are_refused():
    view, model = _make_view_and_model(targets=["t0", "t0"])

    with pytest.raises(ValueError, match="one actuator per tendon"):
        resolve_fixed_tendon_actuator_columns(view, model)


def test_tendons_no_actuator_transmits_to_are_named_once(caplog):
    articulation = _make_articulation(["rh_FFJ0", "passive", "rh_LFJ0"])

    with caplog.at_level(logging.WARNING):
        MjcTendonControl(articulation, np.array([1, -1, 2], dtype=np.int32), articulation.root_view)

    assert "passive" in caplog.text
    assert "rh_FFJ0" not in caplog.text


def test_a_fully_actuated_model_warns_about_nothing(caplog):
    articulation = _make_articulation(["a", "b", "c"])

    with caplog.at_level(logging.WARNING):
        MjcTendonControl(articulation, np.array([1, 2, 3], dtype=np.int32), articulation.root_view)

    assert caplog.text == ""


def test_several_instances_per_world_is_refused_rather_than_mispaired():
    """The command buffer is written one instance per environment, so refuse any other layout.

    Writing ``commands[env_id, 0, column]`` is only correct while the view holds one instance per
    world. Silently keeping that indexing for a multi-instance view would drive the wrong
    articulation's tendons, which is indistinguishable from a tuning problem downstream.
    """
    articulation = _make_articulation(["rh_FFJ0", "rh_LFJ0"], count_per_world=2)

    with pytest.raises(NotImplementedError, match="instances per world"):
        MjcTendonControl(articulation, np.array([1, 2], dtype=np.int32), articulation.root_view)


def test_the_mask_command_writes_only_the_selected_cells():
    """The mask form must reach the same buffer as the index form, honouring both masks.

    Newton exposes masks so the write can be captured in a CUDA graph, so this path resolves
    ``None`` to a full mask rather than converting to indices, which would read on the host.
    """
    articulation = _make_articulation(["rh_FFJ0", "rh_MFJ0"])
    buffer = wp.zeros((3, 2), dtype=wp.float32, device="cpu")
    articulation.data = SimpleNamespace(_fixed_tendon_position_target=buffer)
    articulation.assert_shape_and_dtype = lambda *a, **k: None
    articulation._resolve_mask = lambda mask, full: full if mask is None else mask
    articulation._ALL_ENV_MASK = wp.array([True, True, True], dtype=wp.bool, device="cpu")
    articulation._ALL_FIXED_TENDON_MASK = wp.array([True, True], dtype=wp.bool, device="cpu")

    control = MjcTendonControl(articulation, np.array([1, 2], dtype=np.int32), articulation.root_view)
    target = wp.array(np.full((3, 2), 7.0, dtype=np.float32), dtype=wp.float32, device="cpu")

    control.set_position_target_mask(
        target=target,
        env_mask=wp.array([True, False, True], dtype=wp.bool, device="cpu"),
        fixed_tendon_mask=wp.array([False, True], dtype=wp.bool, device="cpu"),
    )

    # Only envs 0 and 2, and only tendon 1, are selected.
    np.testing.assert_array_equal(buffer.numpy(), [[0.0, 7.0], [0.0, 0.0], [0.0, 7.0]])
