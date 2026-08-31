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


def _make_view_and_model(trntypes=(_JOINT, _TENDON, _TENDON), tendon_count=2):
    """Build a view with ``trntypes`` actuators, of which the tendon ones drive ``tendon_count``.

    ``mujoco:actuator_trnid`` is deliberately absent: the USD importer never writes it for a
    tendon actuator, so the resolver must not depend on it.

    Args:
        trntypes: ``mujoco:actuator_trntype`` per actuator column.
        tendon_count: Number of fixed tendons the view reports.
    """
    actuator_count = len(trntypes)
    attrs = {"mujoco.actuator_trntype": np.array([[list(trntypes)]], dtype=np.int32)}
    view = SimpleNamespace(
        device=wp.get_device("cpu"),
        custom_frequency_counts={"mujoco:actuator": actuator_count, "mujoco:tendon": tendon_count},
        custom_frequency_labels={"mujoco:actuator": [f"act{i}" for i in range(actuator_count)]},
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


def test_tendon_actuators_pair_with_tendons_in_declaration_order():
    view, model = _make_view_and_model()

    columns = resolve_fixed_tendon_actuator_columns(view, model)

    # Actuator 0 drives a joint; columns 1 and 2 are the tendon actuators, paired to tendons 0
    # and 1 by declaration order because both orders come from the same stage traversal.
    np.testing.assert_array_equal(columns, [1, 2])


def test_a_model_with_no_tendon_actuators_resolves_to_nothing():
    view, model = _make_view_and_model(trntypes=(_JOINT, _JOINT))

    assert resolve_fixed_tendon_actuator_columns(view, model) is None


def test_a_view_without_the_actuator_frequency_resolves_to_nothing():
    view, model = _make_view_and_model()
    view.custom_frequency_counts["mujoco:actuator"] = 0

    assert resolve_fixed_tendon_actuator_columns(view, model) is None


def test_a_tendon_no_actuator_drives_is_refused_rather_than_guessed():
    """Ordering can only pair the two lists when they are the same length.

    With a passive tendon the counts differ and no ordering rule recovers which tendon is the
    unactuated one, so the mismatch has to be reported instead of silently shifting every pair.
    """
    view, model = _make_view_and_model(tendon_count=3)

    with pytest.raises(ValueError, match="3 fixed tendons but 2 actuators"):
        resolve_fixed_tendon_actuator_columns(view, model)


def test_tendons_no_actuator_transmits_to_are_named_once(caplog):
    articulation = _make_articulation(["rh_FFJ0", "passive", "rh_LFJ0"])

    with caplog.at_level(logging.WARNING):
        MjcTendonControl(articulation, np.array([1, -1, 2], dtype=np.int32), articulation.root_view)

    assert "passive" in caplog.text
    assert "rh_FFJ0" not in caplog.text


@pytest.mark.parametrize("columns", [[1, 2, 3], [0, 0, 0]])
def test_a_fully_actuated_model_warns_about_nothing(caplog, columns):
    articulation = _make_articulation(["a", "b", "c"])

    with caplog.at_level(logging.WARNING):
        MjcTendonControl(articulation, np.array(columns, dtype=np.int32), articulation.root_view)

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
