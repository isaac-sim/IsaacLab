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
    resolve_fixed_tendon_control_rows,
)
from newton.solvers import SolverMuJoCo

_TENDON = int(SolverMuJoCo.TrnType.TENDON)
_DIRECT = int(SolverMuJoCo.CtrlSource.CTRL_DIRECT)
_GLOBAL_WORLD = -1


def _make_root_view_and_model(actuator_worlds=(0, 0, 0, 1, 1, 1), extra_actuator=None):
    """Build a two-world model whose middle tendon in each world has no direct actuator.

    Args:
        actuator_worlds: ``mujoco:actuator_world`` per actuator row. ``-1`` scopes an actuator to
            every world.
        extra_actuator: Optional ``(target_label, world)`` appended as a second direct actuator, used
            to provoke the ambiguity rejection.
    """
    tendon_labels = ["/Robot/Physics/rh_FFJ0", "/Robot/Physics/passive", "/Robot/Physics/rh_LFJ0"] * 2
    actuator_target_labels = [
        "/Robot/joint",
        tendon_labels[0],
        tendon_labels[2],
        "/Robot/joint",
        tendon_labels[3],
        tendon_labels[5],
    ]
    trntypes = [0, _TENDON, _TENDON, 0, _TENDON, _TENDON]
    ctrl_sources = [_DIRECT] * 6
    actuator_worlds = list(actuator_worlds)
    if extra_actuator is not None:
        target_label, world = extra_actuator
        actuator_target_labels.append(target_label)
        trntypes.append(_TENDON)
        ctrl_sources.append(_DIRECT)
        actuator_worlds.append(world)

    mujoco = SimpleNamespace(
        tendon_label=tendon_labels,
        tendon_world=wp.array([0, 0, 0, 1, 1, 1], dtype=wp.int32, device="cpu"),
        actuator_target_label=actuator_target_labels,
        actuator_world=wp.array(actuator_worlds, dtype=wp.int32, device="cpu"),
        actuator_trntype=wp.array(trntypes, dtype=wp.int32, device="cpu"),
        ctrl_source=wp.array(ctrl_sources, dtype=wp.int32, device="cpu"),
    )
    tendon_layout = SimpleNamespace(
        offset=0,
        stride_between_worlds=3,
        stride_within_worlds=3,
        indices=None,
        slice=slice(0, 3),
    )
    root_view = SimpleNamespace(
        device=wp.get_device("cpu"),
        count=2,
        world_count=2,
        count_per_world=1,
        # Newton derives these from the same labels; the module checks its rows name the same tendons.
        tendon_names=["rh_FFJ0", "passive", "rh_LFJ0"],
        frequency_layouts={"mujoco:tendon": tendon_layout},
        # Newton applies the layout itself here; the module cross-checks its own addressing
        # against this read. Shape mirrors the view's (world_count, count_per_world, value_count).
        get_attribute=lambda name, source: wp.array(
            np.array([[[0, 0, 0]], [[1, 1, 1]]], dtype=np.int32), dtype=wp.int32, device="cpu"
        ),
    )
    return root_view, SimpleNamespace(mujoco=mujoco)


def test_control_rows_keep_the_full_fixed_tendon_id_space():
    """Map passive fixed tendons to -1 rather than compacting local tendon IDs.

    Compacting them would put the returned rows in a different index space from
    ``find_fixed_tendons``, so a caller's tendon ID would silently address another tendon.
    """
    root_view, model = _make_root_view_and_model()

    control_rows = resolve_fixed_tendon_control_rows(root_view, model)

    np.testing.assert_array_equal(control_rows, [[1, -1, 2], [4, -1, 5]])


def test_world_agnostic_actuator_drives_every_world():
    """Honour ``actuator_world == -1``, the attribute's own default, as a fallback per world."""
    root_view, model = _make_root_view_and_model(actuator_worlds=(0, _GLOBAL_WORLD, _GLOBAL_WORLD, 1, 1, 1))

    control_rows = resolve_fixed_tendon_control_rows(root_view, model)

    # World 1's tendons carry world-1 labels, so only world 0 falls back to the global actuators.
    np.testing.assert_array_equal(control_rows, [[1, -1, 2], [4, -1, 5]])


def test_actuator_shared_between_worlds_is_rejected():
    """Reject a world-agnostic actuator that every world falls back to.

    One ``ctrl`` row cannot carry a per-environment target: the scatter would have every
    environment write the same address, so they would silently drive each other's commands.
    """
    # No world-local tendon actuators at all, so both worlds resolve to the same global rows.
    root_view, model = _make_root_view_and_model(actuator_worlds=(0, _GLOBAL_WORLD, _GLOBAL_WORLD, 0, 0, 0))
    model.mujoco.actuator_target_label[4] = "/Robot/joint"
    model.mujoco.actuator_target_label[5] = "/Robot/joint"

    with pytest.raises(ValueError, match="drive more than one articulation instance"):
        resolve_fixed_tendon_control_rows(root_view, model)


def test_row_addressing_that_aliases_articulations_is_rejected():
    """Reject a layout whose strides map two instances onto the same tendons.

    Gathering a per-world value cannot catch this -- both reads agree precisely because the rows
    are the same -- so distinctness is the only check that sees it.
    """
    root_view, model = _make_root_view_and_model()
    root_view.frequency_layouts["mujoco:tendon"].stride_between_worlds = 0

    with pytest.raises(RuntimeError, match="duplicate rows"):
        resolve_fixed_tendon_control_rows(root_view, model)


def test_rows_naming_other_tendons_are_rejected():
    """Reject an offset error that stays in range and in-world.

    ``tendon_world`` is constant within a world, so only the tendon's identity reveals this.
    """
    root_view, model = _make_root_view_and_model()
    root_view.tendon_names = ["rh_FFJ0", "rh_MFJ0", "rh_LFJ0"]  # what the view believes it selected

    with pytest.raises(RuntimeError, match="ArticulationView names"):
        resolve_fixed_tendon_control_rows(root_view, model)


def test_two_actuators_on_one_tendon_are_rejected():
    """Reject an ambiguous target instead of silently picking whichever indexed first."""
    root_view, model = _make_root_view_and_model(extra_actuator=("/Robot/Physics/rh_FFJ0", 0))

    with pytest.raises(ValueError, match="Multiple direct MuJoCo tendon actuators"):
        resolve_fixed_tendon_control_rows(root_view, model)


def _make_articulation(position_target: wp.array) -> SimpleNamespace:
    """Stand in for the Newton articulation the control adapter drives."""
    return SimpleNamespace(
        device=wp.get_device("cpu"),
        fixed_tendon_names=["rh_FFJ0", "passive", "rh_LFJ0"],
        data=SimpleNamespace(_fixed_tendon_position_target=position_target),
    )


def test_write_skips_passive_tendons_and_preserves_other_controls():
    """Write only the rows the binding selects, leaving every other native control untouched."""
    root_view, model = _make_root_view_and_model()
    control_rows = resolve_fixed_tendon_control_rows(root_view, model)
    position_target = wp.array([[0.0, 99.0, 0.0], [0.0, 99.0, 1.25]], dtype=wp.float32, device="cpu")
    ctrl = wp.full(6, -7.0, dtype=wp.float32, device="cpu")
    control = MjcTendonControl(_make_articulation(position_target), control_rows)

    control.write_data_to_sim(SimpleNamespace(mujoco=SimpleNamespace(ctrl=ctrl)))

    # Rows 0 and 3 drive joints, not tendons; the 99.0 target on the passive tendon reaches nothing.
    np.testing.assert_allclose(ctrl.numpy(), [-7.0, 0.0, 0.0, -7.0, 0.0, 1.25])


def test_passive_tendon_is_named_once_at_construction(caplog):
    """Name an uncommandable tendon at start-up, so the per-step path needs no device readback."""
    root_view, model = _make_root_view_and_model()
    control_rows = resolve_fixed_tendon_control_rows(root_view, model)
    position_target = wp.zeros((2, 3), dtype=wp.float32, device="cpu")

    with caplog.at_level(logging.WARNING):
        MjcTendonControl(_make_articulation(position_target), control_rows)

    assert "passive" in caplog.text


def test_write_without_mujoco_ctrl_is_rejected():
    """Report the missing control array rather than failing inside the kernel launch."""
    root_view, model = _make_root_view_and_model()
    control_rows = resolve_fixed_tendon_control_rows(root_view, model)
    position_target = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    control = MjcTendonControl(_make_articulation(position_target), control_rows)

    with pytest.raises(RuntimeError, match="mujoco.ctrl"):
        control.write_data_to_sim(SimpleNamespace())


def test_row_addressing_that_disagrees_with_the_view_is_rejected():
    """Fail loudly if this module's layout arithmetic drifts from Newton's own.

    The rows would still be in range, so without this the drift binds the wrong tendons silently.
    """
    root_view, model = _make_root_view_and_model()
    # Stand in for a Newton layout convention this module no longer matches.
    root_view.get_attribute = lambda name, source: wp.array(
        np.array([[[1, 1, 1]], [[0, 0, 0]]], dtype=np.int32), dtype=wp.int32, device="cpu"
    )

    with pytest.raises(RuntimeError, match="disagrees with ArticulationView"):
        resolve_fixed_tendon_control_rows(root_view, model)


def test_model_without_mujoco_tendons_resolves_to_none():
    """Treat a model built without MuJoCo tendon attributes as nothing to resolve, not an error."""
    root_view, _ = _make_root_view_and_model()

    assert resolve_fixed_tendon_control_rows(root_view, SimpleNamespace()) is None
