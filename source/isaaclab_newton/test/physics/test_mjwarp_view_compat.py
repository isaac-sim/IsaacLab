# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics._mjwarp_view_compat import ensure_newton_custom_frequency_api

pytestmark = pytest.mark.unit


def _model(hands_per_world=2, worlds=2, actuators_per_hand=2, tendons_per_hand=1):
    """Build a model shaped the way Newton reports one.

    Ownership comes from the per-row world assignments, not the labels: MEASURED, a 2-clone
    Shadow Hand scene reports ``articulation_label`` as the template path for every clone.
    """
    total_hands = hands_per_world * worlds
    actuator_world, tendon_world = [], []
    for world in range(worlds):
        actuator_world += [world] * (hands_per_world * actuators_per_hand)
        tendon_world += [world] * (hands_per_world * tendons_per_hand)
    return SimpleNamespace(
        articulation_count=total_hands,
        # Template path, identical for every clone -- deliberately useless for ownership.
        articulation_label=["/World/Env_0/Robot"] * total_hands,
        custom_frequency_counts={
            "mujoco:actuator": len(actuator_world),
            "mujoco:tendon": len(tendon_world),
        },
        mujoco=SimpleNamespace(
            actuator_world=np.array(actuator_world),
            tendon_world=np.array(tendon_world),
            actuator_target_label=[f"/World/Env_0/Robot/joint_{i}" for i in range(len(actuator_world))],
            ctrl=wp.zeros(len(actuator_world), dtype=wp.float32, device="cpu"),
        ),
    )


def _view(articulation_ids, *, world_count=2, count_per_world=1, tendon_offset=0, tendon_count=1):
    return SimpleNamespace(
        device=wp.get_device("cpu"),
        world_count=world_count,
        count_per_world=count_per_world,
        count=world_count * count_per_world,
        articulation_ids=np.array(articulation_ids).reshape(world_count, count_per_world),
        tendon_count=tendon_count,
        tendon_names=[f"t{i}" for i in range(tendon_count)],
        # The offset is this articulation's first tendon row, which is what places it among the
        # articulations sharing its world.
        frequency_layouts={"mujoco:tendon": SimpleNamespace(offset=tendon_offset)},
    )


def _right_hand(model=None):
    """The first hand in each world: tendon offset 0."""
    return ensure_newton_custom_frequency_api(_view([0, 2], tendon_offset=0), model or _model())[0]


def test_a_view_that_already_has_the_api_is_returned_unchanged():
    """On Newton 1.6 the wrapper must not interpose at all."""
    view = _view([0, 2])
    view.custom_frequency_counts = {"mujoco:actuator": 2}

    assert ensure_newton_custom_frequency_api(view, _model())[0] is view


def test_counts_come_from_the_world_partition_not_the_totals():
    """8 actuators over 4 articulations happens to divide, but the world partition is what rules."""
    model = _model()

    assert model.custom_frequency_counts["mujoco:actuator"] == 8
    assert _right_hand(model).custom_frequency_counts["mujoco:actuator"] == 2


def test_cloned_articulations_take_their_own_world_block():
    """The CI failure: one articulation cloned across worlds, whose labels are all identical."""
    model = _model(hands_per_world=1, worlds=2, actuators_per_hand=20, tendons_per_hand=4)
    adapted = ensure_newton_custom_frequency_api(_view([0, 1], tendon_offset=0, tendon_count=4), model)[0]

    assert adapted.custom_frequency_counts["mujoco:actuator"] == 20
    rows = adapted._rows_per_instance
    # World 0 owns rows 0-19, world 1 owns 20-39; the clones must not share.
    assert rows[0].tolist() == list(range(20))
    assert rows[1].tolist() == list(range(20, 40))


def test_unrelated_attributes_are_delegated_to_the_wrapped_view():
    view = _view([0, 2])
    view.some_newton_attribute = "delegated"
    adapted = ensure_newton_custom_frequency_api(view, _model())[0]

    assert adapted.some_newton_attribute == "delegated"
    assert adapted.world_count == 2


def test_two_articulations_in_one_world_do_not_claim_each_others_actuators():
    """The handover scene puts two hands in each world, and each view must take only its own."""
    model = _model()
    right = ensure_newton_custom_frequency_api(_view([0, 2], tendon_offset=0), model)[0]
    left = ensure_newton_custom_frequency_api(_view([1, 3], tendon_offset=1), model)[0]

    right.set_attribute(
        "mujoco.ctrl", model, wp.array(np.array([[[1.0, 2.0]], [[5.0, 6.0]]], dtype=np.float32), device="cpu")
    )
    left.set_attribute(
        "mujoco.ctrl", model, wp.array(np.array([[[3.0, 4.0]], [[7.0, 8.0]]], dtype=np.float32), device="cpu")
    )

    # Right owns rows 0-1 and 4-5, left owns 2-3 and 6-7; neither overwrote the other.
    np.testing.assert_array_equal(model.mujoco.ctrl.numpy(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


def test_actuator_rows_are_read_back_in_view_shape():
    model = _model()
    model.mujoco.actuator_trntype = np.arange(8, dtype=np.int32)

    got = _right_hand(model).get_attribute("mujoco.actuator_trntype", model).numpy()

    assert got.shape == (2, 1, 2)
    # The right hand owns rows 0-1 in world 0 and 4-5 in world 1.
    np.testing.assert_array_equal(got[:, 0, :], [[0, 1], [4, 5]])


def test_a_mismatched_command_shape_is_refused():
    """Silently broadcasting the wrong shape would drive arbitrary actuators."""
    model = _model()

    with pytest.raises(ValueError, match="Expected values shaped"):
        _right_hand(model).set_attribute("mujoco.ctrl", model, wp.zeros((2, 1, 3), dtype=wp.float32, device="cpu"))
