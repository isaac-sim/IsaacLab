# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused unit tests for :class:`isaaclab.utils.wrench_composer.WrenchComposer`."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.wrench_composer import WrenchComposer

pytestmark = pytest.mark.unit


class _AssetData:
    """Minimal asset data consumed by :class:`WrenchComposer`."""

    def __init__(self, com_pos_w: torch.Tensor, link_quat_w: torch.Tensor) -> None:
        self.body_com_pos_w = ProxyArray(wp.from_torch(com_pos_w, dtype=wp.vec3f))
        self.body_link_quat_w = ProxyArray(wp.from_torch(link_quat_w, dtype=wp.quatf))


def _make_composer(*, com_pos_w: torch.Tensor | None = None, link_quat_w: torch.Tensor | None = None) -> WrenchComposer:
    """Create a two-environment, two-body composer with literal fixture data."""
    if com_pos_w is None:
        com_pos_w = torch.zeros((2, 2, 3), dtype=torch.float32)
    if link_quat_w is None:
        link_quat_w = torch.tensor(
            [
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
    asset = SimpleNamespace(
        num_instances=2,
        num_bodies=2,
        device="cpu",
        data=_AssetData(com_pos_w, link_quat_w),
    )
    return WrenchComposer(asset)


def _vectors(values: list[list[list[float]]]) -> wp.array:
    """Convert a literal tensor-shaped list to a Warp vector array."""
    return wp.from_torch(torch.tensor(values, dtype=torch.float32).contiguous(), dtype=wp.vec3f)


def _mask(values: list[bool]) -> wp.array:
    """Convert literal booleans to a Warp mask array with owned tensor storage."""
    return wp.from_torch(torch.tensor(values, dtype=torch.bool).contiguous(), dtype=wp.bool)


def test_local_force_and_torque_at_position_compose_to_literal_wrench() -> None:
    composer = _make_composer()

    composer.add_forces_and_torques_index(
        forces=_vectors([[[2.0, 0.0, 0.0]]]),
        torques=_vectors([[[0.0, 0.0, 5.0]]]),
        positions=_vectors([[[0.0, 3.0, 0.0]]]),
        env_ids=torch.tensor([1]),
        body_ids=[0],
    )

    expected_force = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    expected_torque = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]])

    torch.testing.assert_close(composer.out_force_b.torch, expected_force)
    torch.testing.assert_close(composer.out_torque_b.torch, expected_torque)


def test_add_accumulates_local_wrenches() -> None:
    composer = _make_composer()

    composer.add_forces_and_torques_index(
        forces=_vectors([[[1.0, 2.0, 3.0]]]), torques=_vectors([[[0.0, 1.0, 0.0]]]), body_ids=[1], env_ids=[0]
    )
    composer.add_forces_and_torques_index(
        forces=_vectors([[[4.0, -1.0, 0.0]]]), torques=_vectors([[[2.0, 0.0, -3.0]]]), body_ids=[1], env_ids=[0]
    )

    expected_force = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 1.0, 3.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    expected_torque = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 1.0, -3.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

    torch.testing.assert_close(composer.out_force_b.torch, expected_force)
    torch.testing.assert_close(composer.out_torque_b.torch, expected_torque)


def test_global_force_at_position_rotates_force_and_induced_torque() -> None:
    quarter_turn_z = 2.0**-0.5
    composer = _make_composer(
        com_pos_w=torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        link_quat_w=torch.tensor(
            [
                [[0.0, 0.0, quarter_turn_z, quarter_turn_z], [0.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            ]
        ),
    )

    composer.add_forces_and_torques_index(
        forces=_vectors([[[2.0, 0.0, 0.0]]]),
        positions=_vectors([[[1.0, 4.0, 3.0]]]),
        body_ids=[0],
        env_ids=[0],
        is_global=True,
    )

    expected_force = torch.tensor([[[0.0, -2.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    expected_torque = torch.tensor([[[0.0, 0.0, -4.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])

    torch.testing.assert_close(composer.out_force_b.torch, expected_force, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(composer.out_torque_b.torch, expected_torque, atol=1.0e-6, rtol=1.0e-6)


def test_index_and_mask_selection_change_only_selected_cells() -> None:
    composer = _make_composer()

    composer.add_forces_and_torques_index(
        forces=_vectors([[[1.0, 0.0, 0.0]]]), body_ids=torch.tensor([1], dtype=torch.int64), env_ids=torch.tensor([0])
    )
    composer.add_forces_and_torques_mask(
        forces=_vectors(
            [
                [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
                [[30.0, 0.0, 0.0], [40.0, 0.0, 0.0]],
            ]
        ),
        env_mask=_mask([False, True]),
        body_mask=_mask([True, False]),
    )

    expected_force = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[30.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    torch.testing.assert_close(composer.out_force_b.torch, expected_force)


def test_set_clears_only_targeted_environment_before_writing() -> None:
    composer = _make_composer()
    composer.add_forces_and_torques_index(
        forces=_vectors(
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        )
    )

    composer.set_forces_and_torques_index(forces=_vectors([[[9.0, 0.0, 0.0]]]), body_ids=[1], env_ids=[0])

    expected_force = torch.tensor([[[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]], [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    torch.testing.assert_close(composer.out_force_b.torch, expected_force)


def test_mask_set_clears_only_masked_environments_before_writing() -> None:
    composer = _make_composer()
    composer.add_forces_and_torques_index(
        forces=_vectors(
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        )
    )

    composer.set_forces_and_torques_mask(
        forces=_vectors(
            [
                [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
                [[7.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
            ]
        ),
        env_mask=_mask([False, True]),
        body_mask=_mask([False, True]),
    )

    expected_force = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]])
    torch.testing.assert_close(composer.out_force_b.torch, expected_force)


def test_partial_and_full_reset_clear_their_documented_scope() -> None:
    composer = _make_composer()
    composer.add_forces_and_torques_index(
        forces=_vectors(
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ]
        )
    )

    composer.reset(env_ids=[0])
    expected_after_partial = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    torch.testing.assert_close(composer.out_force_b.torch, expected_after_partial)
    assert composer.active

    composer.reset()
    torch.testing.assert_close(composer.out_force_b.torch, torch.zeros((2, 2, 3)))
    torch.testing.assert_close(composer.out_torque_b.torch, torch.zeros((2, 2, 3)))
    assert not composer.active


def test_permanent_and_instantaneous_composers_remain_independent() -> None:
    permanent = _make_composer()
    instantaneous = _make_composer()
    permanent.add_forces_and_torques_index(forces=_vectors([[[5.0, 0.0, 0.0]]]), body_ids=[0], env_ids=[0])
    instantaneous.add_forces_and_torques_index(forces=_vectors([[[0.0, 7.0, 0.0]]]), body_ids=[0], env_ids=[0])

    instantaneous.reset()

    expected_permanent = torch.tensor([[[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    torch.testing.assert_close(permanent.out_force_b.torch, expected_permanent)
    torch.testing.assert_close(instantaneous.out_force_b.torch, torch.zeros((2, 2, 3)))


def test_raw_buffer_merge_accumulates_and_ignores_inactive_source() -> None:
    destination = _make_composer()
    source = _make_composer()
    inactive_source = _make_composer()
    destination.add_forces_and_torques_index(forces=_vectors([[[1.0, 0.0, 0.0]]]), body_ids=[0], env_ids=[0])
    source.add_forces_and_torques_index(forces=_vectors([[[0.0, 2.0, 0.0]]]), body_ids=[0], env_ids=[0])

    destination.add_raw_buffers_from(source)
    destination.add_raw_buffers_from(inactive_source)

    expected_force = torch.tensor([[[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    torch.testing.assert_close(destination.out_force_b.torch, expected_force)


def test_invalid_selection_and_empty_wrench_input_are_reported() -> None:
    composer = _make_composer()

    with pytest.raises(TypeError, match="env_ids must be"):
        composer.add_forces_and_torques_index(forces=_vectors([[[1.0, 0.0, 0.0]]]), env_ids=(0,))
    with pytest.warns(UserWarning, match="No forces or torques"):
        composer.add_forces_and_torques_index()
    assert not composer.active


def test_deprecated_wrappers_warn_and_preserve_wrench_behavior() -> None:
    composer = _make_composer()

    with pytest.warns(DeprecationWarning, match="add_forces_and_torques.*deprecated"):
        composer.add_forces_and_torques(
            forces=_vectors([[[3.0, 0.0, 0.0]]]),
            torques=_vectors([[[0.0, 0.0, 2.0]]]),
            body_ids=[1],
            env_ids=[1],
        )
    with pytest.warns(DeprecationWarning, match="composed_force.*deprecated"):
        force = composer.composed_force.torch
    with pytest.warns(DeprecationWarning, match="composed_torque.*deprecated"):
        torque = composer.composed_torque.torch

    expected_force = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    expected_torque = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]])
    torch.testing.assert_close(force, expected_force)
    torch.testing.assert_close(torque, expected_torque)

    with pytest.warns(DeprecationWarning, match="set_forces_and_torques.*deprecated"):
        composer.set_forces_and_torques(forces=_vectors([[[4.0, 0.0, 0.0]]]), body_ids=[1], env_ids=[1])

    expected_force[1, 1] = torch.tensor([4.0, 0.0, 0.0])
    torch.testing.assert_close(composer.out_force_b.torch, expected_force)
    torch.testing.assert_close(composer.out_torque_b.torch, torch.zeros((2, 2, 3)))
