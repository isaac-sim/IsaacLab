# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tensor-state ownership and row selection at the fixed storage boundary."""

import pytest
import torch

from isaaclab.envs.reset import ResetStateBuffer

pytestmark = [pytest.mark.unit, pytest.mark.kitless, pytest.mark.windows_ci]


def test_buffer_owns_snapshots_and_returns_selected_copies() -> None:
    poses = torch.arange(12, dtype=torch.float64).reshape(3, 4).requires_grad_()
    states = {"rigid_object": {"cube": {"root_pose": poses}}, "joint_ids": torch.tensor([7, 8, 9])}
    buffer = ResetStateBuffer(states)
    assert buffer.num_states == 3
    row_ids = torch.tensor([2, 0, 2])
    selected = buffer.get_state(row_ids)
    expected = torch.tensor([[8, 9, 10, 11], [0, 1, 2, 3], [8, 9, 10, 11]], dtype=torch.float64)
    torch.testing.assert_close(selected["rigid_object"]["cube"]["root_pose"], expected)
    assert not selected["rigid_object"]["cube"]["root_pose"].requires_grad
    torch.testing.assert_close(selected["joint_ids"], torch.tensor([9, 7, 9]))
    with torch.no_grad():
        poses.zero_()
    states.clear()
    selected["rigid_object"]["cube"]["root_pose"].zero_()
    exported = buffer.states
    exported["rigid_object"]["cube"]["root_pose"].resize_(0)
    exported["joint_ids"].zero_()
    exported.clear()
    assert buffer.num_states == 3
    torch.testing.assert_close(buffer.get_state(row_ids)["rigid_object"]["cube"]["root_pose"], expected)
    torch.testing.assert_close(buffer.states["joint_ids"], torch.tensor([7, 8, 9]))
    empty = buffer.get_state(torch.empty(0, dtype=torch.long))
    assert empty["rigid_object"]["cube"]["root_pose"].shape == (0, 4)


@pytest.mark.parametrize(
    ("states", "message"),
    [
        ({"empty": {}}, "non-empty tree"),
        ({"pose": torch.tensor(1.0)}, "leading row dimension"),
        ({"pose": "invalid"}, "leading row dimension"),
        ({"pose": torch.zeros(3, 7), "joint": {"position": torch.zeros(2, 4)}}, "share a leading row dimension"),
        ({"pose": torch.zeros(2, 7), "joint": torch.empty(2, 4, device="meta")}, "and device"),
    ],
)
def test_buffer_rejects_invalid_tensor_trees(states, message) -> None:
    with pytest.raises(ValueError, match=message):
        ResetStateBuffer(states)
