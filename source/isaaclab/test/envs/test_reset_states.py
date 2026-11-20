# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Snapshot ownership, default-state construction, and captured masked restoration."""

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.envs.reset import ResetStateBuffer
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.warp import ProxyArray

pytestmark = [pytest.mark.unit, pytest.mark.kitless, pytest.mark.windows_ci]


def test_buffer_owns_snapshots_and_returns_selected_copies() -> None:
    poses = torch.arange(12, dtype=torch.float64).reshape(3, 4).requires_grad_()
    history = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    states = {
        "rigid_object": {"cube": {"root_pose": poses}},
        "joint_ids": torch.tensor([7, 8, 9]),
        "history": history,
        "batched_history": history[:, None],
    }
    buffer = ResetStateBuffer(states)
    assert buffer.num_states == 3
    row_ids = torch.tensor([2, 0, 2])
    selected = buffer.get_state(row_ids)
    expected = torch.tensor([[8, 9, 10, 11], [0, 1, 2, 3], [8, 9, 10, 11]], dtype=torch.float64)
    torch.testing.assert_close(selected["rigid_object"]["cube"]["root_pose"].torch, expected)
    assert not selected["rigid_object"]["cube"]["root_pose"].torch.requires_grad
    torch.testing.assert_close(selected["joint_ids"].torch, torch.tensor([9, 7, 9]))
    expected_history = torch.tensor([[[9, 10], [11, 12]], [[1, 2], [3, 4]], [[9, 10], [11, 12]]])
    torch.testing.assert_close(selected["history"].torch, expected_history)
    torch.testing.assert_close(selected["batched_history"].torch, expected_history[:, None])
    with torch.no_grad():
        poses.zero_()
    states.clear()
    selected["rigid_object"]["cube"]["root_pose"].warp.zero_()
    exported = buffer.states
    exported["rigid_object"]["cube"]["root_pose"].warp.zero_()
    exported["joint_ids"].warp.zero_()
    exported.clear()
    assert buffer.num_states == 3
    torch.testing.assert_close(buffer.get_state(row_ids)["rigid_object"]["cube"]["root_pose"].torch, expected)
    torch.testing.assert_close(buffer.states["joint_ids"].torch, torch.tensor([7, 8, 9]))
    empty = buffer.get_state(torch.empty(0, dtype=torch.long))
    assert empty["rigid_object"]["cube"]["root_pose"].shape == (0, 4)
    selected = buffer.get_state(torch.tensor([2**32, 0, 2 - 2**32]))
    torch.testing.assert_close(selected["joint_ids"].torch, torch.tensor([0, 7, 0]))


def test_empty_buffers_and_repeated_default_states() -> None:
    empty = ResetStateBuffer({})
    assert empty.num_states == 0
    assert empty.get_state(wp.empty(0, dtype=wp.int32, device="cpu")) == {}
    zero_rows = ResetStateBuffer({"joint_pos": torch.empty(0, 2)})
    assert zero_rows.num_states == 0
    assert zero_rows.states["joint_pos"].shape == (0, 2)
    default = ProxyArray(wp.array([[0.25, -0.5]], dtype=wp.float32, device="cpu"))
    buffer = ResetStateBuffer({"robot": {"joint_pos": default}}, num_states=3)
    default.warp.zero_()
    assert buffer.num_states == 3
    np.testing.assert_array_equal(buffer.states["robot"]["joint_pos"].warp.numpy(), [[0.25, -0.5]] * 3)
    with pytest.raises(ValueError, match="empty buffer"):
        empty.get_state(torch.tensor([0]))
    with pytest.raises(ValueError, match="single default state"):
        ResetStateBuffer({"pose": torch.zeros(2, 7)}, num_states=3)


@pytest.mark.parametrize(
    ("states", "message"),
    [
        ({"pose": torch.tensor(1.0)}, "leading row dimension"),
        ({"pose": "invalid"}, "leading row dimension"),
        ({"pose": torch.zeros(3, 7), "joint": {"position": torch.zeros(2, 4)}}, "share a leading row dimension"),
    ],
)
def test_buffer_rejects_invalid_array_trees(states, message) -> None:
    with pytest.raises(ValueError, match=message):
        ResetStateBuffer(states)


def test_buffer_rejects_malformed_output_before_writing() -> None:
    buffer = ResetStateBuffer({"first": torch.tensor([1, 2]), "nested": {"second": torch.tensor([3, 4])}})
    out = {"first": torch.tensor([-1]), "nested": None}
    with pytest.raises(ValueError, match="out must match the stored state tree"):
        buffer.get_state(torch.tensor([0]), out=out)
    torch.testing.assert_close(out["first"], torch.tensor([-1]))
    assert out["nested"] is None


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_masked_state_gather_replays_changed_indices_without_touching_other_worlds(device) -> None:
    poses = wp.array(
        [[1, 2, 3, 0, 0, 0, 1], [4, 5, 6, 0, 0, 1, 0], [7, 8, 9, 0, 1, 0, 0]],
        dtype=wp.transformf,
        device=device,
    )
    joints = torch.tensor([[10.0, 99.0, 11.0, 99.0], [20.0, 99.0, 21.0, 99.0], [30.0, 99.0, 31.0, 99.0]])
    with pytest.raises(ValueError, match="and device"):
        ResetStateBuffer({"pose": poses, "joint": joints})
    joints = joints.to(device)[:, ::2]
    buffer = ResetStateBuffer({"pose": poses, "joint": joints})
    ids = wp.array([2, 0, 1], dtype=wp.int64, device=device)
    mask = wp.array([True, False, True], dtype=wp.bool, device=device)
    pose_out = wp.full(3, wp.transformf(-7.0), device=device)
    padded = wp.full((3, 4), -7.0, device=device)
    out = {"pose": ProxyArray(pose_out), "joint": padded[:, ::2]}
    buffer.get_state(ids, out=out, mask=mask)
    with wp.ScopedDevice("cpu"), wp.ScopedCapture(device=device) as capture:
        assert buffer.get_state(ids, out=out, mask=mask) is out
    ids.assign(np.array([0, 2, 1], dtype=np.int64))
    mask.assign(np.array([False, True, False]))
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(
        pose_out.numpy(), [[7, 8, 9, 0, 1, 0, 0], [7, 8, 9, 0, 1, 0, 0], [4, 5, 6, 0, 0, 1, 0]]
    )
    np.testing.assert_array_equal(padded.numpy(), [[30, -7, 31, -7], [30, -7, 31, -7], [20, -7, 21, -7]])
    mask.fill_(False)
    ids.fill_(-1)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(padded.numpy(), [[30, -7, 31, -7], [30, -7, 31, -7], [20, -7, 21, -7]])
    mask.fill_(True)
    ids.assign(np.array([2**32, 2**32, 2 - 2**32], dtype=np.int64))
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(padded.numpy(), [[30, -7, 31, -7], [30, -7, 31, -7], [20, -7, 21, -7]])
