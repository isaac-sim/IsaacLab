# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows from here."""

from isaaclab.utils import CircularBuffer


@pytest.fixture
def circular_buffer():
    """Create a circular buffer for testing."""
    max_len = 5
    batch_size = 3
    device = "cpu"
    return CircularBuffer(max_len, batch_size, device)


def test_initialization(circular_buffer):
    """Test initialization of the circular buffer."""
    assert circular_buffer.max_length == 5
    assert circular_buffer.batch_size == 3
    assert circular_buffer.device == "cpu"
    assert circular_buffer.current_length.tolist() == [0, 0, 0]


def test_reset(circular_buffer):
    """Test resetting the circular buffer."""
    # append some data
    data = torch.ones((circular_buffer.batch_size, 2), device=circular_buffer.device)
    circular_buffer.append(data)
    # reset the buffer
    circular_buffer.reset()

    # check if the buffer has zeros entries
    assert circular_buffer.current_length.tolist() == [0, 0, 0]


def test_reset_subset(circular_buffer):
    """Test resetting a subset of batches in the circular buffer."""
    data1 = torch.ones((circular_buffer.batch_size, 2), device=circular_buffer.device)
    data2 = 2.0 * data1.clone()
    data3 = 3.0 * data1.clone()
    circular_buffer.append(data1)
    circular_buffer.append(data2)
    # reset the buffer
    reset_batch_id = 1
    circular_buffer.reset(batch_ids=[reset_batch_id])
    # check that correct batch is reset
    assert circular_buffer.current_length.tolist()[reset_batch_id] == 0
    # Append new set of data
    circular_buffer.append(data3)
    # check if the correct number of entries are in each batch
    expected_length = [3, 3, 3]
    expected_length[reset_batch_id] = 1
    assert circular_buffer.current_length.tolist() == expected_length
    # check that all entries of the recently reset and appended batch are equal
    for i in range(circular_buffer.max_length):
        torch.testing.assert_close(circular_buffer.buffer[reset_batch_id, 0], circular_buffer.buffer[reset_batch_id, i])


def test_append_and_retrieve(circular_buffer):
    """Test appending and retrieving data from the circular buffer."""
    # append some data
    data1 = torch.tensor([[1, 1], [1, 1], [1, 1]], device=circular_buffer.device)
    data2 = torch.tensor([[2, 2], [2, 2], [2, 2]], device=circular_buffer.device)

    circular_buffer.append(data1)
    circular_buffer.append(data2)

    assert circular_buffer.current_length.tolist() == [2, 2, 2]

    retrieved_data = circular_buffer[torch.tensor([0, 0, 0], device=circular_buffer.device)]
    assert torch.equal(retrieved_data, data2)

    retrieved_data = circular_buffer[torch.tensor([1, 1, 1], device=circular_buffer.device)]
    assert torch.equal(retrieved_data, data1)


def test_buffer_overflow(circular_buffer):
    """Test buffer overflow.

    If the buffer is full, the oldest data should be overwritten.
    """
    # add data in ascending order
    for count in range(circular_buffer.max_length + 2):
        data = torch.full((circular_buffer.batch_size, 4), count, device=circular_buffer.device)
        circular_buffer.append(data)

    # check buffer length is correct
    assert circular_buffer.current_length.tolist() == [
        circular_buffer.max_length,
        circular_buffer.max_length,
        circular_buffer.max_length,
    ]

    # retrieve most recent data
    key = torch.tensor([0, 0, 0], device=circular_buffer.device)
    retrieved_data = circular_buffer[key]
    expected_data = torch.full_like(data, circular_buffer.max_length + 1)

    assert torch.equal(retrieved_data, expected_data)

    # retrieve the oldest data
    key = torch.tensor(
        [circular_buffer.max_length - 1, circular_buffer.max_length - 1, circular_buffer.max_length - 1],
        device=circular_buffer.device,
    )
    retrieved_data = circular_buffer[key]
    expected_data = torch.full_like(data, 2)

    assert torch.equal(retrieved_data, expected_data)


def test_empty_buffer_access(circular_buffer):
    """Test accessing an empty buffer."""
    with pytest.raises(RuntimeError):
        circular_buffer[torch.tensor([0, 0, 0], device=circular_buffer.device)]


def test_invalid_batch_size(circular_buffer):
    """Test appending data with an invalid batch size."""
    data = torch.ones((circular_buffer.batch_size + 1, 2), device=circular_buffer.device)
    with pytest.raises(ValueError):
        circular_buffer.append(data)

    with pytest.raises(ValueError):
        circular_buffer[torch.tensor([0, 0], device=circular_buffer.device)]


def test_key_greater_than_pushes(circular_buffer):
    """Test retrieving data with a key greater than the number of pushes.

    In this case, the oldest data should be returned.
    """
    data1 = torch.tensor([[1, 1], [1, 1], [1, 1]], device=circular_buffer.device)
    data2 = torch.tensor([[2, 2], [2, 2], [2, 2]], device=circular_buffer.device)

    circular_buffer.append(data1)
    circular_buffer.append(data2)

    retrieved_data = circular_buffer[torch.tensor([5, 5, 5], device=circular_buffer.device)]
    assert torch.equal(retrieved_data, data1)


def test_empty_batch_ids_reset_is_noop(circular_buffer):
    """``reset(batch_ids=[])`` must leave the buffer readable.

    Regression test for the CPU-flag path: a zero-length ``batch_ids`` must
    not flip ``_any_first_push_pending`` (nothing was actually zeroed). The
    previous ``torch.any(num_pushes == 0)`` probe was immune because it read
    the real tensor state; the flag-based guard must match that behavior to
    avoid spuriously raising on callers that pass empty done-env lists.
    """
    data = torch.ones((circular_buffer.batch_size, 2), device=circular_buffer.device)
    circular_buffer.append(data)
    circular_buffer.append(data)
    circular_buffer.reset(batch_ids=[])
    # Must not raise — all batch indices still have num_pushes > 0.
    retrieved = circular_buffer[torch.tensor([0, 0, 0], device=circular_buffer.device)]
    torch.testing.assert_close(retrieved, data)
    # Also accept a zero-length tensor (common when built from a termination mask).
    circular_buffer.reset(batch_ids=torch.tensor([], dtype=torch.long, device=circular_buffer.device))
    retrieved = circular_buffer[torch.tensor([0, 0, 0], device=circular_buffer.device)]
    torch.testing.assert_close(retrieved, data)


def test_partial_reset_then_read_raises(circular_buffer):
    """``__getitem__`` must still raise after a partial reset without a follow-up append.

    Guards the CPU-side ``_any_first_push_pending`` flag that replaces the
    previous per-call ``torch.any(num_pushes == 0)`` probe: the flag must be
    set by :meth:`reset` and only cleared once :meth:`append` replicates the
    first push, otherwise a reader could silently observe uninitialised slots.
    """
    data = torch.ones((circular_buffer.batch_size, 2), device=circular_buffer.device)
    circular_buffer.append(data)
    circular_buffer.append(data)
    circular_buffer.reset(batch_ids=[0])
    with pytest.raises(RuntimeError):
        circular_buffer[torch.tensor([0, 0, 0], device=circular_buffer.device)]


def test_interleaved_partial_reset_and_append(circular_buffer):
    """Cycle several partial resets + appends and verify each reset env's
    history is fully replicated with its first post-reset sample (first-push
    invariant) while untouched envs keep their real history."""
    d1 = torch.tensor([[1, 1], [2, 2], [3, 3]], device=circular_buffer.device)
    d2 = 10 * d1
    d3 = 100 * d1

    circular_buffer.append(d1)
    circular_buffer.append(d2)

    # Partial reset env 0 then append d3. Env 0's entire history should now
    # be d3[0]; envs 1, 2 should have history including d1, d2, d3.
    circular_buffer.reset(batch_ids=[0])
    circular_buffer.append(d3)
    for i in range(circular_buffer.max_length):
        torch.testing.assert_close(circular_buffer.buffer[0, 0], circular_buffer.buffer[0, i])
    torch.testing.assert_close(circular_buffer.buffer[1, -1], d3[1])
    torch.testing.assert_close(circular_buffer.buffer[1, -2], d2[1])

    # Now partial reset env 1 and append d1 again. Env 1's history should
    # now be d1[1] everywhere; env 0 still keeps its d3 history (plus the new
    # d1 append at the head), env 2 shifts normally.
    circular_buffer.reset(batch_ids=[1])
    circular_buffer.append(d1)
    for i in range(circular_buffer.max_length):
        torch.testing.assert_close(circular_buffer.buffer[1, 0], circular_buffer.buffer[1, i])
    torch.testing.assert_close(circular_buffer.buffer[1, -1], d1[1])
    torch.testing.assert_close(circular_buffer.buffer[0, -1], d1[0])
    torch.testing.assert_close(circular_buffer.buffer[2, -1], d1[2])


def test_return_buffer_prop(circular_buffer):
    """Test retrieving the whole buffer for correct size and contents.
    Returning the whole buffer should have the shape [batch_size,max_len,data.shape[1:]]
    """
    num_overflow = 2
    for i in range(circular_buffer.max_length + num_overflow):
        data = torch.tensor([[i]], device=circular_buffer.device).repeat(3, 2)
        circular_buffer.append(data)

    retrieved_buffer = circular_buffer.buffer
    # check shape
    assert retrieved_buffer.shape == torch.Size([circular_buffer.batch_size, circular_buffer.max_length, 2])
    # check that batch is first dimension
    torch.testing.assert_close(retrieved_buffer[0], retrieved_buffer[1])
    # check oldest
    torch.testing.assert_close(
        retrieved_buffer[:, 0], torch.tensor([[num_overflow]], device=circular_buffer.device).repeat(3, 2)
    )
    # check most recent
    torch.testing.assert_close(
        retrieved_buffer[:, -1],
        torch.tensor([[circular_buffer.max_length + num_overflow - 1]], device=circular_buffer.device).repeat(3, 2),
    )
    # check that it is returned oldest first
    for idx in range(circular_buffer.max_length - 1):
        assert torch.all(torch.le(retrieved_buffer[:, idx], retrieved_buffer[:, idx + 1]))
