# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

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


# ---------------------------------------------------------------------------
# Stacked-output mode (stack_dim) tests
# ---------------------------------------------------------------------------


def test_reset_subset_zeroes_buffer_storage_default_mode():
    """reset(batch_ids=[i]) must zero the buffer slots for the reset rows (default mode).

    Regression: a prior implementation used ``self._buffer[:, ids].zero_()`` which is
    getitem-then-inplace; advanced indexing with a list/tensor returns a copy, so
    ``.zero_()`` zeroed the temporary and left the original buffer untouched.
    """
    buf = CircularBuffer(max_len=3, batch_size=4, device="cpu")
    buf.append(torch.full((4, 2), 5.0))
    buf.append(torch.full((4, 2), 5.0))
    buf.reset(batch_ids=[1, 3])
    # Reset rows must read as zero in the raw buffer; non-reset rows must still hold 5.0.
    torch.testing.assert_close(buf._buffer[:, [1, 3]], torch.zeros((3, 2, 2)))
    torch.testing.assert_close(buf._buffer[:, [0, 2]], torch.full((3, 2, 2), 5.0))


def test_reset_subset_zeroes_buffer_storage_stack_dim_mode():
    """reset(batch_ids=[i]) must zero the buffer slots for the reset rows (stack_dim mode).

    Same regression as :func:`test_reset_subset_zeroes_buffer_storage_default_mode` but for
    the stack_dim layout where the batch dim is dim 0 of the internal storage.
    """
    buf = CircularBuffer(max_len=2, batch_size=4, device="cpu", stack_dim=-1)
    buf.append(torch.full((4, 8, 8, 3), 5.0))
    buf.append(torch.full((4, 8, 8, 3), 5.0))
    buf.reset(batch_ids=[1, 3])
    torch.testing.assert_close(buf._buffer[[1, 3]], torch.zeros((2, 8, 8, 2, 3)))
    torch.testing.assert_close(buf._buffer[[0, 2]], torch.full((2, 8, 8, 2, 3), 5.0))


def test_stack_dim_zero_rejected():
    """stack_dim=0 (batch dim) must be rejected at construction."""
    with pytest.raises(ValueError, match="stack_dim must not be 0"):
        CircularBuffer(max_len=2, batch_size=4, device="cpu", stack_dim=0)


def test_stack_dim_out_of_range_rejected_on_first_append():
    """Invalid stack_dim for the appended data's rank raises on first append."""
    buf = CircularBuffer(max_len=2, batch_size=4, device="cpu", stack_dim=-5)
    data = torch.zeros(4, 8, 8, 3)  # ndim=4, valid stack_dim range is [-3,-1] or [1,3]
    with pytest.raises(IndexError, match="stack_dim=-5"):
        buf.append(data)


def test_stack_dim_minus_one_output_shape():
    """stack_dim=-1 on (B,H,W,C) data yields .stacked shape (B,H,W,K*C)."""
    B, H, W, C, K = 4, 8, 8, 3, 2
    buf = CircularBuffer(max_len=K, batch_size=B, device="cpu", stack_dim=-1)
    data = torch.zeros(B, H, W, C)
    buf.append(data)
    assert buf.stacked.shape == (B, H, W, K * C)
    # And .buffer still honors the legacy (B, K, *frame_shape) contract.
    assert buf.buffer.shape == (B, K, H, W, C)


def test_stack_dim_oldest_to_newest_channel_order():
    """Channels must appear in oldest-to-newest order along the stacked dim."""
    B, H, W, C, K = 2, 4, 4, 3, 2
    buf = CircularBuffer(max_len=K, batch_size=B, device="cpu", stack_dim=-1)
    # First frame: all 1s
    f1 = torch.ones(B, H, W, C)
    buf.append(f1)
    # Second frame: all 2s
    f2 = torch.full((B, H, W, C), 2.0)
    buf.append(f2)
    stacked = buf.stacked  # (B, H, W, 2*C)
    # Oldest C channels should equal f1 (i.e., 1.0); newest C channels should equal f2 (i.e., 2.0).
    torch.testing.assert_close(stacked[..., :C], torch.ones(B, H, W, C))
    torch.testing.assert_close(stacked[..., C:], torch.full((B, H, W, C), 2.0))


def test_stack_dim_warmup_fills_all_slots_with_first_frame():
    """The first append must fill all K slots with the first frame (warmup contract)."""
    B, H, W, C, K = 2, 4, 4, 3, 2
    buf = CircularBuffer(max_len=K, batch_size=B, device="cpu", stack_dim=-1)
    f1 = torch.full((B, H, W, C), 7.0)
    buf.append(f1)
    stacked = buf.stacked
    # Both K slots should be 7.0 after the warmup.
    torch.testing.assert_close(stacked, torch.full((B, H, W, K * C), 7.0))


def test_stack_dim_minus_three_output_shape():
    """stack_dim=-3 (height-stack) on (B,H,W,C) yields .stacked shape (B,K*H,W,C)."""
    B, H, W, C, K = 2, 4, 5, 3, 3
    buf = CircularBuffer(max_len=K, batch_size=B, device="cpu", stack_dim=-3)
    data = torch.zeros(B, H, W, C)
    buf.append(data)
    assert buf.stacked.shape == (B, K * H, W, C)


def test_stack_dim_positive_index_equivalent_to_negative():
    """stack_dim=3 should behave identically to stack_dim=-1 for 4D data."""
    B, H, W, C, K = 2, 4, 4, 3, 2
    buf_neg = CircularBuffer(max_len=K, batch_size=B, device="cpu", stack_dim=-1)
    buf_pos = CircularBuffer(max_len=K, batch_size=B, device="cpu", stack_dim=3)
    f1 = torch.randn(B, H, W, C)
    f2 = torch.randn(B, H, W, C)
    buf_neg.append(f1)
    buf_neg.append(f2)
    buf_pos.append(f1)
    buf_pos.append(f2)
    torch.testing.assert_close(buf_neg.stacked, buf_pos.stacked)


def test_stack_dim_ring_shift_after_overflow():
    """After K+1 frames, the oldest slot must be frame 1 (frame 0 evicted), newest = last."""
    B, H, W, C, K = 2, 4, 4, 3, 2
    buf = CircularBuffer(max_len=K, batch_size=B, device="cpu", stack_dim=-1)
    f0 = torch.full((B, H, W, C), 0.0)
    f1 = torch.full((B, H, W, C), 1.0)
    f2 = torch.full((B, H, W, C), 2.0)
    buf.append(f0)
    buf.append(f1)
    buf.append(f2)
    stacked = buf.stacked  # K=2, so slots are [f1, f2]
    torch.testing.assert_close(stacked[..., :C], torch.full((B, H, W, C), 1.0))
    torch.testing.assert_close(stacked[..., C:], torch.full((B, H, W, C), 2.0))


def test_stack_dim_reset_clears_buffer():
    """reset() should re-trigger warmup behavior on the next append."""
    B, H, W, C, K = 2, 4, 4, 3, 2
    buf = CircularBuffer(max_len=K, batch_size=B, device="cpu", stack_dim=-1)
    buf.append(torch.full((B, H, W, C), 1.0))
    buf.append(torch.full((B, H, W, C), 2.0))
    buf.reset()
    # Next append should be a warmup fill: both K slots equal the new frame.
    buf.append(torch.full((B, H, W, C), 9.0))
    torch.testing.assert_close(buf.stacked, torch.full((B, H, W, K * C), 9.0))


def test_stack_dim_getitem_raises():
    """__getitem__ is not supported in stack_dim mode."""
    buf = CircularBuffer(max_len=2, batch_size=4, device="cpu", stack_dim=-1)
    buf.append(torch.zeros(4, 8, 8, 3))
    with pytest.raises(NotImplementedError, match="stacked-output mode"):
        _ = buf[torch.zeros(4, dtype=torch.long)]


def test_stack_dim_stacked_raises_when_default_mode():
    """.stacked must raise in default mode (legacy CircularBuffer use)."""
    buf = CircularBuffer(max_len=2, batch_size=4, device="cpu")  # no stack_dim
    buf.append(torch.zeros(4, 3))
    with pytest.raises(RuntimeError, match="stack_dim"):
        _ = buf.stacked
