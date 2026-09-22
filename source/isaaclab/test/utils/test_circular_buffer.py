# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.utils import CircularBuffer

pytestmark = pytest.mark.unit

MAX_LEN, BATCH_SIZE = 5, 3


@pytest.fixture
def circular_buffer():
    return CircularBuffer(MAX_LEN, BATCH_SIZE, "cpu")


def frame(value: float, shape: tuple[int, ...] = (BATCH_SIZE, 2)) -> torch.Tensor:
    return torch.full(shape, value, dtype=torch.float32)


def test_initialization_and_reset(circular_buffer):
    assert circular_buffer.max_length == MAX_LEN
    assert circular_buffer.batch_size == BATCH_SIZE
    assert circular_buffer.device == "cpu"
    assert circular_buffer.current_length.tolist() == [0, 0, 0]

    circular_buffer.append(frame(1.0))
    circular_buffer.reset()
    assert circular_buffer.current_length.tolist() == [0, 0, 0]


def test_reset_subset_zeroes_storage_and_restarts_warmup(circular_buffer):
    circular_buffer.append(frame(1.0))
    circular_buffer.append(frame(2.0))
    circular_buffer.reset(batch_ids=[1])
    assert circular_buffer.current_length.tolist() == [2, 0, 2]
    # the raw storage of the reset batch is zeroed while the other batches keep their history
    torch.testing.assert_close(circular_buffer.buffer[1], torch.zeros(MAX_LEN, 2))
    torch.testing.assert_close(circular_buffer.buffer[0, -1], frame(2.0)[0])

    circular_buffer.append(frame(3.0))
    assert circular_buffer.current_length.tolist() == [3, 1, 3]
    # the first append after a reset fills every slot of that batch
    torch.testing.assert_close(circular_buffer.buffer[1], frame(3.0, (MAX_LEN, 2)))


def test_getitem_returns_newest_first_and_clamps_to_oldest(circular_buffer):
    circular_buffer.append(frame(1.0))
    circular_buffer.append(frame(2.0))
    assert circular_buffer.current_length.tolist() == [2, 2, 2]
    torch.testing.assert_close(circular_buffer[torch.tensor([0, 0, 0])], frame(2.0))
    torch.testing.assert_close(circular_buffer[torch.tensor([1, 1, 1])], frame(1.0))
    # keys beyond the number of pushes return the oldest data
    torch.testing.assert_close(circular_buffer[torch.tensor([5, 5, 5])], frame(1.0))


def test_overflow_keeps_the_newest_entries(circular_buffer):
    num_overflow = 2
    for value in range(MAX_LEN + num_overflow):
        circular_buffer.append(frame(value))
    assert circular_buffer.current_length.tolist() == [MAX_LEN] * BATCH_SIZE
    torch.testing.assert_close(circular_buffer[torch.tensor([0, 0, 0])], frame(MAX_LEN + num_overflow - 1))
    torch.testing.assert_close(circular_buffer[torch.tensor([MAX_LEN - 1] * BATCH_SIZE)], frame(num_overflow))

    # .buffer is (batch, max_len, ...) ordered oldest to newest
    buffer = circular_buffer.buffer
    assert buffer.shape == (BATCH_SIZE, MAX_LEN, 2)
    expected = torch.arange(num_overflow, MAX_LEN + num_overflow, dtype=torch.float32)
    torch.testing.assert_close(buffer[:, :, 0], expected.expand(BATCH_SIZE, MAX_LEN))


def test_invalid_access(circular_buffer):
    with pytest.raises(RuntimeError):
        circular_buffer[torch.tensor([0, 0, 0])]  # empty buffer
    with pytest.raises(ValueError):
        circular_buffer.append(frame(1.0, (BATCH_SIZE + 1, 2)))
    circular_buffer.append(frame(1.0))
    with pytest.raises(ValueError):
        circular_buffer[torch.tensor([0, 0])]  # wrong key batch size
    with pytest.raises(RuntimeError, match="stack_dim"):
        _ = circular_buffer.stacked


def test_stack_dim_validation():
    with pytest.raises(ValueError, match="stack_dim must not be 0"):
        CircularBuffer(max_len=2, batch_size=4, device="cpu", stack_dim=0)
    buf = CircularBuffer(max_len=2, batch_size=4, device="cpu", stack_dim=-5)
    with pytest.raises(IndexError, match="stack_dim=-5"):
        buf.append(torch.zeros(4, 8, 8, 3))
    buf = CircularBuffer(max_len=2, batch_size=4, device="cpu", stack_dim=-1)
    buf.append(torch.zeros(4, 8, 8, 3))
    with pytest.raises(NotImplementedError, match="stacked-output mode"):
        _ = buf[torch.zeros(4, dtype=torch.long)]


@pytest.mark.parametrize(
    ("stack_dim", "stacked_shape"), [(-1, (2, 4, 5, 3 * 3)), (-3, (2, 3 * 4, 5, 3)), (3, (2, 4, 5, 3 * 3))]
)
def test_stack_dim_output_shapes(stack_dim, stacked_shape):
    buf = CircularBuffer(max_len=3, batch_size=2, device="cpu", stack_dim=stack_dim)
    buf.append(torch.zeros(2, 4, 5, 3))
    assert buf.stacked.shape == stacked_shape
    # .buffer still honors the (batch, max_len, *frame_shape) contract
    assert buf.buffer.shape == (2, 3, 4, 5, 3)


def test_stack_dim_channel_order_warmup_and_reset():
    shape = (2, 4, 4, 3)
    buf = CircularBuffer(max_len=2, batch_size=2, device="cpu", stack_dim=-1)

    # the first frame fills every slot
    buf.append(frame(7.0, shape))
    torch.testing.assert_close(buf.stacked, frame(7.0, (2, 4, 4, 6)))
    # slots are ordered oldest to newest and the oldest is evicted on overflow
    buf.append(frame(1.0, shape))
    buf.append(frame(2.0, shape))
    torch.testing.assert_close(buf.stacked[..., :3], frame(1.0, shape))
    torch.testing.assert_close(buf.stacked[..., 3:], frame(2.0, shape))

    # a partial reset zeroes the storage of the reset batch only
    buf.reset(batch_ids=[1])
    torch.testing.assert_close(buf._buffer[1], torch.zeros(4, 4, 2, 3))
    torch.testing.assert_close(buf.stacked[0, ..., :3], frame(1.0, shape)[0])
    # a full reset restarts the warmup
    buf.reset()
    buf.append(frame(9.0, shape))
    torch.testing.assert_close(buf.stacked, frame(9.0, (2, 4, 4, 6)))
