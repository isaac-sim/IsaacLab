# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`isaaclab.envs.utils.frame_stack`. Pure tensor logic; no Kit launch."""

from __future__ import annotations

import pytest
import torch

from isaaclab.envs.utils import FrameStackBuffer

pytestmark = pytest.mark.isaacsim_ci

# Shorthand shape (num_envs, H, W, C) used across tests.
NUM_ENVS = 4
HEIGHT = 8
WIDTH = 8
CHANNELS = 3
SINGLE_SHAPE = (NUM_ENVS, HEIGHT, WIDTH, CHANNELS)


def _make_frame(value: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    """Build a constant-valued (N, H, W, C) tensor on CPU."""
    return torch.full(SINGLE_SHAPE, value, dtype=dtype)


class TestFrameStackBuffer:
    """Pure-tensor tests of the ring buffer."""

    def test_output_shape_and_channels(self):
        buf = FrameStackBuffer(SINGLE_SHAPE, frame_stack=3, device="cpu")
        assert buf.output_shape == (NUM_ENVS, HEIGHT, WIDTH, CHANNELS * 3)
        assert buf.output_channels == CHANNELS * 3
        # The narrow+copy_ rebuild writes into a single pre-allocated buffer; output must stay contiguous.
        stacked = buf.update(_make_frame(1))
        assert stacked.is_contiguous()

    def test_init_fills_all_slots_on_first_update(self):
        """First update post-construction fills every history slot with the new frame."""
        buf = FrameStackBuffer(SINGLE_SHAPE, frame_stack=2, device="cpu")
        f0 = _make_frame(7)
        stacked = buf.update(f0)
        # Both slots equal F0.
        assert torch.equal(stacked[..., :CHANNELS], f0)
        assert torch.equal(stacked[..., CHANNELS:], f0)

    def test_ring_buffer_shifts_correctly(self):
        """After the second update, oldest slot = first frame; newest slot = second frame."""
        buf = FrameStackBuffer(SINGLE_SHAPE, frame_stack=2, device="cpu")
        f0 = _make_frame(10)
        f1 = _make_frame(20)
        buf.update(f0)
        stacked = buf.update(f1)
        assert torch.equal(stacked[..., :CHANNELS], f0), "Oldest slot must be the previous frame"
        assert torch.equal(stacked[..., CHANNELS:], f1), "Newest slot must be the latest frame"

    def test_newest_slot_equals_latest_single(self):
        """Ring-buffer correctness invariant: newest slot post-update == the latest single input."""
        buf = FrameStackBuffer(SINGLE_SHAPE, frame_stack=2, device="cpu")
        buf.update(_make_frame(1))
        f_latest = _make_frame(99)
        stacked = buf.update(f_latest)
        assert torch.equal(stacked[..., CHANNELS:], f_latest)

    def test_three_frame_stack_oldest_to_newest_order(self):
        """frame_stack=3 produces oldest→newest across 3 channel slices."""
        buf = FrameStackBuffer(SINGLE_SHAPE, frame_stack=3, device="cpu")
        buf.update(_make_frame(10))  # init: all 3 slots = 10
        buf.update(_make_frame(20))  # slots: [10, 10, 20]
        stacked = buf.update(_make_frame(30))  # slots: [10, 20, 30]
        assert torch.equal(stacked[..., :CHANNELS], _make_frame(10))
        assert torch.equal(stacked[..., CHANNELS : 2 * CHANNELS], _make_frame(20))
        assert torch.equal(stacked[..., 2 * CHANNELS :], _make_frame(30))

    def test_reset_all_envs(self):
        """reset() with no args re-inits every env on the next update."""
        buf = FrameStackBuffer(SINGLE_SHAPE, frame_stack=2, device="cpu")
        buf.update(_make_frame(1))
        buf.update(_make_frame(2))  # ring filled
        buf.reset()  # mark all envs for init
        stacked = buf.update(_make_frame(50))
        # All slots filled with 50.
        assert torch.equal(stacked[..., :CHANNELS], _make_frame(50))
        assert torch.equal(stacked[..., CHANNELS:], _make_frame(50))

    def test_reset_partial_envs_preserves_others(self):
        """Resetting env 0 should re-init only env 0; other envs keep their history."""
        buf = FrameStackBuffer(SINGLE_SHAPE, frame_stack=2, device="cpu")
        buf.update(_make_frame(1))
        buf.update(_make_frame(2))
        buf.reset(torch.tensor([0]))
        stacked = buf.update(_make_frame(9))
        # Env 0: both slots == 9 (init).
        assert torch.equal(stacked[0, ..., :CHANNELS], torch.full((HEIGHT, WIDTH, CHANNELS), 9, dtype=torch.uint8))
        assert torch.equal(stacked[0, ..., CHANNELS:], torch.full((HEIGHT, WIDTH, CHANNELS), 9, dtype=torch.uint8))
        # Env 1: oldest == 2 (ring shifted from previous), newest == 9.
        assert torch.equal(stacked[1, ..., :CHANNELS], torch.full((HEIGHT, WIDTH, CHANNELS), 2, dtype=torch.uint8))
        assert torch.equal(stacked[1, ..., CHANNELS:], torch.full((HEIGHT, WIDTH, CHANNELS), 9, dtype=torch.uint8))

    def test_frame_stack_one_passthrough(self):
        """frame_stack=1 effectively echoes the input (single-slot ring)."""
        buf = FrameStackBuffer(SINGLE_SHAPE, frame_stack=1, device="cpu")
        assert buf.output_shape == SINGLE_SHAPE
        f = _make_frame(42)
        stacked = buf.update(f)
        assert torch.equal(stacked, f)

    def test_invalid_frame_stack_raises(self):
        with pytest.raises(ValueError, match="frame_stack must be >= 1"):
            FrameStackBuffer(SINGLE_SHAPE, frame_stack=0, device="cpu")

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="at least 2 dims"):
            FrameStackBuffer((10,), frame_stack=2, device="cpu")

    def test_wrong_input_shape_raises(self):
        """update() rejects a frame whose shape doesn't match the construction shape."""
        buf = FrameStackBuffer(SINGLE_SHAPE, frame_stack=2, device="cpu")
        with pytest.raises(ValueError, match="does not match expected"):
            buf.update(torch.zeros((NUM_ENVS, HEIGHT, WIDTH, CHANNELS + 1), dtype=torch.uint8))

    def test_dtype_preserved_uint8(self):
        buf = FrameStackBuffer(SINGLE_SHAPE, frame_stack=2, device="cpu", dtype=torch.uint8)
        stacked = buf.update(_make_frame(5))
        assert stacked.dtype == torch.uint8

    def test_dtype_preserved_float32(self):
        buf = FrameStackBuffer(SINGLE_SHAPE, frame_stack=2, device="cpu", dtype=torch.float32)
        stacked = buf.update(_make_frame(5, dtype=torch.float32))
        assert stacked.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available in this env")
    def test_buffer_on_cuda(self):
        """Buffer allocates and operates correctly on a CUDA device."""
        buf = FrameStackBuffer(SINGLE_SHAPE, frame_stack=2, device="cuda")
        f0 = torch.full(SINGLE_SHAPE, 7, dtype=torch.uint8, device="cuda")
        stacked = buf.update(f0)
        assert stacked.device.type == "cuda"
        assert stacked.shape == (NUM_ENVS, HEIGHT, WIDTH, CHANNELS * 2)
        # Both slots filled with f0 on the init path.
        assert torch.equal(stacked[..., :CHANNELS], f0)
        assert torch.equal(stacked[..., CHANNELS:], f0)
        # Steady-state shift works on CUDA too.
        f1 = torch.full(SINGLE_SHAPE, 13, dtype=torch.uint8, device="cuda")
        stacked = buf.update(f1)
        assert torch.equal(stacked[..., :CHANNELS], f0)
        assert torch.equal(stacked[..., CHANNELS:], f1)

    def test_long_run_ring_stability(self):
        """After many updates exceeding frame_stack cycles, the oldest-to-newest layout stays correct."""
        buf = FrameStackBuffer(SINGLE_SHAPE, frame_stack=3, device="cpu")
        # Push 11 frames with values 0..10. After the last update, the ring slots should
        # hold the 3 most-recent frames: [8, 9, 10] in oldest-to-newest order.
        for i in range(11):
            stacked = buf.update(_make_frame(i))
        assert torch.equal(stacked[..., :CHANNELS], _make_frame(8))
        assert torch.equal(stacked[..., CHANNELS : 2 * CHANNELS], _make_frame(9))
        assert torch.equal(stacked[..., 2 * CHANNELS :], _make_frame(10))

    def test_reset_accepts_python_sequence(self):
        """reset() accepts a plain ``list[int]`` (the type DirectRLEnv hands to ``_reset_idx``)."""
        buf = FrameStackBuffer(SINGLE_SHAPE, frame_stack=2, device="cpu")
        buf.update(_make_frame(1))
        buf.update(_make_frame(2))
        buf.reset([0, 2])
        stacked = buf.update(_make_frame(9))
        per_env_shape = (HEIGHT, WIDTH, CHANNELS)
        nines = torch.full(per_env_shape, 9, dtype=torch.uint8)
        twos = torch.full(per_env_shape, 2, dtype=torch.uint8)
        for env_id in (0, 2):
            assert torch.equal(stacked[env_id, ..., :CHANNELS], nines), f"env {env_id} oldest"
            assert torch.equal(stacked[env_id, ..., CHANNELS:], nines), f"env {env_id} newest"
        for env_id in (1, 3):
            assert torch.equal(stacked[env_id, ..., :CHANNELS], twos), f"env {env_id} oldest"
            assert torch.equal(stacked[env_id, ..., CHANNELS:], nines), f"env {env_id} newest"

    def test_reset_multi_env_subset_preserves_unrelated(self):
        """Resetting envs [0, 2] should re-init only those; envs [1, 3] keep their history."""
        buf = FrameStackBuffer(SINGLE_SHAPE, frame_stack=2, device="cpu")
        buf.update(_make_frame(1))
        buf.update(_make_frame(2))  # ring filled
        buf.reset(torch.tensor([0, 2]))
        stacked = buf.update(_make_frame(9))
        per_env_shape = (HEIGHT, WIDTH, CHANNELS)
        nines = torch.full(per_env_shape, 9, dtype=torch.uint8)
        twos = torch.full(per_env_shape, 2, dtype=torch.uint8)
        # Reset envs: both slots = 9 (init).
        for env_id in (0, 2):
            assert torch.equal(stacked[env_id, ..., :CHANNELS], nines), f"env {env_id} oldest"
            assert torch.equal(stacked[env_id, ..., CHANNELS:], nines), f"env {env_id} newest"
        # Untouched envs: oldest = 2 (shifted from previous newest), newest = 9.
        for env_id in (1, 3):
            assert torch.equal(stacked[env_id, ..., :CHANNELS], twos), f"env {env_id} oldest"
            assert torch.equal(stacked[env_id, ..., CHANNELS:], nines), f"env {env_id} newest"
