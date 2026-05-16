# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`isaaclab.envs.mdp.observations.stacked_image`.

Camera output is mocked via :func:`unittest.mock.patch` so the tests exercise the
ring-buffer + channel-stacking logic without needing a Kit launch or a real sensor.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

pytestmark = pytest.mark.isaacsim_ci

from isaaclab.envs.mdp.observations import stacked_image

NUM_ENVS = 4
HEIGHT = 8
WIDTH = 8
CHANNELS = 3


def _make_env(num_envs: int = NUM_ENVS, device: str = "cpu") -> SimpleNamespace:
    """Minimal mock env surface needed by ``stacked_image``."""
    return SimpleNamespace(num_envs=num_envs, device=device)


def _make_cfg(frame_stack: int) -> SimpleNamespace:
    """Minimal mock cfg with the params dict the term reads at init."""
    return SimpleNamespace(params={"frame_stack": frame_stack})


def _frame(value: int) -> torch.Tensor:
    """Build a constant-valued ``(N, H, W, C)`` frame."""
    return torch.full((NUM_ENVS, HEIGHT, WIDTH, CHANNELS), value, dtype=torch.float32)


class TestStackedImage:
    """Tests for the ``stacked_image`` observation term."""

    def test_output_shape_channel_stacked(self):
        """Output shape is ``(N, H, W, K * C)``."""
        env = _make_env()
        term = stacked_image(_make_cfg(frame_stack=3), env)
        with mock.patch("isaaclab.envs.mdp.observations.image", return_value=_frame(1)):
            out = term(env)
        assert out.shape == (NUM_ENVS, HEIGHT, WIDTH, CHANNELS * 3)

    def test_warmup_fills_all_slots_with_first_frame(self):
        """First call after construction fills all ``K`` slots with that one frame."""
        env = _make_env()
        term = stacked_image(_make_cfg(frame_stack=2), env)
        with mock.patch("isaaclab.envs.mdp.observations.image", return_value=_frame(7)):
            out = term(env)
        f7 = _frame(7)
        assert torch.equal(out[..., :CHANNELS], f7)
        assert torch.equal(out[..., CHANNELS:], f7)

    def test_oldest_to_newest_channel_order(self):
        """K=3 with three distinct frames produces oldest→newest along the channel dim."""
        env = _make_env()
        term = stacked_image(_make_cfg(frame_stack=3), env)
        with mock.patch("isaaclab.envs.mdp.observations.image") as patched:
            patched.return_value = _frame(10)
            term(env)  # init: all 3 slots = 10
            patched.return_value = _frame(20)
            term(env)  # slots: [10, 10, 20]
            patched.return_value = _frame(30)
            out = term(env)  # slots: [10, 20, 30]
        assert torch.equal(out[..., :CHANNELS], _frame(10))
        assert torch.equal(out[..., CHANNELS : 2 * CHANNELS], _frame(20))
        assert torch.equal(out[..., 2 * CHANNELS :], _frame(30))

    def test_reset_all_envs_clears_history(self):
        """``reset()`` with no args re-inits every env on the next call."""
        env = _make_env()
        term = stacked_image(_make_cfg(frame_stack=2), env)
        with mock.patch("isaaclab.envs.mdp.observations.image") as patched:
            patched.return_value = _frame(1)
            term(env)
            patched.return_value = _frame(2)
            term(env)  # ring filled
            term.reset()
            patched.return_value = _frame(50)
            out = term(env)
        assert torch.equal(out[..., :CHANNELS], _frame(50))
        assert torch.equal(out[..., CHANNELS:], _frame(50))

    def test_reset_partial_envs_preserves_others(self):
        """Resetting env 0 re-inits only env 0; other envs keep their history."""
        env = _make_env()
        term = stacked_image(_make_cfg(frame_stack=2), env)
        with mock.patch("isaaclab.envs.mdp.observations.image") as patched:
            patched.return_value = _frame(1)
            term(env)
            patched.return_value = _frame(2)
            term(env)
            term.reset(torch.tensor([0]))
            patched.return_value = _frame(9)
            out = term(env)
        per_env_shape = (HEIGHT, WIDTH, CHANNELS)
        nines = torch.full(per_env_shape, 9, dtype=torch.float32)
        twos = torch.full(per_env_shape, 2, dtype=torch.float32)
        # Env 0: both slots = 9 (init path fired again).
        assert torch.equal(out[0, ..., :CHANNELS], nines)
        assert torch.equal(out[0, ..., CHANNELS:], nines)
        # Env 1: oldest = 2 (shifted from previous newest), newest = 9.
        assert torch.equal(out[1, ..., :CHANNELS], twos)
        assert torch.equal(out[1, ..., CHANNELS:], nines)

    def test_invalid_frame_stack_raises(self):
        """``frame_stack < 1`` is rejected at construction time."""
        with pytest.raises(ValueError, match="frame_stack must be >= 1"):
            stacked_image(_make_cfg(frame_stack=0), _make_env())

    def test_frame_stack_one_passthrough(self):
        """``frame_stack=1`` short-circuits the buffer; output equals the single input frame."""
        env = _make_env()
        term = stacked_image(_make_cfg(frame_stack=1), env)
        f = _frame(42)
        with mock.patch("isaaclab.envs.mdp.observations.image", return_value=f):
            out = term(env)
        assert out.shape == (NUM_ENVS, HEIGHT, WIDTH, CHANNELS)
        assert torch.equal(out, f)

    def test_long_run_ring_stability(self):
        """After updates well past ``frame_stack`` cycles, the layout stays correct."""
        env = _make_env()
        term = stacked_image(_make_cfg(frame_stack=3), env)
        with mock.patch("isaaclab.envs.mdp.observations.image") as patched:
            for i in range(11):
                patched.return_value = _frame(i)
                out = term(env)
        # 11 frames with values 0..10; final ring holds the 3 most recent in oldest→newest order.
        assert torch.equal(out[..., :CHANNELS], _frame(8))
        assert torch.equal(out[..., CHANNELS : 2 * CHANNELS], _frame(9))
        assert torch.equal(out[..., 2 * CHANNELS :], _frame(10))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available in this env")
    def test_buffer_on_cuda(self):
        """Term allocates and operates correctly on a CUDA device."""
        env = _make_env(device="cuda")
        term = stacked_image(_make_cfg(frame_stack=2), env)
        cuda_frame = torch.full((NUM_ENVS, HEIGHT, WIDTH, CHANNELS), 7.0, dtype=torch.float32, device="cuda")
        with mock.patch("isaaclab.envs.mdp.observations.image", return_value=cuda_frame):
            out = term(env)
        assert out.device.type == "cuda"
        assert out.shape == (NUM_ENVS, HEIGHT, WIDTH, CHANNELS * 2)
        # Init path fires; both slots hold the same frame.
        assert torch.equal(out[..., :CHANNELS], cuda_frame)
        assert torch.equal(out[..., CHANNELS:], cuda_frame)
