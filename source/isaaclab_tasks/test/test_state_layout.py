# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`StateLayout`."""

from __future__ import annotations

import pytest
import torch

from isaaclab_tasks.core.multi_task.utils import StateLayout


def test_state_layout_basic_2d():
    """Terrain-shaped layout: 2D coords, spawn != target, paired items."""
    coords = torch.rand(50, 2)
    spawn = torch.randint(0, 50, (100,), dtype=torch.long)
    target = torch.randint(0, 50, (100,), dtype=torch.long)
    layout = StateLayout(coords=coords, spawn_index=spawn, target_index=target)
    assert layout.num_states == 50
    assert layout.num_items == 100
    assert layout.coord_dim == 2
    assert layout.target_index is not None


def test_state_layout_basic_3d():
    """Factory-shaped layout: 3D coords, slot==item (target_index=None)."""
    coords = torch.rand(64, 3)
    spawn = torch.arange(64, dtype=torch.long)
    layout = StateLayout(coords=coords, spawn_index=spawn, target_index=None)
    assert layout.num_states == 64
    assert layout.num_items == 64
    assert layout.coord_dim == 3
    assert layout.target_index is None


def test_state_layout_target_default_is_none():
    """Default for ``target_index`` is ``None`` (slot==item topology)."""
    coords = torch.rand(10, 2)
    spawn = torch.arange(10, dtype=torch.long)
    layout = StateLayout(coords=coords, spawn_index=spawn)
    assert layout.target_index is None


def test_state_layout_high_dim_coords_allowed():
    """``coord_dim`` is arbitrary; a 5D pool should still construct."""
    coords = torch.rand(20, 5)
    spawn = torch.arange(20, dtype=torch.long)
    layout = StateLayout(coords=coords, spawn_index=spawn)
    assert layout.coord_dim == 5
    assert layout.num_states == 20


def test_state_layout_shape_mismatch_raises():
    """``spawn_index`` and ``target_index`` lengths must match."""
    coords = torch.rand(50, 2)
    spawn = torch.randint(0, 50, (100,), dtype=torch.long)
    target = torch.randint(0, 50, (50,), dtype=torch.long)  # wrong size
    with pytest.raises(ValueError):
        StateLayout(coords=coords, spawn_index=spawn, target_index=target)


def test_state_layout_invalid_coord_shape_raises():
    """``coords`` must be 2D ``[num_states, coord_dim]``."""
    spawn = torch.arange(50, dtype=torch.long)
    with pytest.raises(ValueError):
        StateLayout(coords=torch.rand(50), spawn_index=spawn)  # 1D coords
    with pytest.raises(ValueError):
        StateLayout(coords=torch.rand(50, 2, 3), spawn_index=spawn)  # 3D coords


def test_state_layout_invalid_spawn_shape_raises():
    """``spawn_index`` must be 1D."""
    with pytest.raises(ValueError):
        StateLayout(
            coords=torch.rand(50, 2),
            spawn_index=torch.randint(0, 50, (10, 2), dtype=torch.long),
        )
