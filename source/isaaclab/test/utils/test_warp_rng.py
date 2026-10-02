# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the process-wide per-environment Warp random number generator state."""

import numpy as np
import pytest
import torch

from isaaclab.test.utils import test_devices
from isaaclab.utils.seed import WarpRng, configure_seed


@pytest.fixture(autouse=True)
def fresh_warp_rng():
    """Start and end every test without a state or a set seed."""
    WarpRng.state = None
    WarpRng._seed = None
    yield
    WarpRng.state = None
    WarpRng._seed = None


@pytest.mark.parametrize("device", test_devices())
def test_initialize_allocates_a_new_state(device):
    """Every :meth:`WarpRng.initialize` call allocates and seeds a new state of the given size."""
    WarpRng.initialize(4, device)
    state = WarpRng.state
    WarpRng.initialize(5, device)
    assert WarpRng.state is not state
    assert WarpRng.state.shape == (5,)


@pytest.mark.parametrize("device", test_devices())
def test_configure_seed_reseeds_in_place(device):
    """configure_seed reseeds the existing state in place and reproducibly."""
    configure_seed(7)
    WarpRng.initialize(4, device)
    state = WarpRng.state
    seeded = state.numpy().copy()

    state.fill_(0)
    configure_seed(7)
    assert WarpRng.state.ptr == state.ptr
    np.testing.assert_array_equal(state.numpy(), seeded)

    configure_seed(8)
    assert not np.array_equal(state.numpy(), seeded)


@pytest.mark.parametrize("device", test_devices())
def test_default_seed_follows_torch(device):
    """Without a set seed, the state is seeded from torch's initial seed."""
    torch.manual_seed(5)
    WarpRng.initialize(4, device)
    default = WarpRng.state.numpy().copy()

    WarpRng.state = None
    WarpRng.seed(5)
    WarpRng.initialize(4, device)
    np.testing.assert_array_equal(WarpRng.state.numpy(), default)
