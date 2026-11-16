# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""The torch operation guard a recording holds while a graph records (see :class:`_TorchOpGuard`)."""

import pytest
import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.capture import _TorchOpGuard

pytestmark = pytest.mark.unit

CPU = torch.device("cpu")


@pytest.mark.parametrize(
    "operation",
    [
        lambda t: t + 1,
        lambda t: t.sum(),
        lambda t: t[0].item(),
        lambda t: t.zero_(),
        lambda t: torch.zeros(3, device="cpu"),
        lambda t: t.clone(),
    ],
    ids=["add", "sum", "item", "zero_", "zeros", "clone"],
)
def test_the_guard_raises_on_work_on_its_device(operation):
    t = torch.ones(4)
    with pytest.raises(RuntimeError, match="runs outside the graph a recording captures"), _TorchOpGuard(CPU):
        operation(t)


def test_the_guard_passes_views_and_warp_memory():
    """Views launch no work; wrapping Warp memory is a view of it (``aten.lift_fresh``)."""
    t = torch.ones(4, 2)
    array = wp.zeros(4, dtype=wp.float32, device="cpu")
    proxy = ProxyArray(wp.zeros(4, dtype=wp.vec3f, device="cpu"))
    with _TorchOpGuard(CPU):
        t.view(8)
        t[1:]
        t.reshape(2, 4)
        t.t()
        wp.to_torch(array)
        proxy.torch[0:2]


def test_the_guard_passes_work_on_another_device():
    with _TorchOpGuard(torch.device("cuda", 0)):
        torch.ones(4).sum()
