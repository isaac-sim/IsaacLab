# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Under Kit, a recording captures a nonblocking stream in relaxed mode, where CUDA lets torch work on the legacy stream
run once at recording time; the recording's torch guard must still raise."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation()

import pytest  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

from isaaclab.utils.version import has_kit  # noqa: E402
from isaaclab.utils.warp.capture import GraphedCallable  # noqa: E402

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

DEVICE = "cuda:0"


def test_a_torch_operation_while_recording_under_kit_raises(monkeypatch):
    monkeypatch.setattr(GraphedCallable, "enabled", True)
    assert has_kit(), "this test checks the relaxed capture Kit uses"
    values = torch.ones(4, device=DEVICE)
    graphed = GraphedCallable(lambda: values.add_(1), DEVICE)
    with pytest.raises(RuntimeError, match=r"while it records.*aten\.add_"):
        graphed()
    assert values.sum().item() == 4.0, "the operation must not have run"
