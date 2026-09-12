# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.utils import DelayBuffer

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("invalid_lag", [-1, 4])
def test_invalid_time_lag_does_not_mutate_active_state(invalid_lag):
    buffer = DelayBuffer(history_length=3, batch_size=3, device="cpu")
    buffer.set_time_lag(torch.tensor([0, 1, 2], dtype=torch.int))

    previous_lags = buffer.time_lags.clone()
    previous_min = buffer.min_time_lag
    previous_max = buffer.max_time_lag

    with pytest.raises(ValueError):
        buffer.set_time_lag(invalid_lag, batch_ids=[1])

    torch.testing.assert_close(buffer.time_lags, previous_lags)
    assert buffer.min_time_lag == previous_min
    assert buffer.max_time_lag == previous_max

    # The rejected update must not change externally observable delayed output.
    for step in range(3):
        buffer.compute(torch.tensor([[step], [10 + step], [20 + step]], dtype=torch.float))
    output = buffer.compute(torch.tensor([[3], [13], [23]], dtype=torch.float))

    expected = torch.tensor([[3], [12], [21]], dtype=torch.float)
    torch.testing.assert_close(output, expected)
