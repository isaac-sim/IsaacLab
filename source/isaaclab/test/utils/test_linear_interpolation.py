# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.utils.interpolation import LinearInterpolation

pytestmark = pytest.mark.unit


def test_compute_accepts_noncontiguous_query_tensor():
    """Arbitrarily shaped query tensors should not need contiguous storage."""
    interpolation = LinearInterpolation(
        x=torch.tensor([0.0, 1.0, 2.0, 3.0]),
        y=torch.tensor([0.0, 10.0, 20.0, 30.0]),
        device="cpu",
    )
    query = torch.tensor([[0.5, 1.5, 2.5], [3.0, 2.0, 1.0]]).T
    assert not query.is_contiguous()

    output = interpolation.compute(query)

    assert output.shape == query.shape
    torch.testing.assert_close(output, query * 10.0)
