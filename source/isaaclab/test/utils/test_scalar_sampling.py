# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.utils.math import sample_log_uniform, sample_triangle, sample_uniform

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("sampler", [sample_uniform, sample_triangle, sample_log_uniform])
@pytest.mark.parametrize("size", [(), 3, (2, 3), (0,)])
def test_sampler_shape_and_bounds(sampler, size):
    expected_shape = (size,) if isinstance(size, int) else size
    result = sampler(1.0, 4.0, size, "cpu")
    assert result.shape == expected_shape
    assert torch.all(result >= 1.0)
    assert torch.all(result <= 4.0)


@pytest.mark.parametrize("sampler", [sample_uniform, sample_triangle, sample_log_uniform])
def test_scalar_sampler_fixed_bounds(sampler):
    result = sampler(2.0, 2.0, (), "cpu")
    torch.testing.assert_close(result, torch.tensor(2.0))
