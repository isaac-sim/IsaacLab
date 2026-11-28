# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.utils.math import sample_gaussian

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mean, std", [(2, 0), (2, 0.0), (2.0, 0)])
def test_integer_gaussian_scalar_parameters(mean, std):
    samples = sample_gaussian(mean, std, (2, 3), "cpu")
    torch.testing.assert_close(samples, torch.full((2, 3), 2.0))


def test_integer_gaussian_tensor_parameters():
    mean = torch.tensor([[1], [2]])
    samples = sample_gaussian(mean, torch.tensor(0), (2, 3), "cpu")
    torch.testing.assert_close(samples, mean.to(torch.get_default_dtype()).expand(2, 3))


@pytest.mark.parametrize(
    "mean, std",
    [(16777217, torch.tensor(0.0, dtype=torch.float64)), (torch.tensor(16777217.0, dtype=torch.float64), 0)],
)
def test_gaussian_preserves_floating_parameter_precision(mean, std):
    samples = sample_gaussian(mean, std, (3,), "cpu")
    torch.testing.assert_close(samples, torch.full((3,), 16777217.0, dtype=torch.float64), rtol=0, atol=0)


def test_gaussian_integer_negative_std_is_rejected():
    with pytest.raises(RuntimeError, match="std"):
        sample_gaussian(0, -1, 3, "cpu")
