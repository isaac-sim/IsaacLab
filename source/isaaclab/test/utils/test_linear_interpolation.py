# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for piecewise linear interpolation."""

import pytest
import torch

from isaaclab.utils.interpolation import LinearInterpolation

pytestmark = pytest.mark.unit


def test_interpolation_boundary_gradients() -> None:
    """Constant extrapolation should have zero, finite query gradients."""
    x = torch.tensor([0.0, 1.0, 3.0])
    y = torch.tensor([1.0, 3.0, 11.0])
    interpolator = LinearInterpolation(x, y, device="cpu")
    q = torch.tensor([[-2.0, 0.5], [2.0, 4.0]], requires_grad=True)
    result = interpolator.compute(q)
    torch.testing.assert_close(result, torch.tensor([[1.0, 2.0], [7.0, 11.0]]))
    (grad,) = torch.autograd.grad(result.sum(), q)
    torch.testing.assert_close(grad, torch.tensor([[0.0, 2.0], [4.0, 0.0]]))


def test_single_sample_interpolation_gradients() -> None:
    """A one-sample curve is constant over the entire domain."""
    interpolator = LinearInterpolation(torch.tensor([1.0]), torch.tensor([5.0]), device="cpu")
    q = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    result = interpolator.compute(q)
    torch.testing.assert_close(result, torch.full_like(q, 5.0))
    (grad,) = torch.autograd.grad(result.sum(), q)
    torch.testing.assert_close(grad, torch.zeros_like(q))
