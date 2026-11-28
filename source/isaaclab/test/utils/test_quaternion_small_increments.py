# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import pytest
import torch

from isaaclab.utils.math import quat_box_plus, quat_mul, quat_unique

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("angle", [0.0, 1e-12, 1e-8, 5e-7, 1e-6, 0.2])
def test_box_plus_preserves_increment_magnitude(dtype, angle):
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=dtype)
    delta = torch.tensor([[angle, 0.0, 0.0]], dtype=dtype)
    expected = torch.tensor([[math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)]], dtype=dtype)
    torch.testing.assert_close(quat_box_plus(q, delta), expected, atol=0.0, rtol=1e-6)


def test_box_plus_zero_increment_has_correct_gradient():
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    delta = torch.zeros((1, 3), dtype=q.dtype, requires_grad=True)
    result = quat_box_plus(q, delta)
    torch.testing.assert_close(result, q)
    result[..., :3].sum().backward()
    torch.testing.assert_close(delta.grad, torch.full_like(delta, 0.5))


def test_box_plus_composes_with_nonidentity_orientation():
    q = torch.tensor([[0.0, 0.6, 0.0, 0.8]], dtype=torch.float64)
    angle = 1e-8
    delta = torch.tensor([[0.0, 0.0, angle]], dtype=q.dtype)
    update = torch.tensor([[0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]], dtype=q.dtype)
    torch.testing.assert_close(quat_box_plus(q, delta), quat_unique(quat_mul(update, q)), atol=1e-15, rtol=1e-12)

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quat_mul_preserves_small_components_and_hamilton_product(dtype):
    q = torch.tensor([[1e-12, 2e-12, -3e-12, 1.0]], dtype=dtype)
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=dtype)
    torch.testing.assert_close(quat_mul(q, identity), q, atol=0.0, rtol=0.0)
    torch.testing.assert_close(quat_mul(identity, q), q, atol=0.0, rtol=0.0)
    i = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype)
    j = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=dtype)
    k = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=dtype)
    torch.testing.assert_close(quat_mul(i, j), k)
    torch.testing.assert_close(quat_mul(j, i), -k)
