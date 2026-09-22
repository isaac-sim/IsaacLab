# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.test.utils import test_devices

pytestmark = [pytest.mark.unit, pytest.mark.arm_ci]


@pytest.mark.parametrize("device", test_devices())
def test_array_slicing(device):
    """Check that using ellipsis and slices work for torch tensors."""

    size = (400, 300, 5)
    my_tensor = torch.rand(size, device=device)

    assert my_tensor[..., 0].shape == (400, 300)
    assert my_tensor[:, :, 0].shape == (400, 300)
    assert my_tensor[slice(None), slice(None), 0].shape == (400, 300)
    with pytest.raises(IndexError):
        my_tensor[..., ..., 0]

    assert my_tensor[0, ...].shape == (300, 5)
    assert my_tensor[0, :, :].shape == (300, 5)
    assert my_tensor[0, slice(None), slice(None)].shape == (300, 5)
    assert my_tensor[0, ..., ...].shape == (300, 5)

    assert my_tensor[..., 0, 0].shape == (400,)
    assert my_tensor[slice(None), 0, 0].shape == (400,)
    assert my_tensor[:, 0, 0].shape == (400,)


@pytest.mark.parametrize("device", test_devices())
def test_array_circular(device):
    """Check circular buffer implementation in torch."""

    size = (10, 30, 5)
    my_tensor = torch.rand(size, device=device)

    # roll up the tensor without cloning
    my_tensor_1 = my_tensor.clone()
    my_tensor_1[:, 1:, :] = my_tensor_1[:, :-1, :]
    my_tensor_1[:, 0, :] = my_tensor[:, -1, :]
    # check that circular buffer works as expected
    error = torch.max(torch.abs(my_tensor_1 - my_tensor.roll(1, dims=1)))
    assert error.item() != 0.0
    assert not torch.allclose(my_tensor_1, my_tensor.roll(1, dims=1))

    # roll up the tensor with cloning
    my_tensor_2 = my_tensor.clone()
    my_tensor_2[:, 1:, :] = my_tensor_2[:, :-1, :].clone()
    my_tensor_2[:, 0, :] = my_tensor[:, -1, :]
    # check that circular buffer works as expected
    error = torch.max(torch.abs(my_tensor_2 - my_tensor.roll(1, dims=1)))
    assert error.item() == 0.0
    assert torch.allclose(my_tensor_2, my_tensor.roll(1, dims=1))

    # roll up the tensor with detach operation
    my_tensor_3 = my_tensor.clone()
    my_tensor_3[:, 1:, :] = my_tensor_3[:, :-1, :].detach()
    my_tensor_3[:, 0, :] = my_tensor[:, -1, :]
    # check that circular buffer works as expected
    error = torch.max(torch.abs(my_tensor_3 - my_tensor.roll(1, dims=1)))
    assert error.item() != 0.0
    assert not torch.allclose(my_tensor_3, my_tensor.roll(1, dims=1))

    # roll up the tensor with roll operation
    my_tensor_4 = my_tensor.clone()
    my_tensor_4 = my_tensor_4.roll(1, dims=1)
    my_tensor_4[:, 0, :] = my_tensor[:, -1, :]
    # check that circular buffer works as expected
    error = torch.max(torch.abs(my_tensor_4 - my_tensor.roll(1, dims=1)))
    assert error.item() == 0.0
    assert torch.allclose(my_tensor_4, my_tensor.roll(1, dims=1))


@pytest.mark.parametrize("device", test_devices())
def test_array_circular_copy(device):
    """Check that circular buffer implementation in torch is copying data."""

    size = (10, 30, 5)
    my_tensor = torch.rand(size, device=device)
    my_tensor_clone = my_tensor.clone()

    # roll up the tensor
    my_tensor_1 = my_tensor.clone()
    my_tensor_1[:, 1:, :] = my_tensor_1[:, :-1, :].clone()
    my_tensor_1[:, 0, :] = my_tensor[:, -1, :]
    # change the source tensor
    my_tensor[:, 0, :] = 1000
    # check that circular buffer works as expected
    assert not torch.allclose(my_tensor_1, my_tensor.roll(1, dims=1))
    assert torch.allclose(my_tensor_1, my_tensor_clone.roll(1, dims=1))


@pytest.mark.parametrize("device", test_devices())
def test_array_multi_indexing(device):
    """Check multi-indexing works for torch tensors."""

    size = (400, 300, 5)
    my_tensor = torch.rand(size, device=device)

    # this fails since array indexing cannot be broadcasted!!
    with pytest.raises(IndexError):
        my_tensor[[0, 1, 2, 3], [0, 1, 2, 3, 4]]


@pytest.mark.parametrize("device", test_devices())
def test_array_single_indexing(device):
    """Check how indexing effects the returned tensor."""

    size = (400, 300, 5)
    my_tensor = torch.rand(size, device=device)

    # obtain a slice of the tensor
    my_slice = my_tensor[0, ...]
    assert my_slice.untyped_storage().data_ptr() == my_tensor.untyped_storage().data_ptr()

    # obtain a slice over ranges
    my_slice = my_tensor[0:2, ...]
    assert my_slice.untyped_storage().data_ptr() == my_tensor.untyped_storage().data_ptr()

    # obtain a slice over list
    my_slice = my_tensor[[0, 1], ...]
    assert my_slice.untyped_storage().data_ptr() != my_tensor.untyped_storage().data_ptr()

    # obtain a slice over tensor
    my_slice = my_tensor[torch.tensor([0, 1]), ...]
    assert my_slice.untyped_storage().data_ptr() != my_tensor.untyped_storage().data_ptr()


@pytest.mark.parametrize("device", test_devices())
def test_logical_or(device):
    """Bitwise ``|`` on boolean tensors matches ``torch.logical_or``."""
    size = (400, 300, 5)
    my_tensor_1 = torch.rand(size, device=device) > 0.5
    my_tensor_2 = torch.rand(size, device=device) < 0.5

    assert torch.equal(torch.logical_or(my_tensor_1, my_tensor_2), my_tensor_1 | my_tensor_2)
