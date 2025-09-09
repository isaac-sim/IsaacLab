# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.utils.benchmark as benchmark

import pytest


@pytest.mark.isaacsim_ci
def test_array_slicing():
    """Check that using ellipsis and slices work for torch tensors."""

    size = (400, 300, 5)
    my_tensor = torch.rand(size, device="cuda:0")

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


@pytest.mark.isaacsim_ci
def test_array_circular():
    """Check circular buffer implementation in torch."""

    size = (10, 30, 5)
    my_tensor = torch.rand(size, device="cuda:0")

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


@pytest.mark.isaacsim_ci
def test_array_circular_copy():
    """Check that circular buffer implementation in torch is copying data."""

    size = (10, 30, 5)
    my_tensor = torch.rand(size, device="cuda:0")
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


@pytest.mark.isaacsim_ci
def test_array_multi_indexing():
    """Check multi-indexing works for torch tensors."""

    size = (400, 300, 5)
    my_tensor = torch.rand(size, device="cuda:0")

    # this fails since array indexing cannot be broadcasted!!
    with pytest.raises(IndexError):
        my_tensor[[0, 1, 2, 3], [0, 1, 2, 3, 4]]


@pytest.mark.isaacsim_ci
def test_array_single_indexing():
    """Check how indexing effects the returned tensor."""

    size = (400, 300, 5)
    my_tensor = torch.rand(size, device="cuda:0")

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


@pytest.mark.isaacsim_ci
def test_logical_or():
    """Test bitwise or operation."""

    size = (400, 300, 5)
    my_tensor_1 = torch.rand(size, device="cuda:0") > 0.5
    my_tensor_2 = torch.rand(size, device="cuda:0") < 0.5

    # check the speed of logical or
    timer_logical_or = benchmark.Timer(
        stmt="torch.logical_or(my_tensor_1, my_tensor_2)",
        globals={"my_tensor_1": my_tensor_1, "my_tensor_2": my_tensor_2},
    )
    timer_bitwise_or = benchmark.Timer(
        stmt="my_tensor_1 | my_tensor_2", globals={"my_tensor_1": my_tensor_1, "my_tensor_2": my_tensor_2}
    )

    print("Time for logical or:", timer_logical_or.timeit(number=1000))
    print("Time for bitwise or:", timer_bitwise_or.timeit(number=1000))
    # check that logical or works as expected
    output_logical_or = torch.logical_or(my_tensor_1, my_tensor_2)
    output_bitwise_or = my_tensor_1 | my_tensor_2

    assert torch.allclose(output_logical_or, output_bitwise_or)
