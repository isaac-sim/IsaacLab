# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


class LinearInterpolation:
    """Linearly interpolates a sampled scalar function for arbitrary query points.

    This class implements a linear interpolation for a scalar function. The function maps from real values, x, to
    real values, y. It expects a set of samples from the function's domain, x, and the corresponding values, y.
    The class allows querying the function's values at any arbitrary point.

    The interpolation is done by finding the two closest points in x to the query point and then linearly
    interpolating between the corresponding y values. For the query points that are outside the input points,
    the class does a zero-order-hold extrapolation based on the boundary values. This means that the class
    returns the value of the closest point in x.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, device: str):
        """Initializes the linear interpolation.

        The scalar function maps from real values, x, to real values, y. The input to the class is a set of samples
        from the function's domain, x, and the corresponding values, y.

        Note:
            The input tensor x should be sorted in ascending order.

        Args:
            x: An vector of samples from the function's domain. The values should be sorted in ascending order.
                Shape is (num_samples,)
            y: The function's values associated to the input x. Shape is (num_samples,)
            device: The device used for processing.

        Raises:
            ValueError: If the input tensors are empty or have different sizes.
            ValueError: If the input tensor x is not sorted in ascending order.
        """
        # make sure that input tensors are 1D of size (num_samples,)
        self._x = x.view(-1).clone().to(device=device)
        self._y = y.view(-1).clone().to(device=device)

        # make sure sizes are correct
        if self._x.numel() == 0:
            raise ValueError("Input tensor x is empty!")
        if self._x.numel() != self._y.numel():
            raise ValueError(f"Input tensors x and y have different sizes: {self._x.numel()} != {self._y.numel()}")
        # make sure that x is sorted
        if torch.any(self._x[1:] < self._x[:-1]):
            raise ValueError("Input tensor x is not sorted in ascending order!")

    def compute(self, q: torch.Tensor) -> torch.Tensor:
        """Calculates a linearly interpolated values for the query points.

        Args:
           q: The query points. It can have any arbitrary shape.

        Returns:
            The interpolated values at query points. It has the same shape as the input tensor.
        """
        # serialized q
        q_1d = q.view(-1)
        # Number of elements in the x that are strictly smaller than query points (use int32 instead of int64)
        num_smaller_elements = torch.sum(self._x.unsqueeze(1) < q_1d.unsqueeze(0), dim=0, dtype=torch.int)

        # The index pointing to the first element in x such that x[lower_bound_i] < q_i
        # If a point is smaller that all x elements, it will assign 0
        lower_bound = torch.clamp(num_smaller_elements - 1, min=0)
        # The index pointing to the first element in x such that x[upper_bound_i] >= q_i
        # If a point is greater than all x elements, it will assign the last elements' index
        upper_bound = torch.clamp(num_smaller_elements, max=self._x.numel() - 1)

        # compute the weight as: (q_i - x_lb) / (x_ub - x_lb)
        weight = (q_1d - self._x[lower_bound]) / (self._x[upper_bound] - self._x[lower_bound])
        # If a point is out of bounds assign weight 0.0
        weight[upper_bound == lower_bound] = 0.0

        # Perform linear interpolation
        fq = self._y[lower_bound] + weight * (self._y[upper_bound] - self._y[lower_bound])

        # deserialized fq
        fq = fq.view(q.shape)
        return fq
