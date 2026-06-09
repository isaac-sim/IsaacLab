# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------------------------

import torch


def _soft_dtw(D: torch.Tensor, gamma: float, bandwidth: float) -> torch.Tensor:
    """Compute SoftDTW from a batched pairwise distance matrix using Torch ops."""
    batch_size, len_x, len_y = D.shape
    inf = torch.full((batch_size,), float("inf"), device=D.device, dtype=D.dtype)
    prev_row = [inf] * (len_y + 2)
    prev_row[0] = torch.zeros((batch_size,), device=D.device, dtype=D.dtype)

    for i in range(1, len_x + 1):
        curr_row = [inf] * (len_y + 2)
        for j in range(1, len_y + 1):
            if 0 < bandwidth < abs(i - j):
                continue

            if gamma == 0:
                softmin = torch.minimum(torch.minimum(prev_row[j - 1], prev_row[j]), curr_row[j - 1])
            else:
                previous_costs = torch.stack((prev_row[j - 1], prev_row[j], curr_row[j - 1]))
                softmin = -gamma * torch.logsumexp(-previous_costs / gamma, dim=0)

            curr_row[j] = D[:, i - 1, j - 1] + softmin
        prev_row = curr_row

    return prev_row[len_y]


class SoftDTW(torch.nn.Module):
    """Soft Dynamic Time Warping implemented with Torch tensor operations.

    The ``use_cuda`` and ``device`` arguments are kept for compatibility with the
    previous AutoMate SoftDTW helper. The implementation runs on the device of the
    input tensors and does not require Numba.
    """

    def __init__(self, use_cuda, device, gamma=1.0, normalize=False, bandwidth=None, dist_func=None):
        """Initializes a new instance using the supplied parameters.

        Args:
            use_cuda: Preserved for API compatibility. Inputs already determine the execution device.
            device: Preserved for API compatibility. Inputs already determine the execution device.
            gamma: The SoftDTW gamma parameter. Set to 0 for original DTW without smoothing.
            normalize: Whether to perform normalization. Default is False.
            bandwidth: Sakoe-Chiba bandwidth for pruning. Default is None, which disables pruning.
            dist_func: The point-wise distance function to use. Default is squared Euclidean distance.
        """
        super().__init__()
        self.normalize = normalize
        self.gamma = float(gamma)
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda
        self.device = device

        if dist_func is not None:
            self.dist_func = dist_func
        else:
            self.dist_func = SoftDTW._euclidean_dist_func

    @staticmethod
    def _euclidean_dist_func(x, y):
        """Calculates the squared Euclidean distance between each element in x and y per timestep."""
        num_x = x.size(1)
        num_y = y.size(1)
        dims = x.size(2)
        x = x.unsqueeze(2).expand(-1, num_x, num_y, dims)
        y = y.unsqueeze(1).expand(-1, num_x, num_y, dims)
        return torch.pow(x - y, 2).sum(3)

    def forward(self, X, Y):
        """Compute the SoftDTW value between ``X`` and ``Y``."""
        if self.normalize:
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = _soft_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 0.5 * (out_xx + out_yy)

        D_xy = self.dist_func(X, Y)
        return _soft_dtw(D_xy, self.gamma, self.bandwidth)
