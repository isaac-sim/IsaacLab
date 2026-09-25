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


def _soft_dtw_autograd(D: torch.Tensor, gamma: float, bandwidth: float) -> torch.Tensor:
    """Compute SoftDTW using Torch ops that preserve autograd."""
    batch_size, len_x, len_y = D.shape
    R = torch.full((batch_size, len_x + 2, len_y + 2), float("inf"), device=D.device, dtype=D.dtype)
    R[:, 0, 0] = 0

    band_size = int(bandwidth) if bandwidth > 0 else max(len_x, len_y)
    for i in range(1, len_x + 1):
        j_start = max(1, i - band_size)
        j_end = min(len_y, i + band_size) + 1

        for j in range(j_start, j_end):
            r0 = R[:, i - 1, j - 1]
            r1 = R[:, i - 1, j]
            r2 = R[:, i, j - 1]

            if gamma == 0:
                softmin = torch.minimum(torch.minimum(r0, r1), r2)
            else:
                previous_costs = torch.stack((r0, r1, r2))
                softmin = -gamma * torch.logsumexp(-previous_costs / gamma, dim=0)

            R[:, i, j] = D[:, i - 1, j - 1] + softmin

    return R[:, len_x, len_y]


def _soft_dtw_no_grad(D: torch.Tensor, gamma: float, bandwidth: float) -> torch.Tensor:
    """Compute SoftDTW by evaluating each dynamic-programming anti-diagonal in one batched op."""
    batch_size, len_x, len_y = D.shape
    R = torch.full((batch_size, len_x + 2, len_y + 2), float("inf"), device=D.device, dtype=D.dtype)
    R[:, 0, 0] = 0

    for diag in range(2, len_x + len_y + 1):
        i = torch.arange(max(1, diag - len_y), min(len_x, diag - 1) + 1, device=D.device)
        j = diag - i

        if bandwidth > 0:
            keep = torch.abs(i - j) <= bandwidth
            i = i[keep]
            j = j[keep]
            if i.numel() == 0:
                continue

        if gamma == 0:
            softmin = torch.minimum(torch.minimum(R[:, i - 1, j - 1], R[:, i - 1, j]), R[:, i, j - 1])
        else:
            previous_costs = torch.stack((R[:, i - 1, j - 1], R[:, i - 1, j], R[:, i, j - 1]))
            softmin = -gamma * torch.logsumexp(-previous_costs / gamma, dim=0)

        R[:, i, j] = D[:, i - 1, j - 1] + softmin

    return R[:, len_x, len_y]


def _soft_dtw_variable_y_no_grad(
    D: torch.Tensor, y_lengths: torch.Tensor, gamma: float, bandwidth: float
) -> torch.Tensor:
    """Compute SoftDTW for a batch where each Y sequence can have a different length."""
    batch_size, len_x, len_y = D.shape
    y_lengths = y_lengths.to(device=D.device, dtype=torch.long)
    if y_lengths.min().item() < 1 or y_lengths.max().item() > len_y:
        raise ValueError("y_lengths entries must be in [1, len_y]")

    R = torch.full((batch_size, len_x + 2, len_y + 2), float("inf"), device=D.device, dtype=D.dtype)
    R[:, 0, 0] = 0

    for diag in range(2, len_x + len_y + 1):
        i = torch.arange(max(1, diag - len_y), min(len_x, diag - 1) + 1, device=D.device)
        j = diag - i

        if bandwidth > 0:
            keep = torch.abs(i - j) <= bandwidth
            i = i[keep]
            j = j[keep]
            if i.numel() == 0:
                continue

        if gamma == 0:
            softmin = torch.minimum(torch.minimum(R[:, i - 1, j - 1], R[:, i - 1, j]), R[:, i, j - 1])
        else:
            previous_costs = torch.stack((R[:, i - 1, j - 1], R[:, i - 1, j], R[:, i, j - 1]))
            softmin = -gamma * torch.logsumexp(-previous_costs / gamma, dim=0)

        values = D[:, i - 1, j - 1] + softmin
        valid_y = j.unsqueeze(0) <= y_lengths.unsqueeze(1)
        R[:, i, j] = torch.where(valid_y, values, torch.full_like(values, float("inf")))

    batch_ids = torch.arange(batch_size, device=D.device)
    return R[batch_ids, len_x, y_lengths]


def _soft_dtw(D: torch.Tensor, gamma: float, bandwidth: float) -> torch.Tensor:
    """Compute batched SoftDTW from a pairwise distance tensor.

    Args:
        D: Pairwise distance tensor of shape ``(batch, len_x, len_y)``.
        gamma: SoftDTW smoothing parameter. Set to 0 to compute hard DTW.
        bandwidth: Optional Sakoe-Chiba bandwidth. Values <= 0 disable the band constraint.
    """
    if torch.is_grad_enabled() and D.requires_grad:
        return _soft_dtw_autograd(D, gamma, bandwidth)
    return _soft_dtw_no_grad(D, gamma, bandwidth)


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

    def forward_with_lengths(self, X, Y, y_lengths):
        """Compute SoftDTW when each Y sequence is padded to a per-sample length."""
        if self.normalize:
            raise ValueError("forward_with_lengths does not support normalize=True")

        if torch.is_grad_enabled() and (X.requires_grad or Y.requires_grad):
            outputs = []
            for batch_id, y_length in enumerate(y_lengths.tolist()):
                outputs.append(self.forward(X[batch_id : batch_id + 1], Y[batch_id : batch_id + 1, : int(y_length)]))
            return torch.cat(outputs)

        D_xy = self.dist_func(X, Y)
        return _soft_dtw_variable_y_no_grad(D_xy, y_lengths, self.gamma, self.bandwidth)
