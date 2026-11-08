# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional forward-only braking exploration and guarded MicroDuck navigation Beta means."""

import torch
from rsl_rl.modules.distribution import BetaDistribution


class _BrakingBetaMean(torch.nn.Module):
    """Keep ordinary Beta means; brake when the forward head rounds to uniform."""

    def __init__(self, forward_braking_only: bool = False) -> None:
        super().__init__()
        self.forward_braking_only = forward_braking_only

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        alpha_raw, beta_raw = torch.unbind(logits, dim=-2)
        alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
        beta = torch.nn.functional.softplus(beta_raw) + 1.0
        denominator = alpha + beta
        mean = 2.0 * (alpha / denominator) - 1.0
        collapsed = (alpha[..., 0] == 1.0) & (beta[..., 0] == 1.0)
        forward = (mean[..., 0] - 1.0) / 2.0 if self.forward_braking_only else mean[..., 0]
        forward = torch.where(collapsed, -torch.ones_like(forward), forward)
        adjusted = torch.cat((forward.unsqueeze(-1), mean[..., 1:]), dim=-1)
        invalid = ~torch.isfinite(logits).all(dim=-1).all(dim=-1) | ~torch.isfinite(denominator).all(dim=-1)
        braking = torch.stack((-torch.ones_like(forward), torch.zeros_like(forward), torch.zeros_like(forward)), dim=-1)
        return torch.where(invalid.unsqueeze(-1), braking, adjusted)


class MicroDuckBrakingBetaDistribution(BetaDistribution):
    """Native Beta exploration with guarded deterministic navigation outputs.

    With ``forward_braking_only=True``, forward samples and ordinary means use
    [-1, 0]; lateral and yaw remain in [-1, 1]. This flag must be retained in the
    actor configuration when loading weights; it is not stored in the state dict.

    A forward Beta whose two concentrations round to1 requests residual -1,
    which targets zero forward speed through the task's existing command adapter.
    Its acceleration limits still apply. With default support, a healthy symmetric
    distribution may return0 normally. This catches numerical collapse,
    not all unfamiliar inputs; it is not a general uncertainty estimator.
    """

    def __init__(
        self,
        output_dim: int,
        action_range: tuple[float, float] = (-1.0, 1.0),
        forward_braking_only: bool = False,
    ) -> None:
        if output_dim != 3 or tuple(action_range) != (-1.0, 1.0):
            raise ValueError("MicroDuck braking requires three navigation residuals in [-1, 1].")
        super().__init__(output_dim, action_range)
        self.forward_braking_only = forward_braking_only
        self._mean_output = _BrakingBetaMean(forward_braking_only)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return self._mean_output(mlp_output)

    def as_deterministic_output_module(self) -> torch.nn.Module:
        return self._mean_output

    def sample(self) -> torch.Tensor:
        """Sample native Beta residuals, optionally restricting forward to braking."""
        samples = super().sample()
        if self.forward_braking_only:
            samples[..., 0] = (samples[..., 0] - 1.0) / 2.0
        return samples

    @property
    def mean(self) -> torch.Tensor:
        """Return ordinary distribution means in the configured residual support."""
        mean = super().mean
        if self.forward_braking_only:
            mean[..., 0] = (mean[..., 0] - 1.0) / 2.0
        return mean

    @property
    def std(self) -> torch.Tensor:
        """Return residual standard deviations after the optional forward scaling."""
        std = super().std
        if self.forward_braking_only:
            std[..., 0] /= 2.0
        return std

    @property
    def entropy(self) -> torch.Tensor:
        """Keep native entropy reporting, accounting for the halved forward range."""
        entropy = super().entropy
        if self.forward_braking_only:
            entropy = entropy - self._log_range_scale
        return entropy

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Evaluate residual density with the optional forward affine Jacobian."""
        if self.forward_braking_only:
            outputs = outputs.clone()
            outputs[..., 0] = 2.0 * outputs[..., 0] + 1.0
            return super().log_prob(outputs) + self._log_range_scale
        return super().log_prob(outputs)
