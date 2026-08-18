# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL models with shared encoders for structured observation groups."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, GaussianDistribution, HiddenState
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from torch.distributions import Normal


class BoundedGaussianDistribution(GaussianDistribution):
    """Version-tolerant Gaussian distribution with an in-place standard-deviation bound.

    The Isaac Lab 3.0 beta 2 image carries RSL-RL 5.0.1, whose base distribution does not accept
    ``std_range``, while current development uses RSL-RL 5.4.1. Keeping the bound in this task-owned
    subclass gives both versions the same policy behavior.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1.0e-6, 1.0e6),
        std_type: str = "scalar",
    ) -> None:
        """Initialize a Gaussian whose learnable scale stays inside ``std_range``."""
        minimum_std, maximum_std = std_range
        if not all(math.isfinite(value) for value in (minimum_std, init_std, maximum_std)) or not (
            0.0 < minimum_std <= init_std <= maximum_std
        ):
            raise ValueError(
                "Expected 0 < minimum_std <= init_std <= maximum_std, got "
                f"std_range={std_range} and init_std={init_std}."
            )

        # Only pass arguments supported by every RSL-RL 5.x release used by Isaac Lab images.
        super().__init__(output_dim, init_std=init_std, std_type=std_type)
        self.std_range = (float(minimum_std), float(maximum_std))
        self.log_std_range = (math.log(minimum_std), math.log(maximum_std))

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Normal distribution after bounding its learnable scale parameter."""
        if self.std_type == "scalar":
            with torch.no_grad():
                self.std_param.clamp_(*self.std_range)
            std = self.std_param
        elif self.std_type == "log":
            with torch.no_grad():
                self.log_std_param.clamp_(*self.log_std_range)
            std = torch.exp(self.log_std_param)
        else:  # Defensive: the RSL-RL base constructor currently rejects this first.
            raise ValueError(f"Unknown standard deviation type: {self.std_type}.")
        self._distribution = Normal(mlp_output, std)


class SharedEncoderMLPModel(MLPModel):
    """Encode selected one-dimensional observation groups before the MLP head.

    Encoded groups bypass the empirical observation normalizer. Their values should therefore be
    explicitly bounded before they reach this model. The remaining observation groups are concatenated
    and normalized by :class:`~rsl_rl.models.MLPModel`.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoder_cfg: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the encoded-observation MLP model.

        Args:
            obs: Batched observation dictionary.
            obs_groups: Observation groups assigned to each model observation set.
            obs_set: Observation set used by this model, such as ``actor`` or ``critic``.
            output_dim: Number of model outputs.
            hidden_dims: Hidden dimensions of the MLP head.
            activation: Activation function of the MLP head.
            obs_normalization: Whether to normalize non-encoded observations.
            distribution_cfg: Optional output-distribution configuration.
            encoder_cfg: Per-group MLP encoder configurations. Each entry contains ``hidden_dims`` and
                ``latent_dim`` and may contain ``activation`` and ``last_activation``.
        """
        if not encoder_cfg:
            raise ValueError("At least one encoder configuration must be provided.")

        active_obs_groups = obs_groups[obs_set]
        encoder_keys = set(encoder_cfg)
        if not encoder_keys.issubset(active_obs_groups):
            invalid_groups = sorted(encoder_keys - set(active_obs_groups))
            raise ValueError(
                f"The encoder observation groups {invalid_groups} are not part of the '{obs_set}' observation groups"
                f" {active_obs_groups}."
            )

        self.encoder_obs_groups = [group for group in active_obs_groups if group in encoder_keys]
        self.encoder_input_dims: list[int] = []
        for obs_group in self.encoder_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The MLP encoders only support 1D observations, got shape {obs[obs_group].shape} for"
                    f" '{obs_group}'."
                )
            self.encoder_input_dims.append(obs[obs_group].shape[-1])

        encoders: dict[str, nn.Module] = {}
        self.encoder_latent_dim = 0
        for obs_group, input_dim in zip(self.encoder_obs_groups, self.encoder_input_dims):
            group_cfg = dict(encoder_cfg[obs_group])
            latent_dim = group_cfg["latent_dim"]
            encoders[obs_group] = MLP(
                input_dim=input_dim,
                output_dim=latent_dim,
                hidden_dims=group_cfg["hidden_dims"],
                activation=group_cfg.get("activation", "elu"),
                last_activation=group_cfg.get("last_activation"),
            )
            self.encoder_latent_dim += latent_dim

        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        self.encoders = nn.ModuleDict(encoders)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent from raw observations and encoded groups."""
        latents = [self.encoders[group](obs[group]) for group in self.encoder_obs_groups]
        if self.obs_groups:
            latents.insert(0, super().get_latent(obs, masks, hidden_state))
        return torch.cat(latents, dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update normalization statistics for non-encoded observation groups."""
        if self.obs_groups:
            super().update_normalization(obs)

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-compatible version of the model."""
        return _TorchSharedEncoderModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX-compatible version of the model."""
        return _OnnxSharedEncoderModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select non-encoded observation groups and compute their total dimension."""
        active_obs_groups = obs_groups[obs_set]
        raw_obs_groups = []
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The MLP model only supports 1D observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            if obs_group not in self.encoder_obs_groups:
                raw_obs_groups.append(obs_group)
                obs_dim += obs[obs_group].shape[-1]
        return raw_obs_groups, obs_dim

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim + self.encoder_latent_dim


class SharedEncoderPPO(PPO):
    """Share the actor's observation encoders with the critic."""

    def __init__(self, actor: MLPModel, critic: MLPModel, storage: RolloutStorage, **kwargs: Any) -> None:
        """Replace the critic encoders before PPO registers optimizer parameters."""
        if not isinstance(actor, SharedEncoderMLPModel) or not isinstance(critic, SharedEncoderMLPModel):
            raise TypeError("SharedEncoderPPO requires SharedEncoderMLPModel actor and critic models.")
        if (
            actor.encoder_obs_groups != critic.encoder_obs_groups
            or actor.encoder_latent_dim != critic.encoder_latent_dim
        ):
            raise ValueError("The actor and critic encoder configurations must match.")

        # The actor owns the shared modules so optimizer, checkpoint, and gradient traversal see them once.
        del critic.encoders
        object.__setattr__(critic, "encoders", actor.encoders)
        super().__init__(actor, critic, storage, **kwargs)


class _TorchSharedEncoderModel(nn.Module):
    """Exportable shared-encoder model for TorchScript."""

    def __init__(self, model: SharedEncoderMLPModel) -> None:
        """Create a TorchScript-compatible model copy."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoders = nn.ModuleList([copy.deepcopy(model.encoders[group]) for group in model.encoder_obs_groups])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_raw: torch.Tensor, obs_encoded: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from raw and encoded-group inputs."""
        latents = [self.obs_normalizer(obs_raw)]
        for index, encoder in enumerate(self.encoders):
            latents.append(encoder(obs_encoded[index]))
        return self.deterministic_output(self.mlp(torch.cat(latents, dim=-1)))

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state."""
        pass


class _OnnxSharedEncoderModel(_TorchSharedEncoderModel):
    """Exportable shared-encoder model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: SharedEncoderMLPModel, verbose: bool) -> None:
        """Create an ONNX-compatible model copy."""
        super().__init__(model)
        self.verbose = verbose
        self.encoder_obs_groups = list(model.encoder_obs_groups)
        self.encoder_input_dims = list(model.encoder_input_dims)
        self.obs_dim_raw = model.obs_dim

    def forward(self, obs: torch.Tensor, *obs_encoded: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        return super().forward(obs, list(obs_encoded))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        dummy_raw = torch.zeros(1, self.obs_dim_raw)
        dummy_encoded = [torch.zeros(1, dim) for dim in self.encoder_input_dims]
        return (dummy_raw, *dummy_encoded)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", *self.encoder_obs_groups]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
