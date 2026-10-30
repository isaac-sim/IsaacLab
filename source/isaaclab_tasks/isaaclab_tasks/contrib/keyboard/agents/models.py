# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL models for the SO-101 keyboard-typing task."""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, HiddenState
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


class SharedEncoderMLPModel(MLPModel):
    """Encode selected 1-D observation groups before the MLP head."""

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
        """Initialize the encoded-observation MLP model."""
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
        self.encoder_input_dims = []
        for obs_group in self.encoder_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The MLP encoders only support 1D observations, got shape {obs[obs_group].shape} for"
                    f" '{obs_group}'."
                )
            self.encoder_input_dims.append(obs[obs_group].shape[-1])

        encoders = {}
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
            latents.insert(0, super().get_latent(obs))
        return torch.cat(latents, dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update normalization statistics of non-encoded observation groups."""
        if self.obs_groups:
            super().update_normalization(obs)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchSharedEncoderModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
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
        """Replace the critic's encoders before PPO registers optimizer parameters."""
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
        self.encoders = nn.ModuleList([copy.deepcopy(model.encoders[g]) for g in model.encoder_obs_groups])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_raw: torch.Tensor, obs_encoded: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from raw and encoded-group inputs."""
        latents = [self.obs_normalizer(obs_raw)]
        for i, encoder in enumerate(self.encoders):
            latents.append(encoder(obs_encoded[i]))
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
