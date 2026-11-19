# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL models shared by Factory task compositions."""

from __future__ import annotations

import copy
import math
from dataclasses import MISSING
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, resolve_nn_activation, unpad_trajectories
from tensordict import TensorDict
from torch.utils.checkpoint import checkpoint

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg


@configclass
class SimBaModelCfg:
    """SimBa model with optional encoders keyed by observation group."""

    @configclass
    class EncoderCfg:
        class_name: str = MISSING
        output_dim: int = MISSING

    @configclass
    class MLPEncoderCfg(EncoderCfg):
        class_name: str = "isaaclab_tasks.contrib.nist.config.agents.models:MLPEncoder"
        hidden_dims: list[int] = MISSING
        activation: str = MISSING
        last_activation: str | None = None

    class_name: str = "isaaclab_tasks.contrib.nist.config.agents.models:SimBaModel"
    hidden_dim: int = MISSING
    num_blocks: int = 2
    expansion_factor: int = 4
    activation: str = "relu"
    norm: bool = True
    obs_normalization: bool = False
    encoder_normalization: bool = False
    activation_checkpointing: bool = False
    encoder_cfg: dict[str, EncoderCfg] | None = None
    distribution_cfg: RslRlMLPModelCfg.DistributionCfg | None = None


class MLPEncoder(nn.Module):
    """Flatten an observation group and encode it with an MLP."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
        last_activation: str | None = None,
    ) -> None:
        super().__init__()
        if not input_shape:
            raise ValueError("MLPEncoder requires a non-empty input shape.")
        self.feature_rank = len(input_shape)
        self.output_dim = output_dim
        self.mlp = MLP(math.prod(input_shape), output_dim, hidden_dims, activation, last_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x.flatten(start_dim=-self.feature_rank))


class SimBaBlock(nn.Module):
    """Pre-normalized residual block used by SimBa."""

    def __init__(self, hidden_dim: int, expansion_factor: int, activation: str, norm: bool) -> None:
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_dim) if norm else nn.Identity()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, expansion_factor * hidden_dim),
            resolve_nn_activation(activation),
            nn.Linear(expansion_factor * hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(self.pre_norm(x))


class SimBaNetwork(nn.Sequential):
    """Residual MLP backbone from SimBa."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_blocks: int = 2,
        expansion_factor: int = 4,
        activation: str = "relu",
        norm: bool = True,
    ) -> None:
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
        layers.extend(SimBaBlock(hidden_dim, expansion_factor, activation, norm) for _ in range(num_blocks))
        layers.extend((nn.LayerNorm(hidden_dim) if norm else nn.Identity(), nn.Linear(hidden_dim, output_dim)))
        super().__init__(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_in = nn.init._calculate_correct_fan(module.weight, "fan_in")
                nn.init.uniform_(module.weight, -1.0 / math.sqrt(fan_in), 1.0 / math.sqrt(fan_in))
                nn.init.zeros_(module.bias)

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the network output and its pre-normalization residual features."""
        feature_index = len(self) - 2
        features = x
        for index, module in enumerate(self):
            if index == feature_index:
                features = x
            x = module(x)
        return x, features


class SimBaModel(MLPModel):
    """Compose observation-group encoders with a SimBa policy or value head.

    Encoder classes receive ``input_shape`` plus their configuration fields and must return a tensor whose last
    dimension is ``output_dim``. Groups without an encoder pass directly into the SimBa head.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        expansion_factor: int = 4,
        activation: str = "relu",
        norm: bool = True,
        obs_normalization: bool = False,
        encoder_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoder_cfg: dict[str, dict[str, Any]] | None = None,
        activation_checkpointing: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.activation_checkpointing = activation_checkpointing
        self._encoder_cfg = {group: dict(group_cfg) for group, group_cfg in (encoder_cfg or {}).items()}
        self._encoder_output_dims = {
            group: int(group_cfg["output_dim"]) for group, group_cfg in self._encoder_cfg.items()
        }
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)

        self.obs_normalization = obs_normalization
        self.obs_normalizer = (
            EmpiricalNormalization(self.obs_dim) if obs_normalization and self.obs_dim else nn.Identity()
        )

        encoders: dict[str, nn.Module] = {}
        for group in self.encoded_obs_groups:
            group_cfg = dict(self._encoder_cfg[group])
            encoder_class = resolve_callable(group_cfg.pop("class_name"))
            encoder = encoder_class(input_shape=self.encoder_input_shapes[group], **group_cfg)
            if not isinstance(encoder, nn.Module):
                raise TypeError(f"Encoder for observation group {group!r} must be a torch module.")
            encoders[group] = encoder
        self.encoders = nn.ModuleDict(encoders)

        self.encoder_normalization = encoder_normalization
        self.encoder_normalizers = nn.ModuleDict(
            {
                group: EmpiricalNormalization(self.encoder_input_shapes[group])
                if encoder_normalization
                else nn.Identity()
                for group in self.encoded_obs_groups
            }
        )

        if distribution_cfg is not None:
            distribution_cfg = dict(distribution_cfg)
            distribution_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = distribution_class(output_dim, **distribution_cfg)
            head_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            head_output_dim = output_dim

        self.mlp = SimBaNetwork(
            self.obs_dim + sum(self._encoder_output_dims.values()),
            head_output_dim,
            hidden_dim,
            num_blocks,
            expansion_factor,
            activation,
            norm,
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Run the model, optionally sampling its output distribution."""
        output, _ = self.forward_with_features(obs, masks, hidden_state, stochastic_output)
        return output

    def forward_with_features(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the model output and the backbone's pre-normalization features."""
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        if self.activation_checkpointing and torch.is_grad_enabled():
            raw_output, features = checkpoint(self._forward_network, obs, masks, hidden_state, use_reentrant=False)
        else:
            raw_output, features = self._forward_network(obs, masks, hidden_state)
        if self.distribution is None:
            return raw_output, features
        if stochastic_output:
            self.distribution.update(raw_output)
            return self.distribution.sample(), features
        return self.distribution.deterministic_output(raw_output), features

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Concatenate normalized passthrough observations and encoded observations."""
        parts = []
        if self.obs_groups:
            parts.append(self.obs_normalizer(torch.cat([obs[group] for group in self.obs_groups], dim=-1)))
        for group in self.encoded_obs_groups:
            shape = self.encoder_input_shapes[group]
            value = self.encoder_normalizers[group](obs[group])
            encoded = self.encoders[group](value.reshape(-1, *shape))
            parts.append(encoded.reshape(*value.shape[: -len(shape)], encoded.shape[-1]))
        if not parts:
            raise RuntimeError("SimBaModel requires at least one observation group.")
        return torch.cat(parts, dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update passthrough and encoder-input normalization statistics."""
        if self.obs_normalization and self.obs_groups:
            self.obs_normalizer.update(torch.cat([obs[group] for group in self.obs_groups], dim=-1))  # type: ignore
        if self.encoder_normalization:
            for group in self.encoded_obs_groups:
                self.encoder_normalizers[group].update(obs[group])  # type: ignore

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-compatible inference model."""
        return _TorchSimBaModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX-compatible inference model."""
        return _OnnxSimBaModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_groups = obs_groups[obs_set]
        missing = set(self._encoder_cfg).difference(active_groups)
        if missing:
            raise ValueError(f"Encoder groups are not present in observation set {obs_set!r}: {sorted(missing)}")

        passthrough_groups = []
        passthrough_dim = 0
        self.encoded_obs_groups: list[str] = []
        self.encoder_input_shapes: dict[str, tuple[int, ...]] = {}
        for group in active_groups:
            shape = obs[group].shape
            if group in self._encoder_cfg:
                self.encoded_obs_groups.append(group)
                self.encoder_input_shapes[group] = tuple(int(dim) for dim in shape[1:])
            else:
                if len(shape) != 2:
                    raise ValueError(f"Observation group {group!r} must have an encoder or be flattened.")
                passthrough_groups.append(group)
                passthrough_dim += int(shape[-1])
        return passthrough_groups, passthrough_dim

    def _forward_network(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mlp.forward_with_features(self.get_latent(obs, masks, hidden_state))  # type: ignore


class _TorchSimBaModel(nn.Module):
    def __init__(self, model: SimBaModel) -> None:
        super().__init__()
        if len(model.encoded_obs_groups) != 1:
            raise NotImplementedError("SimBa export currently requires exactly one encoded observation group.")
        group = model.encoded_obs_groups[0]
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder_normalizer = copy.deepcopy(model.encoder_normalizers[group])
        self.encoder = copy.deepcopy(model.encoders[group])
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module() if model.distribution is not None else nn.Identity()
        )

    def forward(self, state: torch.Tensor, encoded_obs: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference from passthrough and encoded observations."""
        state = self.obs_normalizer(state)
        encoded_obs = self.encoder_normalizer(encoded_obs)
        output = self.mlp(torch.cat((state, self.encoder(encoded_obs)), dim=-1))
        return self.deterministic_output(output)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent state (no-op for this feed-forward model)."""
        pass


class _OnnxSimBaModel(_TorchSimBaModel):
    def __init__(self, model: SimBaModel, verbose: bool) -> None:
        super().__init__(model)
        group = model.encoded_obs_groups[0]
        self.verbose = verbose
        self.state_dim = model.obs_dim
        self.encoded_input_shape = model.encoder_input_shapes[group]
        self.encoded_obs_group = group

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return representative passthrough and encoded inputs."""
        return torch.zeros(1, self.state_dim), torch.zeros(1, *self.encoded_input_shape)

    @property
    def input_names(self) -> list[str]:
        """Names of the ONNX inputs."""
        return ["obs", self.encoded_obs_group]

    @property
    def output_names(self) -> list[str]:
        """Names of the ONNX outputs."""
        return ["actions"]
