# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-headed RSL-RL models for contributed heterogeneous manipulation."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, resolve_nn_activation, unpad_trajectories
from tensordict import TensorDict
from torch.distributions import Normal


class TaskHeadedGaussianDistribution(Distribution):
    """Diagonal Gaussian distribution routed over contiguous task action heads."""

    def __init__(
        self,
        output_dim: int,
        task_action_dims: Sequence[int],
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        std_type: str = "scalar",
        learn_std: bool = True,
    ) -> None:
        """Initialize task-specific Gaussian standard deviations.

        Args:
            output_dim: Fixed global action dimension.
            task_action_dims: Action dimensions of the contiguous task heads.
            init_std: Initial standard deviation for every active action dimension.
            std_range: Inclusive standard-deviation clamp range.
            std_type: Standard-deviation parameterization, either ``"scalar"`` or ``"log"``.
            learn_std: Whether the task-specific standard deviations are learnable.

        Raises:
            ValueError: If the task dimensions or standard-deviation configuration are invalid.
        """
        super().__init__(output_dim)
        self.task_action_dims = tuple(task_action_dims)
        if not self.task_action_dims or any(dim <= 0 for dim in self.task_action_dims):
            raise ValueError(f"Expected positive task action dimensions, got {self.task_action_dims}.")
        if sum(self.task_action_dims) != output_dim:
            raise ValueError(
                f"Task action dimensions {self.task_action_dims} sum to {sum(self.task_action_dims)},"
                f" expected {output_dim}."
            )
        if init_std <= 0.0:
            raise ValueError(f"Expected a positive initial standard deviation, got {init_std}.")
        if std_range[0] <= 0.0 or std_range[0] > std_range[1]:
            raise ValueError(f"Expected a positive ordered standard-deviation range, got {std_range}.")
        if std_type not in ("scalar", "log"):
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        self.std_type = std_type
        self.std_range = std_range
        self.log_std_range = (math.log(std_range[0]), math.log(std_range[1]))
        initial_value = init_std if std_type == "scalar" else math.log(init_std)
        self._std_params = nn.ParameterList(
            [nn.Parameter(torch.full((dim,), initial_value), requires_grad=learn_std) for dim in self.task_action_dims]
        )

        action_masks = torch.zeros(len(self.task_action_dims), output_dim)
        start = 0
        for task_id, action_dim in enumerate(self.task_action_dims):
            action_masks[task_id, start : start + action_dim] = 1.0
            start += action_dim
        self.register_buffer("_task_action_masks", action_masks)

        self._distribution: Normal | None = None
        self._action_mask: torch.Tensor | None = None
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor, task_encoding: torch.Tensor | None = None) -> None:
        """Update the routed distribution from global means and one-hot task identities."""
        if task_encoding is None:
            raise ValueError("TaskHeadedGaussianDistribution requires a task encoding when it is updated.")
        if task_encoding.shape[-1] != len(self.task_action_dims):
            raise ValueError(
                f"Expected {len(self.task_action_dims)} task encoding dimensions, got {task_encoding.shape[-1]}."
            )

        action_mask = task_encoding.to(dtype=mlp_output.dtype) @ self._task_action_masks.to(dtype=mlp_output.dtype)
        std = self._global_std().expand_as(mlp_output)
        # Unit standard deviation keeps the inactive Normal parameters numerically valid; all probability terms mask it.
        distribution_std = std * action_mask + (1.0 - action_mask)
        self._distribution = Normal(mlp_output, distribution_std)
        self._action_mask = action_mask

    def sample(self) -> torch.Tensor:
        """Sample the active task head and return zero in every inactive global dimension."""
        return self._require_distribution().sample() * self._require_action_mask()

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return the routed means produced by the task-headed model."""
        return mlp_output

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an identity module because the model already routes its deterministic output."""
        return nn.Identity()

    @property
    def input_dim(self) -> int:
        """Return the fixed global mean dimension."""
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        """Return the current routed global action mean."""
        return self._require_distribution().mean

    @property
    def std(self) -> torch.Tensor:
        """Return the concatenated standard deviations of every task head."""
        return self._global_std()

    @property
    def entropy(self) -> torch.Tensor:
        """Return entropy summed only over the selected task head."""
        return (self._require_distribution().entropy() * self._require_action_mask()).sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return mean, numerically valid standard deviation, and the active action mask."""
        distribution = self._require_distribution()
        return distribution.mean, distribution.stddev, self._require_action_mask()

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Return log-probability summed only over the selected task head."""
        return (self._require_distribution().log_prob(outputs) * self._require_action_mask()).sum(dim=-1)

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Return KL divergence summed only over the task head stored with the old policy."""
        old_mean, old_std, old_action_mask = old_params
        new_mean, new_std, _ = new_params
        divergence = torch.distributions.kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std))
        return (divergence * old_action_mask).sum(dim=-1)

    def _global_std(self) -> torch.Tensor:
        """Concatenate the independently parameterized task-head standard deviations."""
        if self.std_type == "scalar":
            stds = [param.clamp(self.std_range[0], self.std_range[1]) for param in self._std_params]
        else:
            stds = [torch.exp(param.clamp(self.log_std_range[0], self.log_std_range[1])) for param in self._std_params]
        return torch.cat(stds)

    def _require_distribution(self) -> Normal:
        """Return the current distribution or reject access before an update."""
        if self._distribution is None:
            raise RuntimeError("The task-headed Gaussian distribution has not been updated.")
        return self._distribution

    def _require_action_mask(self) -> torch.Tensor:
        """Return the current action mask or reject access before an update."""
        if self._action_mask is None:
            raise RuntimeError("The task-headed Gaussian distribution has not been updated.")
        return self._action_mask


class TaskHeadedMLPModel(MLPModel):
    """Shared observation backbone with task-specific Gaussian action heads."""

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
        task_action_dims: tuple[int, ...] | list[int] = (8, 8, 6),
        task_encoding_slice: tuple[int, int] | list[int] = (0, 3),
    ) -> None:
        """Initialize the shared backbone and contiguous task action heads.

        Args:
            obs: Observation dictionary used to infer model input dimensions.
            obs_groups: Observation groups assigned to each model set.
            obs_set: Observation set consumed by this model.
            output_dim: Fixed global action dimension.
            hidden_dims: Shared-backbone layer dimensions.
            activation: Shared-backbone activation function.
            obs_normalization: Whether to normalize the shared observation input.
            distribution_cfg: Task-headed Gaussian distribution configuration.
            task_action_dims: Action dimensions ordered by task identity.
            task_encoding_slice: Half-open slice of the raw model input containing the task one-hot.

        Raises:
            TypeError: If the configured distribution is not task-headed.
            ValueError: If dimensions or the initial task encoding are invalid.
        """
        if not hidden_dims or any(dim <= 0 for dim in hidden_dims):
            raise ValueError(f"Expected positive shared-backbone dimensions, got {hidden_dims}.")

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=None,
        )

        self.task_action_dims = tuple(task_action_dims)
        if not self.task_action_dims or any(dim <= 0 for dim in self.task_action_dims):
            raise ValueError(f"Expected positive task action dimensions, got {self.task_action_dims}.")
        if sum(self.task_action_dims) != output_dim:
            raise ValueError(
                f"Task action dimensions {self.task_action_dims} sum to {sum(self.task_action_dims)},"
                f" expected {output_dim}."
            )

        if len(task_encoding_slice) != 2:
            raise ValueError(f"Expected a two-element task encoding slice, got {task_encoding_slice}.")
        self.task_encoding_slice = (int(task_encoding_slice[0]), int(task_encoding_slice[1]))
        encoding_start, encoding_end = self.task_encoding_slice
        if encoding_start < 0 or encoding_end > self.obs_dim or encoding_end <= encoding_start:
            raise ValueError(
                f"Task encoding slice {self.task_encoding_slice} is outside the {self.obs_dim}D model input."
            )
        if encoding_end - encoding_start != len(self.task_action_dims):
            raise ValueError(
                f"Task encoding slice {self.task_encoding_slice} has dimension {encoding_end - encoding_start},"
                f" expected {len(self.task_action_dims)}."
            )
        self._validate_task_encoding(self._task_encoding(obs))

        # Replace the standard monolithic MLP with an activated shared backbone and task-local linear heads.
        del self.mlp
        self.backbone = _make_backbone(self.obs_dim, hidden_dims, activation)
        self.action_heads = nn.ModuleList([nn.Linear(hidden_dims[-1], dim) for dim in self.task_action_dims])

        if distribution_cfg is None:
            raise ValueError("TaskHeadedMLPModel requires a task-headed distribution configuration.")
        distribution_cfg = dict(distribution_cfg)
        distribution_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
        if not issubclass(distribution_class, TaskHeadedGaussianDistribution):
            raise TypeError(
                f"TaskHeadedMLPModel requires a TaskHeadedGaussianDistribution, got {distribution_class.__name__}."
            )
        self.distribution = distribution_class(
            output_dim,
            task_action_dims=self.task_action_dims,
            **distribution_cfg,
        )

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Route shared features through the task head selected by the raw task one-hot."""
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.backbone(self.get_latent(obs, masks, hidden_state))
        task_ids = self._task_encoding(obs).argmax(dim=-1)
        task_encoding = F.one_hot(task_ids, num_classes=len(self.task_action_dims)).to(dtype=latent.dtype)
        output = torch.cat(
            [
                head(latent) * task_encoding[..., task_id : task_id + 1]
                for task_id, head in enumerate(self.action_heads)
            ],
            dim=-1,
        )
        if stochastic_output:
            self.distribution.update(output, task_encoding)
            return self.distribution.sample()
        return output

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-compatible deterministic task-headed policy."""
        return _TorchTaskHeadedMLPModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX-compatible deterministic task-headed policy."""
        return _OnnxTaskHeadedMLPModel(self, verbose)

    def _task_encoding(self, obs: TensorDict) -> torch.Tensor:
        """Extract the raw task one-hot before observation normalization."""
        model_input = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        return model_input[..., self.task_encoding_slice[0] : self.task_encoding_slice[1]]

    def _validate_task_encoding(self, task_encoding: torch.Tensor) -> None:
        """Validate one-hot task identities once when the model is constructed."""
        is_binary = torch.logical_or(task_encoding == 0.0, task_encoding == 1.0).all()
        has_one_task = (task_encoding.sum(dim=-1) == 1.0).all()
        if not bool(is_binary and has_one_task):
            raise ValueError("Task encoding observations must be one-hot with exactly one active task per sample.")


class TaskHeadedValueModel(MLPModel):
    """Shared observation backbone with task-specific scalar value heads."""

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
        task_head_count: int = 3,
        task_encoding_slice: tuple[int, int] | list[int] = (0, 3),
    ) -> None:
        """Initialize a shared value backbone and one scalar head per task.

        Args:
            obs: Observation dictionary used to infer model input dimensions.
            obs_groups: Observation groups assigned to each model set.
            obs_set: Observation set consumed by this model.
            output_dim: Critic output dimension, which must be one.
            hidden_dims: Shared-backbone layer dimensions.
            activation: Shared-backbone activation function.
            obs_normalization: Whether to normalize the shared observation input.
            distribution_cfg: Output distribution configuration, which must be ``None`` for a critic.
            task_head_count: Number of task-specific scalar value heads.
            task_encoding_slice: Half-open slice of the raw model input containing the task one-hot.

        Raises:
            ValueError: If the value, backbone, task-head, or task-encoding configuration is invalid.
        """
        if output_dim != 1:
            raise ValueError(f"TaskHeadedValueModel requires a scalar output, got {output_dim}.")
        if not hidden_dims or any(dim <= 0 for dim in hidden_dims):
            raise ValueError(f"Expected positive shared-backbone dimensions, got {hidden_dims}.")
        if task_head_count <= 0:
            raise ValueError(f"Expected a positive task head count, got {task_head_count}.")
        if distribution_cfg is not None:
            raise ValueError("TaskHeadedValueModel does not support an output distribution.")

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=None,
        )

        if len(task_encoding_slice) != 2:
            raise ValueError(f"Expected a two-element task encoding slice, got {task_encoding_slice}.")
        self.task_encoding_slice = (int(task_encoding_slice[0]), int(task_encoding_slice[1]))
        encoding_start, encoding_end = self.task_encoding_slice
        if encoding_start < 0 or encoding_end > self.obs_dim or encoding_end <= encoding_start:
            raise ValueError(
                f"Task encoding slice {self.task_encoding_slice} is outside the {self.obs_dim}D model input."
            )
        if encoding_end - encoding_start != task_head_count:
            raise ValueError(
                f"Task encoding slice {self.task_encoding_slice} has dimension {encoding_end - encoding_start},"
                f" expected {task_head_count}."
            )
        self.task_head_count = task_head_count
        self._validate_task_encoding(self._task_encoding(obs))

        del self.mlp
        self.backbone = _make_backbone(self.obs_dim, hidden_dims, activation)
        self.value_heads = nn.ModuleList([nn.Linear(hidden_dims[-1], 1) for _ in range(task_head_count)])

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Return the scalar value from the head selected by the raw task one-hot."""
        del stochastic_output
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.backbone(self.get_latent(obs, masks, hidden_state))
        task_ids = self._task_encoding(obs).argmax(dim=-1, keepdim=True)
        task_values = torch.cat([head(latent) for head in self.value_heads], dim=-1)
        return torch.gather(task_values, dim=-1, index=task_ids)

    def _task_encoding(self, obs: TensorDict) -> torch.Tensor:
        """Extract the raw task one-hot before observation normalization."""
        model_input = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        return model_input[..., self.task_encoding_slice[0] : self.task_encoding_slice[1]]

    def _validate_task_encoding(self, task_encoding: torch.Tensor) -> None:
        """Validate one-hot task identities once when the model is constructed."""
        is_binary = torch.logical_or(task_encoding == 0.0, task_encoding == 1.0).all()
        has_one_task = (task_encoding.sum(dim=-1) == 1.0).all()
        if not bool(is_binary and has_one_task):
            raise ValueError("Task encoding observations must be one-hot with exactly one active task per sample.")


class _TorchTaskHeadedMLPModel(nn.Module):
    """TorchScript-compatible deterministic task-headed policy."""

    def __init__(self, model: TaskHeadedMLPModel) -> None:
        """Copy inference modules and task routing metadata from a trained model."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.backbone = copy.deepcopy(model.backbone)
        self.action_heads = copy.deepcopy(model.action_heads)
        self.task_encoding_start = model.task_encoding_slice[0]
        self.task_encoding_end = model.task_encoding_slice[1]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return a fixed-width action containing only the selected task head."""
        task_ids = obs[..., self.task_encoding_start : self.task_encoding_end].argmax(dim=-1)
        latent = self.backbone(self.obs_normalizer(obs))
        outputs: list[torch.Tensor] = []
        for task_id, head in enumerate(self.action_heads):
            selected = (task_ids == task_id).unsqueeze(-1).to(dtype=latent.dtype)
            outputs.append(head(latent) * selected)
        return torch.cat(outputs, dim=-1)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for the feed-forward model)."""
        pass


class _OnnxTaskHeadedMLPModel(_TorchTaskHeadedMLPModel):
    """ONNX-compatible deterministic task-headed policy."""

    is_recurrent: bool = False

    def __init__(self, model: TaskHeadedMLPModel, verbose: bool) -> None:
        """Copy a task-headed model and record ONNX export metadata."""
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return a representative task-valid observation for ONNX tracing."""
        dummy_obs = torch.zeros(1, self.input_size)
        dummy_obs[0, self.task_encoding_start] = 1.0
        return (dummy_obs,)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]


def _make_backbone(input_dim: int, hidden_dims: Sequence[int], activation: str) -> nn.Sequential:
    """Create an activated MLP whose final hidden layer is shared by every task head."""
    layers: list[nn.Module] = []
    layer_input_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(layer_input_dim, hidden_dim))
        layers.append(resolve_nn_activation(activation))
        layer_input_dim = hidden_dim
    return nn.Sequential(*layers)
