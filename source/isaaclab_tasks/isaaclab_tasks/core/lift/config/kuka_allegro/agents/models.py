# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom RSL-RL models for the lift camera tasks."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN, HiddenState
from tensordict import TensorDict


class SpatialSoftmax(nn.Module):
    """Spatial soft-argmax over a convolutional feature map.

    Each channel's map is turned into a probability distribution over pixel locations and reduced
    to the expected ``(x, y)`` coordinate of that distribution, so a ``[C, H, W]`` map becomes
    ``2 * C`` numbers. Coordinates are normalized to ``[-1, 1]``.

    Unlike average pooling, this keeps *where* a feature fired while still collapsing the map,
    which is what a policy that must localize an object from pixels needs. The temperature is
    learned per channel: low temperature sharpens the distribution toward a single peak, high
    temperature averages over the map.
    """

    def __init__(self, num_channels: int, height: int, width: int, init_temperature: float = 1.0):
        """Initialize the spatial soft-argmax.

        Args:
            num_channels: Channels in the input feature map.
            height: Height of the input feature map.
            width: Width of the input feature map.
            init_temperature: Initial softmax temperature. Defaults to ``1.0``.
        """
        super().__init__()
        self.num_channels = num_channels
        self.height = height
        self.width = width
        pos_y, pos_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height), torch.linspace(-1.0, 1.0, width), indexing="ij"
        )
        self.register_buffer("pos_x", pos_x.reshape(1, -1))
        self.register_buffer("pos_y", pos_y.reshape(1, -1))
        self.log_temperature = nn.Parameter(
            torch.full((num_channels,), float(torch.log(torch.tensor(init_temperature))))
        )

    @property
    def output_dim(self) -> int:
        """Number of outputs, two coordinates per channel."""
        return 2 * self.num_channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Reduce a ``[batch, C, H, W]`` feature map to ``[batch, 2 * C]`` keypoint coordinates."""
        batch = features.shape[0]
        temperature = self.log_temperature.exp().clamp(min=1e-3).view(1, self.num_channels, 1)
        logits = features.reshape(batch, self.num_channels, -1) / temperature
        weights = torch.softmax(logits, dim=-1)
        expected_x = (weights * self.pos_x).sum(dim=-1)
        expected_y = (weights * self.pos_y).sum(dim=-1)
        return torch.cat([expected_x, expected_y], dim=-1)


class SpatialSoftmaxCNNModel(MLPModel):
    """CNN encoders reduced by spatial soft-argmax, concatenated with 1D groups into an MLP.

    Mirrors :class:`~rsl_rl.models.cnn_model.CNNModel` but replaces the flattened feature map with
    keypoint coordinates, which shrinks the latent by roughly an order of magnitude while keeping
    the spatial information a pixels-only policy depends on.
    """

    is_recurrent: bool = False

    # splitting the observation groups into 1D and 2D sets is identical to the flattened model
    _get_obs_dim = CNNModel._get_obs_dim

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
        cnn_cfg: dict | None = None,
        init_temperature: float = 1.0,
    ) -> None:
        """Initialize the spatial-softmax CNN model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP head.
            activation: Activation function of the CNN and MLP.
            obs_normalization: Whether to normalize the 1D observations.
            distribution_cfg: Configuration dictionary for the output distribution.
            cnn_cfg: Configuration of the CNN encoder(s); ``flatten`` and ``global_pool`` are
                forced off because the soft-argmax consumes the raw feature map.
            init_temperature: Initial softmax temperature of the keypoint layer. Defaults to ``1.0``.
        """
        if cnn_cfg is None:
            raise ValueError("CNN configurations must be provided.")
        # resolve the 2D groups before the parent sizes the MLP through _get_latent_dim
        self._get_obs_dim(obs, obs_groups, obs_set)

        cnn_kwargs = dict(cnn_cfg)
        # global pooling collapses the map to a single cell, which is exactly the spatial
        # information the soft-argmax exists to read; the two cannot both apply
        if cnn_kwargs.get("global_pool", "none") != "none":
            raise ValueError(
                f"global_pool={cnn_kwargs['global_pool']!r} discards the feature map's spatial layout, which the"
                " spatial soft-argmax reads. Set global_pool='none' or use the flattened CNN model."
            )
        # the soft-argmax consumes the map itself, so the encoder must not flatten it
        cnn_kwargs.update(flatten=False, global_pool="none")
        cnns: dict[str, CNN] = {}
        softmaxes: dict[str, SpatialSoftmax] = {}
        self.keypoint_dim = 0
        for idx, obs_group in enumerate(self.obs_groups_2d):
            cnn = CNN(input_dim=self.obs_dims_2d[idx], input_channels=self.obs_channels_2d[idx], **cnn_kwargs)
            channels = cnn.output_channels
            if channels is None:
                raise ValueError("The CNN output must keep its channel dimension for the spatial soft-argmax.")
            height, width = cnn.output_dim
            softmax = SpatialSoftmax(int(channels), int(height), int(width), init_temperature)
            cnns[obs_group] = cnn
            softmaxes[obs_group] = softmax
            self.keypoint_dim += softmax.output_dim

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
        self.cnns = nn.ModuleDict(cnns)
        self.softmaxes = nn.ModuleDict(softmaxes)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent from keypoint coordinates and normalized 1D groups."""
        latent_2d = torch.cat(
            [self.softmaxes[group](self.cnns[group](obs[group])) for group in self.obs_groups_2d], dim=-1
        )
        if not self.obs_groups:
            return latent_2d
        return torch.cat([MLPModel.get_latent(self, obs), latent_2d], dim=-1)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchSpatialSoftmaxModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxSpatialSoftmaxModel(self, verbose)

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim + self.keypoint_dim


class _TorchSpatialSoftmaxModel(nn.Module):
    """Exportable spatial-softmax model for JIT."""

    def __init__(self, model: SpatialSoftmaxCNNModel) -> None:
        """Create a TorchScript-friendly copy of a SpatialSoftmaxCNNModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        # one Sequential per 2D group: TorchScript cannot index a second ModuleList by variable
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(copy.deepcopy(model.cnns[g]), copy.deepcopy(model.softmaxes[g]))
                for g in model.obs_groups_2d
            ]
        )
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from separated 1D and 2D inputs."""
        latents = [self.obs_normalizer(obs_1d)]
        for i, encoder in enumerate(self.encoders):  # obs_2d order matches obs_groups_2d
            latents.append(encoder(obs_2d[i]))
        return self.deterministic_output(self.mlp(torch.cat(latents, dim=-1)))

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for feed-forward exports)."""
        pass


class _OnnxSpatialSoftmaxModel(_TorchSpatialSoftmaxModel):
    """Exportable spatial-softmax model for ONNX.

    Same modules as the JIT export; ONNX needs the 2D inputs as separate positional arguments
    rather than a list, plus the input/output naming used during tracing.
    """

    def __init__(self, model: SpatialSoftmaxCNNModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a SpatialSoftmaxCNNModel."""
        super().__init__(model)
        self.verbose = verbose
        self.obs_groups_2d = model.obs_groups_2d
        self.obs_dims_2d = model.obs_dims_2d
        self.obs_channels_2d = model.obs_channels_2d
        self.obs_dim_1d = model.obs_dim

    def forward(self, obs_1d: torch.Tensor, *obs_2d: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        return super().forward(obs_1d, list(obs_2d))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        dummy_1d = torch.zeros(1, self.obs_dim_1d)
        dummy_2d = []
        for i in range(len(self.obs_groups_2d)):
            h, w = self.obs_dims_2d[i]
            dummy_2d.append(torch.zeros(1, self.obs_channels_2d[i], h, w))
        return (dummy_1d, *dummy_2d)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", *self.obs_groups_2d]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
