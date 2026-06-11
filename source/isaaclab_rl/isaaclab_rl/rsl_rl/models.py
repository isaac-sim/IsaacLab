# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL neural models customized for Isaac Lab."""

from __future__ import annotations

import copy

import torch
from rsl_rl.models.cnn_model import CNNModel as _CNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict
from torch import nn


class CNNModel(_CNNModel):
    """CNN model that supports pure image-only observations.

    The rsl_rl CNN model does not support image-only observations as it calls
    :meth:`get_latent` without checking whether the observation groups are empty.

    The same applies to the export wrappers returned by :meth:`as_jit` and :meth:`as_onnx`:
    the rsl_rl wrappers always expect a 1D observation input, which for image-only models
    becomes a mandatory zero-width ``obs`` tensor that deployment runtimes have to feed.
    For image-only models, this class instead returns export wrappers that only take the
    2D observation groups as inputs.
    """

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        latent_cnn = torch.cat([self.cnns[group](obs[group]) for group in self.obs_groups_2d], dim=-1)
        if not self.obs_groups:
            return latent_cnn
        latent_1d = MLPModel.get_latent(self, obs, masks, hidden_state)
        return torch.cat([latent_1d, latent_cnn], dim=-1)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        if not self.obs_groups:
            return _TorchImageOnlyCNNModel(self)
        return super().as_jit()

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        if not self.obs_groups:
            return _OnnxImageOnlyCNNModel(self, verbose)
        return super().as_onnx(verbose)


class _TorchImageOnlyCNNModel(nn.Module):
    """Exportable image-only CNN model for JIT.

    Unlike ``rsl_rl``'s exportable CNN model, the forward pass only takes the 2D observation
    groups as input, without a placeholder for the (empty) 1D observations.
    """

    def __init__(self, model: CNNModel):
        super().__init__()
        # Convert ModuleDict to ModuleList for ordered iteration
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_2d: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from the 2D observation groups."""
        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):  # We assume obs_2d list matches the order of obs_groups_2d
            latent_cnn_list.append(cnn(obs_2d[i]))
        latent = torch.cat(latent_cnn_list, dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for CNN exports)."""
        pass


class _OnnxImageOnlyCNNModel(nn.Module):
    """Exportable image-only CNN model for ONNX.

    Unlike ``rsl_rl``'s exportable CNN model, the forward pass only takes the 2D observation
    groups as input, without a placeholder for the (empty) 1D observations.
    """

    def __init__(self, model: CNNModel, verbose: bool):
        super().__init__()
        self.verbose = verbose
        # Convert ModuleDict to ModuleList for ordered iteration
        self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self.obs_groups_2d = model.obs_groups_2d
        self.obs_dims_2d = model.obs_dims_2d
        self.obs_channels_2d = model.obs_channels_2d

    def forward(self, *obs_2d: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        latent_cnn_list = []
        for i, cnn in enumerate(self.cnns):
            latent_cnn_list.append(cnn(obs_2d[i]))
        latent = torch.cat(latent_cnn_list, dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        dummy_2d = []
        for i in range(len(self.obs_groups_2d)):
            h, w = self.obs_dims_2d[i]
            c = self.obs_channels_2d[i]
            dummy_2d.append(torch.zeros(1, c, h, w))
        return tuple(dummy_2d)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return list(self.obs_groups_2d)

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
