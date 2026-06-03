# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL neural models customized for Isaac Lab."""

from __future__ import annotations

import torch
from rsl_rl.models.cnn_model import CNNModel as _CNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict


class CNNModel(_CNNModel):
    """CNN model that supports pure image-only observations.

    The rsl_rl CNN model does not support image-only observations as it calls
    :meth:`get_latent` without checking whether the observation groups are empty.
    """

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        latent_cnn = torch.cat([self.cnns[group](obs[group]) for group in self.obs_groups_2d], dim=-1)
        if not self.obs_groups:
            return latent_cnn
        latent_1d = MLPModel.get_latent(self, obs, masks, hidden_state)
        return torch.cat([latent_1d, latent_cnn], dim=-1)
