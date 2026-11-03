# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep the upstream per-rollout-step normalizer timing on RSL-RL 5.5."""

from rsl_rl.models import MLPModel
from tensordict import TensorDict


class MicroDuckVelStandMLPModel(MLPModel):
    """MLP with source-compatible explicit statistics updates.

    RSL-RL 5.0 updates next observations in ``process_env_step``. Version 5.5 instead
    updates a complete batch after PPO; the latter must be suppressed for both student models.
    The native exporter still includes the unmodified empirical normalizer (epsilon .01).
    """

    def update_rollout_normalization(self, obs: TensorDict) -> None:
        """Update once for the next observations of each environment step."""
        super().update_normalization(obs)

    def update_normalization(self, obs: TensorDict) -> None:
        """Suppress the native post-PPO batch update; the algorithm owns update timing."""
