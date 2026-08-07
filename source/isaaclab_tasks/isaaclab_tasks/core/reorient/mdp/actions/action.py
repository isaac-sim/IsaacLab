# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action term implementations for the reorientation task family.

Kept apart from :mod:`~isaaclab_tasks.core.reorient.mdp.actions.action_cfg` because importing the
base action class pulls in the USD stage bindings, which configuration loading must not
require. The configuration there names this module through ``class_type`` instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import EMAJointPositionToLimitsAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .action_cfg import NoisyEMAJointPositionToLimitsActionCfg


class NoisyEMAJointPositionToLimitsAction(EMAJointPositionToLimitsAction):
    """Apply a stateful noise model before EMA joint-position processing."""

    def __init__(self, cfg: NoisyEMAJointPositionToLimitsActionCfg, env: ManagerBasedEnv):
        """Initialize the noisy action term.

        Args:
            cfg: Action configuration including the stateful noise model.
            env: Manager-based environment containing the hand.
        """
        super().__init__(cfg, env)
        self._noise_model = cfg.noise_model.class_type(cfg.noise_model, num_envs=self.num_envs, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Apply noise to normalized actions before scaling and EMA filtering.

        Args:
            actions: Normalized joint actions, shape ``(num_envs, num_actions)``.
        """
        super().process_actions(self._noise_model(actions))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the noise state and standard EMA action buffers.

        Args:
            env_ids: Environment indices to reset, or ``None`` for every environment.
        """
        self._noise_model.reset(env_ids)
        super().reset(env_ids)
