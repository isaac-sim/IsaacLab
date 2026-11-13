# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-guided row selection for cached or procedurally generated reset states."""

from __future__ import annotations

import torch

from isaaclab.utils import instantiate

from .reset_states_cfg import ResetStateSamplerCfg
from .sampling import Sampler
from .success_monitor import SuccessMonitor, SuccessMonitorCfg


class ResetStateSampler:
    """Track the assigned reset row and completed outcomes for each environment.

    Tasks generate a fixed catalog of reset states and restore the selected rows through their
    normal reset events. Rows can index tensor tables, cached scene states, or deterministic
    procedural recipes. This class neither stores physical states nor writes to the scene.

    Args:
        cfg: Outcome history and sampling configuration.
        num_envs: Number of environments sharing the row catalog.
        row_count: Number of reset rows in the catalog.
        device: Device used by the catalog and episode outcomes.
    """

    def __init__(self, cfg: ResetStateSamplerCfg, num_envs: int, row_count: int, device: str | torch.device) -> None:
        if num_envs < 1 or row_count < 1:
            raise ValueError("num_envs and row_count must be positive.")
        self.cfg = cfg
        self.row_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        """Current reset-row assignment for every environment, shape [num_envs]."""
        self.success_monitor = SuccessMonitor(
            SuccessMonitorCfg(monitored_history_len=cfg.monitored_history_len), 1, row_count, device
        )
        """Rolling outcomes indexed by reset row."""
        self.sampler: Sampler = instantiate(cfg.sampling, self.success_monitor.success_rate)
        self._probabilities = self.sampler.probabilities()

    @property
    def probabilities(self) -> torch.Tensor:
        """Copy of the most recently used row probabilities, shape [row_count]."""
        return self._probabilities.clone()

    def reset(
        self,
        env_ids: torch.Tensor,
        success: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Record the finished episodes before selecting their next reset rows.

        Call this before clearing task success flags or restoring physical state. Pass no outcomes
        at startup, or mask zero-step/manual resets with ``valid``. Only selected environments'
        assignments change; outcome history remains shared across the entire row catalog.

        Args:
            env_ids: Resetting environment indices, shape [N], on the sampler device.
            success: Whether each selected episode succeeded, shape [N]. ``None`` skips recording.
            valid: Which selected outcomes to record, shape [N]. ``None`` records every outcome.

        Returns:
            The newly assigned reset-row indices, shape [N], in ``env_ids`` order.
        """
        if success is not None:
            self.success_monitor.success_update(self.row_ids[env_ids], success, valid)
        self._probabilities = self.sampler.probabilities()
        if env_ids.numel():
            self.row_ids[env_ids] = self.sampler.sample(self._probabilities, env_ids.numel())
        return self.row_ids[env_ids]
