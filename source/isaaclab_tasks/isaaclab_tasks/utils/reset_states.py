# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Outcome tracking and ID selection for cached or procedural reset catalogs."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from isaaclab.utils import instantiate

from .reset_states_cfg import ResetStateSamplerCfg
from .sampling import Sampler
from .success_monitor import SuccessMonitor, SuccessMonitorCfg


class ResetStateSampler:
    """Track reset assignments and completed outcomes independently of physical states.

    A catalog can identify tensor states or procedural recipes, such as seeds. Scores receive
    named per-catalog metrics; the default uses measured success rates. Tasks restore selected
    IDs through their normal reset lifecycle. This class does not write physical state.

    Args:
        cfg: Outcome history and sampling configuration.
        num_envs: Number of environments sharing the row catalog.
        num_states: Number of catalog entries.
        device: Device used by the catalog and episode outcomes.
        state_ids: Optional unique integer catalog IDs, shape [num_states]. Defaults to
            contiguous indices. History and metric tensors follow this catalog order.
        metrics: Additional caller-owned tensors indexed in catalog order. Update these tensors
            in place before resetting. The reserved ``success_rate`` metric tracks episode outcomes.
    """

    def __init__(
        self,
        cfg: ResetStateSamplerCfg,
        num_envs: int,
        num_states: int,
        device: str | torch.device,
        *,
        state_ids: torch.Tensor | None = None,
        metrics: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        if num_envs < 1 or num_states < 1:
            raise ValueError("num_envs and num_states must be positive.")
        self.cfg = cfg
        self._indices = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._num_states = num_states
        self._state_ids = None
        self._sorted_ids = None
        self._sorted_indices = None
        if state_ids is not None:
            if state_ids.shape != (num_states,) or state_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("state_ids must be an integer tensor shaped [num_states].")
            self._state_ids = state_ids.to(device=device, dtype=torch.long).clone()
            self._sorted_ids, self._sorted_indices = self._state_ids.sort()
            if (self._sorted_ids[1:] == self._sorted_ids[:-1]).any():
                raise ValueError("Catalog state_ids must be unique.")
        self.success_monitor = SuccessMonitor(
            SuccessMonitorCfg(monitored_history_len=cfg.monitored_history_len), 1, num_states, device
        )
        """Rolling outcomes in catalog order."""
        if metrics is not None and "success_rate" in metrics:
            raise ValueError("The success_rate metric is supplied by the outcome monitor.")
        self._metrics = {"success_rate": self.success_monitor.success_rate, **(metrics or {})}
        self.sampler: Sampler = instantiate(cfg.sampling, num_states, device)
        self._probabilities = self.sampler.probabilities(self._metrics)

    @property
    def probabilities(self) -> torch.Tensor:
        """Copy of the most recently used probabilities in catalog order, shape [num_states]."""
        return self._probabilities.clone()

    def get_state_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return a copy of the current IDs for the selected environments, shape [N]."""
        indices = self._indices[env_ids]
        return indices if self._state_ids is None else self._state_ids[indices]

    def reset(
        self,
        env_ids: torch.Tensor,
        success: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        state_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Record finished outcomes, then assign sampled or explicitly requested IDs.

        Call this before clearing task success flags or restoring physical state. Pass no outcomes
        at startup, or mask zero-step/manual resets with ``valid``. Only selected environments'
        assignments change; outcome history remains shared across the entire row catalog.

        Args:
            env_ids: Resetting environment indices, shape [N], on the sampler device.
            success: Whether each selected episode succeeded, shape [N]. ``None`` skips recording.
            valid: Which selected outcomes to record, shape [N]. ``None`` records every outcome.
            state_ids: Requested catalog IDs, shape [N]. ``None`` draws from the configured scores.
                Explicit requests update the assignment used to credit the next outcome.

        Returns:
            The newly assigned catalog IDs, shape [N], in ``env_ids`` order.
        """
        selected = None
        if state_ids is not None:
            if state_ids.shape != env_ids.shape or state_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("Requested state_ids must be an integer tensor shaped like env_ids.")
            requested = state_ids.to(device=self._indices.device, dtype=torch.long)
            if self._state_ids is None:
                if ((requested < 0) | (requested >= self._num_states)).any():
                    raise ValueError("Requested state_ids are outside the catalog.")
                selected = requested
            else:
                positions = torch.searchsorted(self._sorted_ids, requested).clamp_max(self._num_states - 1)
                if (self._sorted_ids[positions] != requested).any():
                    raise ValueError("Requested state_ids are not in the catalog.")
                selected = self._sorted_indices[positions]
        if success is not None:
            self.success_monitor.success_update(self._indices[env_ids], success, valid)
        self._probabilities = self.sampler.probabilities(self._metrics)
        if env_ids.numel():
            self._indices[env_ids] = (
                self.sampler.sample(self._probabilities, env_ids.numel()) if selected is None else selected
            )
        return self.get_state_ids(env_ids)
