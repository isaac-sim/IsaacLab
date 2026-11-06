# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture input selection for an existing task action term."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionInput
from isaaclab.utils import validate

from .source import TeleopSample, TeleopSource

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .input_cfg import TeleopInputCfg


class TeleopInput(ActionInput):
    """Select human or policy input while preserving the task controller.

    Input acquisition belongs to interactive playback, which activates a source
    on one input and distributes its sample across the task's teleop inputs
    before each environment step. Only the declaring input owns the source's
    lifetime. Inputs never step, render, or reset the simulation themselves.
    """

    cfg: TeleopInputCfg

    def __init__(self, cfg: TeleopInputCfg, env: ManagerBasedEnv, *, action_dim: int) -> None:
        validate(cfg)
        if max(cfg.env_ids) >= env.num_envs:
            raise ValueError(f"TeleopInputCfg.env_ids exceeds the environment batch size ({env.num_envs}).")
        super().__init__(cfg, env, action_dim=action_dim)
        self._source: TeleopSource | None = None
        self._sample: TeleopSample | None = None
        self._closed = False
        self._env_ids = torch.tensor(cfg.env_ids, device=env.device, dtype=torch.long)

    @property
    def sample(self) -> TeleopSample | None:
        """Latest cached input for distribution by the playback owner.

        Treat this sample and its action tensor as read-only. Each input's action
        adapter receives a separate copy when resolving its controller inputs.
        """
        return self._sample

    def activate(self, source: TeleopSource) -> None:
        """Take ownership of a source created after the full environment is ready."""
        if self._closed:
            raise RuntimeError("Cannot activate a closed teleop input.")
        if self._source is not None:
            raise RuntimeError("The teleop input already has an active source.")
        self._source = source
        self._sample = None

    def deactivate(self) -> None:
        """Release the source before borrowed XR or visualizer resources close."""
        self._sample = None
        if self._source is not None:
            self._source.close()
            self._source = None

    def poll_source(self) -> bool:
        """Cache one input sample and report its simulation-wide reset request."""
        self._sample = self._source.advance() if self._source is not None else None
        return self._sample is not None and self._sample.reset

    def set_sample(self, sample: TeleopSample | None) -> None:
        """Receive a shared input sample without owning or polling its source."""
        if self._closed:
            raise RuntimeError("Cannot publish input to a closed teleop input.")
        if self._source is not None:
            raise RuntimeError("A teleop input owning a source must poll that source.")
        self._sample = sample

    def resolve_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Replace selected rows only while the cached source sample is active."""
        sample = self._sample
        if sample is None or sample.reset or not sample.active or sample.actions is None:
            return actions
        values = sample.actions
        if self.cfg.action_adapter is not None:
            if not isinstance(values, torch.Tensor):
                raise TypeError("Teleop source must return a torch.Tensor.")
            values = self.cfg.action_adapter(values.clone(), self._env, self._env_ids)
        if not isinstance(values, torch.Tensor):
            raise TypeError("Teleop source or action_adapter must return a torch.Tensor.")
        if values.ndim == 1 and len(self._env_ids) == 1:
            values = values.unsqueeze(0)
        expected = (len(self._env_ids), self.action_dim)
        if tuple(values.shape) != expected:
            raise ValueError(f"Teleop actions have shape {tuple(values.shape)}; expected {expected}.")
        if not torch.isfinite(values).all():
            raise ValueError("Teleop actions must be finite.")
        selected = actions.clone()
        selected[self._env_ids] = values.to(device=actions.device, dtype=actions.dtype)
        return selected

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Notify Capture and clear cached actions when controlled rows reset."""
        if self._source is None and self._sample is None:
            return
        if env_ids is None:
            selected = self._env_ids
        elif isinstance(env_ids, slice):
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_ids] = True
            selected = self._env_ids[mask[self._env_ids]]
        else:
            reset_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            selected = self._env_ids[torch.isin(self._env_ids, reset_ids)]
        if len(selected):
            self._sample = None
            if self._source is not None:
                self._source.reset(selected)

    def close(self) -> None:
        """Release Capture before the task controllers and simulation resources."""
        if self._closed:
            return
        self.deactivate()
        self._closed = True
