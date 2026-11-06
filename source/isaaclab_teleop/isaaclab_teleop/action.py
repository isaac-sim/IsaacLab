# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task action term that arbitrates Capture input before controller processing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm
from isaaclab.utils import instantiate, validate

from .source import TeleopSample, TeleopSource

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .action_cfg import TeleopActionCfg


class TeleopAction(ActionTerm):
    """Select human or policy input while preserving the task controller.

    Input acquisition belongs to interactive playback, which activates a source
    on one term and distributes its sample across the task's teleop terms before each
    environment step. Only the declaring term owns the activated source's
    lifetime. Terms never step, render, or reset the simulation themselves.
    """

    cfg: TeleopActionCfg

    def __init__(self, cfg: TeleopActionCfg, env: ManagerBasedEnv) -> None:
        validate(cfg)
        if max(cfg.env_ids) >= env.num_envs:
            raise ValueError(f"TeleopActionCfg.env_ids exceeds the environment batch size ({env.num_envs}).")
        self._controller: ActionTerm | None = None
        self._source: TeleopSource | None = None
        self._sample: TeleopSample | None = None
        self._closed = False
        # The nested controller is authoritative for the inherited asset and
        # visualization configuration. No Capture resources start here.
        super().__init__(cfg.controller, env)
        self.cfg = cfg
        self._env_ids = torch.tensor(cfg.env_ids, device=env.device, dtype=torch.long)
        self._controller = instantiate(cfg.controller, env)
        if not isinstance(self._controller, ActionTerm):
            raise TypeError("TeleopActionCfg.controller must construct an ActionTerm.")

    @property
    def controller(self) -> ActionTerm:
        """The existing task controller, including its configuration and subtype API.

        Task functions that previously accessed controller-specific attributes
        through ``action_manager.get_term(name)`` must use this property when
        the term is wrapped. Arbitrary attributes and type checks are not proxied.
        """
        return self._controller

    @property
    def sample(self) -> TeleopSample | None:
        """Latest cached input for distribution by the playback owner.

        Treat this sample and its action tensor as read-only. Each term's action
        adapter receives a separate copy when resolving its controller inputs.
        """
        return self._sample

    @property
    def action_dim(self) -> int:
        return self._controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._controller.raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._controller.processed_actions

    @property
    def has_debug_vis_implementation(self) -> bool:
        return self._controller is not None and self._controller.has_debug_vis_implementation

    def set_debug_vis(self, debug_vis: bool) -> bool:
        return self._controller is not None and self._controller.set_debug_vis(debug_vis)

    def activate(self, source: TeleopSource) -> None:
        """Take ownership of a source created after the full environment is ready."""
        if self._closed:
            raise RuntimeError("Cannot activate a closed teleop action term.")
        if self._source is not None:
            raise RuntimeError("The teleop action term already has an active source.")
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
            raise RuntimeError("Cannot publish input to a closed teleop action term.")
        if self._source is not None:
            raise RuntimeError("A teleop action term owning input must poll its source.")
        self._sample = sample

    def resolve_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Replace selected rows only while the cached source sample is active."""
        sample = self._sample
        if sample is None or sample.reset or not sample.active or sample.actions is None:
            return self._controller.resolve_actions(actions)
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
        return self._controller.resolve_actions(selected)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._controller.process_actions(actions)

    def apply_actions(self) -> None:
        self._controller.apply_actions()

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Reset the controller and notify Capture when controlled rows reset."""
        self._controller.reset(env_ids)
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
        """Release Capture before the nested controller and simulation resources."""
        if self._closed:
            return
        self.deactivate()
        if self._controller is not None:
            self._controller.close()
        super().close()
        self._closed = True
