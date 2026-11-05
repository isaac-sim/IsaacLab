# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Input lifecycle and action ownership at the RL environment boundary."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import torch

from isaaclab.utils import instantiate, validate

from .direct_marl_env_cfg import DirectMARLEnvCfg
from .manager_based_env_cfg import ManagerBasedEnvCfg


@dataclass
class TeleopSample:
    """One non-blocking source poll, expressed in task action conventions.

    ``active=False`` or ``actions=None`` yields control to the baseline policy.
    ``reset=True`` requests the existing simulation-wide episode reset on this step;
    that sample's actions are discarded. This is not a rewind or a physics pause.
    """

    actions: torch.Tensor | None = None
    active: bool = True
    reset: bool = False


class TeleopSource(ABC):
    """An input resource whose lifecycle is independent of the simulation loop.

    Implementations receive ``(cfg, env, *, env_ids)`` at construction. They may
    borrow scene and visualizer resources from ``env`` but must release those
    resources on :meth:`close`, including after partially failed construction.
    Polling must not step, render, or reset the environment. This interface keeps
    the optional Capture dependency outside the core environment package.
    """

    @abstractmethod
    def advance(self) -> TeleopSample:
        """Read the latest actions and control events without advancing simulation."""

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset source state after the specified controlled environments reset."""

    @abstractmethod
    def close(self) -> None:
        """Release input resources. Repeated calls must be harmless."""


class TeleopEnvWrapper(gym.Wrapper):
    """Overlay teleop actions before the task consumes policy actions.

    Install inside RL-library wrappers so their policy normalization and clipping
    do not change human actions. The task retains ownership of action processing,
    observations, rewards, recording, simulation timing, and automatic resets.

    Supports tensor-valued Gymnasium Box, Discrete, and MultiDiscrete action spaces,
    including one selected agent of a direct multi-agent task. Composite action
    spaces require a task-specific integration. The wrapper never clips or changes
    task action semantics; configure the source or adapter accordingly.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        task = env.unwrapped
        cfg = task.cfg.teleop
        if cfg is None:
            raise ValueError("TeleopEnvWrapper requires env.cfg.teleop.")
        validate(cfg)
        if max(cfg.env_ids) >= task.num_envs:
            raise ValueError(f"teleop.env_ids exceeds the environment batch size ({task.num_envs}).")

        if isinstance(task.cfg, DirectMARLEnvCfg):
            if cfg.agent not in task.action_spaces:
                raise ValueError(f"teleop.agent must select one of {list(task.action_spaces)}.")
            space = task.action_spaces[cfg.agent]
        else:
            if cfg.agent is not None:
                raise ValueError("teleop.agent is only supported for direct multi-agent environments.")
            space = task.single_action_space
        if not isinstance(space, (gym.spaces.Box, gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
            raise ValueError(f"Teleop requires a tensor-valued action space, received {space}.")
        self._action_dim = math.prod(space.shape)
        if cfg.action_terms:
            if not isinstance(task.cfg, ManagerBasedEnvCfg):
                raise ValueError("teleop.action_terms requires a manager-based environment.")
            terms = {}
            offset = 0
            for name, dimension in zip(task.action_manager.active_terms, task.action_manager.action_term_dim):
                terms[name] = range(offset, offset + dimension)
                offset += dimension
            unknown = set(cfg.action_terms) - terms.keys()
            if unknown:
                raise ValueError(f"Unknown teleop action terms: {sorted(unknown)}. Available: {list(terms)}.")
            columns = [column for name in cfg.action_terms for column in terms[name]]
        else:
            columns = cfg.action_indices or list(range(self._action_dim))
        if not columns or max(columns) >= self._action_dim:
            raise ValueError(f"Teleop action selection exceeds the action dimension ({self._action_dim}).")

        self._env_ids = torch.tensor(cfg.env_ids, device=task.device, dtype=torch.long)
        self._columns = torch.tensor(columns, device=task.device, dtype=torch.long)
        self._agent = cfg.agent
        self._adapter = cfg.action_adapter
        self._closed = False
        self._source = instantiate(cfg.source, task, env_ids=list(cfg.env_ids))

    def reset(self, **kwargs) -> Any:
        result = self.env.reset(**kwargs)
        self._source.reset(self._env_ids)
        return result

    def step(self, actions: torch.Tensor | dict[str, torch.Tensor]) -> Any:
        sample = self._source.advance()
        if sample.reset:
            self.unwrapped.sim.request_reset()
        elif sample.active and sample.actions is not None:
            baseline = actions if self._agent is None else actions[self._agent]
            if not isinstance(baseline, torch.Tensor):
                raise TypeError("Teleop expects tensor-valued task actions.")
            values = sample.actions
            if self._adapter is not None:
                values = self._adapter(values, self.unwrapped, self._env_ids)
            if not isinstance(values, torch.Tensor):
                raise TypeError("Teleop source or action_adapter must return a torch.Tensor.")
            if values.ndim == 1 and len(self._env_ids) == 1:
                values = values.unsqueeze(0)
            expected = (len(self._env_ids), len(self._columns))
            if tuple(values.shape) != expected:
                raise ValueError(f"Teleop actions have shape {tuple(values.shape)}; expected {expected}.")
            if not torch.isfinite(values).all():
                raise ValueError("Teleop actions must be finite.")
            if not baseline.is_floating_point() and values.is_floating_point():
                raise TypeError("Discrete teleop actions must have an integer dtype; provide an action_adapter.")
            selected = baseline.clone(memory_format=torch.contiguous_format)
            selected.view(self.unwrapped.num_envs, self._action_dim)[self._env_ids[:, None], self._columns[None, :]] = (
                values.to(device=baseline.device, dtype=baseline.dtype)
            )
            actions = selected if self._agent is None else {**actions, self._agent: selected}

        result = self.env.step(actions)
        # Returned done signals include UI resets that are not in reset_buf.
        # Direct MARL resets when all agents terminate OR all agents time out.
        terminated, truncated = result[2:4]
        if isinstance(terminated, dict):
            reset_mask = math.prod(terminated.values()) | math.prod(truncated.values())
        else:
            reset_mask = terminated | truncated
        reset_ids = self._env_ids[reset_mask[self._env_ids].bool()]
        if len(reset_ids):
            self._source.reset(reset_ids)
        return result

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._source.close()
            finally:
                self.env.close()
