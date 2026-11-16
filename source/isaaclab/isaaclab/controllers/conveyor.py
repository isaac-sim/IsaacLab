# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched commands for prescribed conveyor surface motion."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ..utils.backend_utils import FactoryBase

if TYPE_CHECKING:
    from ..sim import SimulationContext
    from .conveyor_cfg import ConveyorCfg


class BaseConveyor(ABC):
    """Common configuration, command buffers and state for conveyor controllers.

    Construct before the first simulation reset. The controller binds to existing
    collision surfaces; scene assets own their geometry. Each command is a
    dimensionless multiplier on all authored surface velocities in one environment.
    """

    def __init__(self, cfg: ConveyorCfg, num_envs: int, *, sim: SimulationContext) -> None:
        cfg.validate()
        if num_envs < 1 or cfg.body_count_per_env < 1:
            raise ValueError("Conveyor environment and transported-body counts must be positive.")
        self.cfg = cfg.copy()
        self.num_envs = num_envs
        self.device = sim.device
        self._speed_scale = torch.ones(num_envs, dtype=torch.float32, device=self.device)
        self._surface_velocities = torch.tensor(
            [surface.velocity for surface in self.cfg.surface_specs], dtype=torch.float32, device=self.device
        )
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """Whether the controller is bound to the current physics model."""
        return self._is_initialized

    @property
    def speed_scale(self) -> torch.Tensor:
        """Copy of the target multiplier for each environment, shape ``(N,)``."""
        return self._speed_scale.clone()

    @property
    def surface_velocities(self) -> torch.Tensor:
        """Commanded signed centerline velocities [m/s], shape ``(N, S)``.

        These are the targets before the startup traction ramp. Collision surfaces
        remain stationary; the backend imparts their prescribed motion to contacts.
        """
        return self._speed_scale[:, None] * self._surface_velocities[None, :]

    def set_speed_scale(self, scale: float | torch.Tensor, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Set target multipliers for selected environments without rebinding physics.

        Positive values preserve the authored directions, zero stops motion and
        negative values reverse it. Commands persist through episode and hard resets.
        Tensor commands must be finite; their values are not checked to avoid a
        device synchronization when issuing commands during CUDA graph capture.

        Args:
            scale: A finite scalar or one value per selected environment.
            env_ids: Environments to command. ``None`` selects all environments.
        """
        if not isinstance(scale, torch.Tensor) and not math.isfinite(scale):
            raise ValueError("Conveyor speed scales must be finite.")
        values = torch.as_tensor(scale, dtype=torch.float32, device=self.device)
        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        count = self.num_envs if env_ids is None else len(env_ids)
        if values.ndim > 1 or (values.ndim == 1 and values.numel() != count):
            raise ValueError("Conveyor scale must be a scalar or one value per selected environment.")
        self._speed_scale[ids] = values

    def stop(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Command zero speed for selected environments."""
        self.set_speed_scale(0.0, env_ids)

    def start(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Restore selected environments to their authored velocities (scale one)."""
        self.set_speed_scale(1.0, env_ids)

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear selected-world force history without changing commanded speed scales."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Detach solver callbacks before the physics scene is destroyed; safe to call twice."""
        raise NotImplementedError


class Conveyor(FactoryBase, BaseConveyor):
    """Create a conveyor controller for the simulation's active physics backend.

    Pass the simulation context explicitly with ``Conveyor(cfg, num_envs, sim=sim)``.
    Backend implementations own solver binding, force application, :meth:`reset`
    and :meth:`close`; the common command/state interface is :class:`BaseConveyor`.
    """

    def __new__(cls, cfg: ConveyorCfg, num_envs: int, *, sim: SimulationContext) -> BaseConveyor:
        """Select the implementation from the supplied simulation context."""
        return super().__new__(cls, cfg, num_envs, sim=sim)
