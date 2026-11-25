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
import warp as wp

from ..utils.backend_utils import FactoryBase
from ..utils.warp import ProxyArray, memory

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
        if not cfg.surface_specs:
            raise ValueError("Provide at least one conveyor surface when constructing the controller.")
        if num_envs < 1:
            raise ValueError("Conveyor environment count must be positive.")
        self.cfg = cfg.copy()
        self.num_envs = num_envs
        self.device = sim.device
        self._speed_scale = ProxyArray(wp.ones(num_envs, dtype=wp.float32, device=self.device))
        self._all_env_ids = wp.array(range(num_envs), dtype=wp.int32, device=self.device)
        self._all_env_mask = wp.ones(num_envs, dtype=wp.bool, device=self.device)
        self._selection_mask = wp.empty(num_envs, dtype=wp.bool, device=self.device)
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """Whether the controller is bound to the current physics model."""
        return self._is_initialized

    @property
    def speed_scale(self) -> torch.Tensor:
        """Copy of the target multiplier for each environment, shape ``(N,)``."""
        return self._speed_scale.torch.clone()

    def set_speed_scale(
        self,
        scale: float | torch.Tensor | wp.array,
        env_ids: Sequence[int] | slice | torch.Tensor | wp.array | None = None,
        *,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
        """Set target multipliers for selected environments without rebinding physics.

        Positive values preserve the authored directions, zero stops motion and
        negative values reverse it. Commands persist through episode and hard resets.
        Tensor commands must be finite; their values are not checked to avoid a
        device synchronization when issuing commands during CUDA graph capture.

        Args:
            scale: A finite scalar or one value per selected environment. Masked array
                commands use the full environment shape. Native arrays use float32 on this device.
            env_ids: Environments to command, in the order of compact array values.
                ``None`` selects all environments. Torch inputs are boundary shims.
            env_mask: Native full-environment mask. Same-device masks and arrays avoid
                index compaction and are suitable for CUDA graph replay.
        """
        if env_ids is not None and env_mask is not None:
            raise ValueError("Specify either env_ids or env_mask, not both.")
        if isinstance(scale, (float, int)):
            if not math.isfinite(scale):
                raise ValueError("Conveyor speed scales must be finite.")
            mask = memory.resolve_1d_mask(
                ids=env_ids, mask=env_mask, all_mask=self._all_env_mask, scratch_mask=self._selection_mask
            )
            wp.launch(
                memory.make_masked_fill(wp.float32, 1),
                self.num_envs,
                inputs=[float(scale), mask, None],
                outputs=[self._speed_scale.warp],
                device=self.device,
            )
            return
        ids = (
            None
            if env_mask is not None
            else memory.resolve_1d_indices(memory.resolve_ids(ids=env_ids, all_ids=self._all_env_ids))
        )
        count = self.num_envs if ids is None else ids.shape[0]
        if isinstance(scale, torch.Tensor):
            scale = scale.to(dtype=torch.float32, device=self.device)
            if scale.ndim == 0:
                scale = scale.expand(count)
            scale = wp.from_torch(scale)
        if scale.shape != (count,) or scale.dtype != wp.float32 or scale.device != self._speed_scale.warp.device:
            raise ValueError("Conveyor scale must be float32 with one value per command on the controller device.")
        if env_mask is None:
            wp.copy(wp.indexedarray(self._speed_scale.warp, [ids]), scale)
        else:
            mask = memory.resolve_1d_mask(mask=env_mask, all_mask=self._all_env_mask, scratch_mask=self._selection_mask)
            wp.launch(
                memory.make_masked_copy(wp.float32, 1),
                self.num_envs,
                inputs=[scale, mask, None],
                outputs=[self._speed_scale.warp],
                device=self.device,
            )

    def stop(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Command zero speed for selected environments."""
        self.set_speed_scale(0.0, env_ids)

    def start(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Restore selected environments to their authored velocities (scale one)."""
        self.set_speed_scale(1.0, env_ids)

    @abstractmethod
    def reset(
        self,
        env_ids: Sequence[int] | slice | torch.Tensor | wp.array | None = None,
        *,
        env_mask: torch.Tensor | wp.array | None = None,
    ) -> None:
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
