# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Partial robot adapter: implement the exercised surface, not the full articulation ABC."""

from __future__ import annotations

import math
from collections.abc import Sequence
from types import SimpleNamespace

import torch
import warp as wp

from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import clone
from isaaclab.utils.string import resolve_matching_names, resolve_matching_names_values
from isaaclab.utils.warp import ProxyArray


class RealArticulation:
    """Expose a single session's measured joints to stock Lab action and observation terms.

    Unsupported capabilities remain absent and fail at the requesting caller. This intentionally
    does not inherit BaseArticulation or claim body state, dynamics, rendering, or state replacement.
    """

    is_graphable = False

    def __init__(self, cfg: ArticulationCfg, *, sim: SimulationContext | None = None) -> None:
        self.cfg = clone(cfg)
        self.sim = sim if sim is not None else SimulationContext.instance()
        manager = self.sim.backend_manager
        plan = self.sim.get_clone_plan()
        if manager.views or (plan is not None and len(plan.topology.world_prototype_layout) != 1):
            raise NotImplementedError("The real backend prototype supports one robot in one environment.")
        if cfg.spawn is not None or cfg.actuators:
            raise NotImplementedError("Use spawn=None and actuators={}; the session owns the physical robot.")
        self.session = manager.session
        self.joint_names = list(self.session.joint_names)
        self.num_joints = len(self.joint_names)
        self.num_instances = 1
        self.device = "cpu"
        self.is_initialized = True
        self.timestamp = -math.inf
        self.data = SimpleNamespace(
            **{
                name: ProxyArray(wp.zeros((1, self.num_joints), dtype=wp.float32, device="cpu"))
                for name in ("joint_pos", "joint_vel", "default_joint_pos", "default_joint_vel", "joint_pos_target")
            }
        )
        for field, values in (
            ("default_joint_pos", cfg.init_state.joint_pos),
            ("default_joint_vel", cfg.init_state.joint_vel),
        ):
            indices, _, defaults = resolve_matching_names_values(values, self.joint_names)
            getattr(self.data, field).torch[:, indices] = torch.tensor(defaults, dtype=torch.float32)
        self.update(0.0)
        self._pending_command = False
        self.data.joint_pos_target.torch.copy_(self.data.joint_pos.torch)
        manager.views[cfg.prim_path] = self

    def find_joints(
        self, name_keys: str | Sequence[str], preserve_order: bool = False, *, as_proxy: bool = False
    ) -> tuple[list[int] | ProxyArray, list[str]]:
        """Resolve joint expressions in session order, or the caller's requested order."""
        indices, names = resolve_matching_names(name_keys, self.joint_names, preserve_order=preserve_order)
        if as_proxy:
            indices = ProxyArray(wp.array(indices, dtype=wp.int32, device="cpu"))
        return indices, names

    def set_joint_position_target_index(
        self,
        target: torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Buffer batched position commands; submitting them does not overwrite telemetry."""
        if env_ids is not None:
            raise NotImplementedError("Explicit environment selection is outside this single-robot prototype.")
        joint_ids = slice(None) if joint_ids is None else joint_ids
        selected = self.data.joint_pos_target.torch[:, joint_ids]
        if target.shape != selected.shape or not torch.isfinite(target).all():
            raise ValueError(f"Position targets must be finite with shape {tuple(selected.shape)}.")
        self.data.joint_pos_target.torch[:, joint_ids] = target
        self._pending_command = True

    def set_joint_position_target(
        self,
        target: torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor | slice | None = None,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Accept the legacy public setter as well as the indexed action-term interface."""
        self.set_joint_position_target_index(target, joint_ids=joint_ids, env_ids=env_ids)

    def write_data_to_sim(self) -> None:
        """Submit the pending command once, keeping session and Lab buffer ownership separate."""
        if self._pending_command:
            self.session.write(self.data.joint_pos_target.torch[0])
            self._pending_command = False

    def update(self, dt: float) -> None:
        """Copy a measured snapshot into stable ProxyArray buffers; dt does not advance the robot."""
        state = self.session.read()
        if not math.isfinite(state.timestamp) or state.timestamp < self.timestamp:
            raise ValueError("Session telemetry must have a finite, nondecreasing timestamp.")
        for destination, source in ((self.data.joint_pos, state.position), (self.data.joint_vel, state.velocity)):
            if source.shape != (self.num_joints,) or source.device.type != "cpu" or not torch.isfinite(source).all():
                raise ValueError(f"Expected finite CPU telemetry with shape ({self.num_joints},).")
            destination.torch[0].copy_(source)
        self.timestamp = state.timestamp

    def reset(
        self, env_ids: Sequence[int] | torch.Tensor | slice | None = None, env_mask: wp.array | None = None
    ) -> None:
        """Discard unsubmitted commands without cancelling a target already accepted by the session."""
        if env_mask is not None and not bool(wp.to_torch(env_mask)[0]):
            return
        if env_ids is not None and torch.arange(self.num_instances)[env_ids].numel() == 0:
            return
        self.update(0.0)
        self.data.joint_pos_target.torch.copy_(self.data.joint_pos.torch)
        self._pending_command = False
