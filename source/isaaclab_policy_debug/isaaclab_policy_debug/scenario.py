# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class PolicyDebugScenarioAdapter(ABC):
    """Task contract for exact synchronized policy-debug scenarios."""

    def configure_run(self, run_dir: str | Path) -> None:
        """Inspect run artifacts before the environment configuration is finalized."""

    def configure_environment(self, env_cfg: Any, capacity: int) -> None:
        """Apply configuration needed before environment construction."""
        env_cfg.scene.num_envs = capacity

    def validate_checkpoint(self, checkpoint: Any, env: Any) -> None:
        """Validate task-specific checkpoint metadata before policy activation."""

    def rollout_failures(self, env: Any, env_ids: Sequence[int]) -> dict[int, str]:
        """Return per-slot numerical failures that must be removed from the rollout."""
        return {}

    def before_step(self, env: Any, env_ids: Sequence[int]) -> None:
        """Prepare task-owned synchronization immediately before one environment step."""

    def after_step(self, env: Any, env_ids: Sequence[int]) -> None:
        """Restore temporary task synchronization state after one environment step."""

    def accept_automatic_reset(self, env: Any, env_ids: Sequence[int]) -> bool:
        """Validate a synchronized same-step autoreset and report whether it is complete.

        Returning ``True`` tells the manager to keep the observations returned by
        ``env.step`` and only clear policy recurrent state. Returning ``False``
        requests the adapter's ordinary :meth:`reset_synchronized` boundary.
        """
        return False

    def comparison_visible_assets(self) -> Sequence[str] | None:
        """Return scene asset names to render in every comparison layer.

        Return ``None`` to preserve every model shape. Tasks can use this
        render-only hook to omit visual context that obscures the comparison,
        without removing its physics representation from the environment.
        """
        return None

    def overlay_visible_assets(self) -> Sequence[str] | None:
        """Return scene asset names to render in translucent comparison layers."""
        return None

    @abstractmethod
    def reset_synchronized(self, env: Any, env_ids: Sequence[int]) -> Any:
        """Advance and apply one exactly repeated scenario to all active slots."""


class ManagerBasedSeededScenarioAdapter(PolicyDebugScenarioAdapter):
    """Conservative built-in adapter for manager-based tasks with verifiable state."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.episode = 0

    def reset_synchronized(self, env: Any, env_ids: Sequence[int]) -> Any:
        import torch

        base = env.unwrapped
        if not hasattr(base, "scene") or not hasattr(base.scene, "get_state") or not hasattr(base, "reset_to"):
            raise RuntimeError("Task cannot be synchronized: manager-based scene state APIs are unavailable")
        ids = list(env_ids)
        if not ids:
            return None
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=base.device)
        base.reset(seed=self.seed + self.episode)
        state = base.scene.get_state(is_relative=True)
        repeated: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        for asset_type, assets in state.items():
            repeated[asset_type] = {}
            for asset_name, fields in assets.items():
                repeated[asset_type][asset_name] = {
                    field_name: value[ids[0] : ids[0] + 1].expand(len(ids), *value.shape[1:]).clone()
                    for field_name, value in fields.items()
                }
        base.reset_to(repeated, env_ids=ids_t, seed=self.seed + self.episode, is_relative=True)
        self._synchronize_commands(base, ids)
        self._verify_scene(base.scene.get_state(is_relative=True), ids)
        self.episode += 1
        return None

    def _synchronize_commands(self, env: Any, ids: list[int]) -> None:
        manager = getattr(env, "command_manager", None)
        if manager is None:
            return
        for name in manager.active_terms:
            command = manager.get_command(name)
            command[ids] = command[ids[0]].clone()
            if not torch_rows_equal(command, ids):
                raise RuntimeError(f"Task cannot be synchronized exactly: command term '{name}' differs across slots")

    def _verify_scene(self, state: dict[str, Any], ids: list[int]) -> None:
        for assets in state.values():
            for asset_name, fields in assets.items():
                for field_name, value in fields.items():
                    if not torch_rows_equal(value, ids):
                        raise RuntimeError(
                            "Task cannot be synchronized exactly: relative scene state "
                            f"'{asset_name}.{field_name}' differs across slots"
                        )


def torch_rows_equal(value: Any, ids: Sequence[int]) -> bool:
    """Return whether selected tensor rows are bitwise equal."""
    import torch

    if len(ids) < 2:
        return True
    selected = value[torch.as_tensor(ids, device=value.device)]
    return bool(torch.equal(selected, selected[0:1].expand_as(selected)))


def resolve_scenario_adapter(task_id: str, explicit: str | PolicyDebugScenarioAdapter | None = None):
    """Resolve explicit, Gym-registered, then built-in scenario adapters."""
    from isaaclab.utils.string import string_to_callable

    if isinstance(explicit, PolicyDebugScenarioAdapter):
        return explicit
    if explicit:
        factory = string_to_callable(explicit, separator=".")
        return factory() if callable(factory) else factory

    import gymnasium as gym

    spec = gym.spec(task_id)
    entry_point = spec.kwargs.get("policy_debug_adapter_entry_point")
    if entry_point:
        factory = string_to_callable(entry_point, separator=".") if isinstance(entry_point, str) else entry_point
        return factory() if callable(factory) else factory
    return ManagerBasedSeededScenarioAdapter()
