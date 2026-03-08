# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager call switch for routing manager stage calls through stable/warp/captured paths."""

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from enum import IntEnum
from typing import Any

import warp as wp
from isaaclab_experimental.utils.timer import Timer


class ManagerCallMode(IntEnum):
    """Execution mode for manager stage calls.

    * ``STABLE``  (0): Call stable Python manager implementations from :mod:`isaaclab.managers`.
    * ``WARP_NOT_CAPTURED`` (1): Call Warp-compatible implementations without CUDA graph capture.
    * ``WARP_CAPTURED`` (2): Call Warp implementations with CUDA graph capture/replay.
    """

    STABLE = 0
    WARP_NOT_CAPTURED = 1
    WARP_CAPTURED = 2


class ManagerCallSwitch:
    """Per-manager call switch for stable/warp/captured execution.

    Routes each manager stage call through the configured execution path:
    stable Python, Warp (eager), or Warp (captured CUDA graph). Optionally
    wraps each call in a :class:`Timer` context for profiling.
    """

    DEFAULT_CONFIG: dict[str, int] = {"default": 2}
    DEFAULT_KEY = "default"
    MANAGER_NAMES: tuple[str, ...] = (
        "ActionManager",
        "ObservationManager",
        "EventManager",
        "RecorderManager",
        "CommandManager",
        "TerminationManager",
        "RewardManager",
        "CurriculumManager",
    )
    # FIXME: Scene_write_data_to_sim calls articulation._apply_actuator_model which
    #  uses wp.to_torch + torch indexing -- not capture-safe on this branch.
    #  Cap Scene stages to WARP_NOT_CAPTURED until the articulation layer is capture-ready.
    MAX_MODE_OVERRIDES: dict[str, int] = {"Scene": ManagerCallMode.WARP_NOT_CAPTURED}

    def __init__(self, cfg_source: str | None = None):
        self._wp_graphs: dict[str, Any] = {}
        self._cfg = self._load_cfg(cfg_source)
        print("[INFO] ManagerCallSwitch configuration:")
        print(f"  - {self.DEFAULT_KEY}: {self._cfg[self.DEFAULT_KEY]}")
        for manager_name in self.MANAGER_NAMES:
            print(f"  - {manager_name}: {int(self.get_mode_for_manager(manager_name))}")

    # ------------------------------------------------------------------
    # Graph management
    # ------------------------------------------------------------------

    def invalidate_graphs(self) -> None:
        """Invalidate cached capture graphs."""
        self._wp_graphs.clear()

    # ------------------------------------------------------------------
    # Stage dispatch
    # ------------------------------------------------------------------

    def call_stage(
        self,
        *,
        stage: str,
        stable_calls: Sequence[dict[str, Any]],
        warp_calls: Sequence[dict[str, Any]],
        timer: bool = True,
    ) -> Any:
        """Run the stage according to configured mode, optionally wrapped in a :class:`Timer`.

        Args:
            stage: Stage identifier in the form ``"ManagerName_function_name"``.
            stable_calls: Call specs for the stable (torch) path.
            warp_calls: Call specs for the warp path.
            timer: Whether to wrap execution in a :class:`Timer`. Defaults to ``True``
                (controlled by the global :attr:`Timer.enable` class-level toggle).
                Pass a module-level flag like ``TIMER_ENABLED_STEP`` to make timing
                conditional on that flag.

        Returns:
            The return value of the last call in the selected path, or ``None``
            when the stage was graph-captured/replayed.
        """
        if timer:
            timer_name = stage.replace("_", ".", 1)
            with Timer(name=timer_name, msg=f"{timer_name} took:", enable=True, format="us"):
                return self._dispatch(stage, stable_calls, warp_calls)
        else:
            return self._dispatch(stage, stable_calls, warp_calls)

    def _dispatch(
        self,
        stage: str,
        stable_calls: Sequence[dict[str, Any]],
        warp_calls: Sequence[dict[str, Any]],
    ) -> Any:
        """Inner dispatch without timer wrapping."""
        manager_name = self._manager_name_from_stage(stage)
        mode = self.get_mode_for_manager(manager_name)
        if mode == ManagerCallMode.STABLE:
            return self._run_calls(stable_calls)
        if mode == ManagerCallMode.WARP_NOT_CAPTURED:
            return self._run_calls(warp_calls)
        self._wp_capture_or_launch(stage=stage, calls=warp_calls)
        return None

    # ------------------------------------------------------------------
    # Manager resolution
    # ------------------------------------------------------------------

    def _manager_name_from_stage(self, stage: str) -> str:
        if "_" not in stage:
            raise ValueError(f"Invalid stage '{stage}'. Expected '{{manager_name}}_{{function_name}}'.")
        return stage.split("_", 1)[0]

    def get_mode_for_manager(self, manager_name: str) -> ManagerCallMode:
        """Return the resolved execution mode for the given manager."""
        default_key = next(iter(self.DEFAULT_CONFIG))
        mode_value = self._cfg.get(manager_name, self._cfg[default_key])
        return ManagerCallMode(mode_value)

    def resolve_manager_class(self, manager_name: str) -> type:
        """Import and return the manager class for the configured mode."""
        module_name = (
            "isaaclab.managers"
            if self.get_mode_for_manager(manager_name) == ManagerCallMode.STABLE
            else "isaaclab_experimental.managers"
        )
        module = importlib.import_module(module_name)
        if not hasattr(module, manager_name):
            raise AttributeError(f"Manager '{manager_name}' not found in module '{module_name}'.")
        return getattr(module, manager_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_calls(self, calls: Sequence[dict[str, Any]]) -> Any:
        result = None
        for spec in calls:
            fn = spec["fn"]
            fn_args = spec.get("args", ())
            fn_kwargs = spec.get("kwargs", {})
            result = fn(*fn_args, **fn_kwargs)
        return result

    def _wp_capture_or_launch(self, stage: str, calls: Sequence[dict[str, Any]]) -> None:
        """Capture Warp CUDA graph for a stage on first call, then replay."""
        graph = self._wp_graphs.get(stage)
        if graph is None:
            with wp.ScopedCapture() as capture:
                for spec in calls:
                    fn = spec["fn"]
                    fn_args = spec.get("args", ())
                    fn_kwargs = spec.get("kwargs", {})
                    fn(*fn_args, **fn_kwargs)
            graph = capture.graph
            self._wp_graphs[stage] = graph
        wp.capture_launch(graph)

    def _load_cfg(self, cfg_source: str | None) -> dict[str, int]:
        if cfg_source is not None and not isinstance(cfg_source, str):
            raise TypeError(f"cfg_source must be a string or None, got: {type(cfg_source)}")
        if cfg_source is None or cfg_source.strip() == "":
            return dict(self.DEFAULT_CONFIG)

        parsed = json.loads(cfg_source)
        if not isinstance(parsed, dict):
            raise TypeError("manager_call_config must decode to a dict.")

        cfg = dict(parsed)
        if self.DEFAULT_KEY not in cfg:
            cfg[self.DEFAULT_KEY] = self.DEFAULT_CONFIG[self.DEFAULT_KEY]

        # validation
        for manager_name, mode_value in cfg.items():
            if not isinstance(mode_value, int):
                raise TypeError(
                    f"manager_call_config value for '{manager_name}' must be int (0/1/2), got: {type(mode_value)}"
                )
            try:
                ManagerCallMode(mode_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid manager_call_config value for '{manager_name}': {mode_value}. Expected 0/1/2."
                ) from exc

        # Apply MAX_MODE_OVERRIDES: bake caps into the resolved config so
        # get_mode_for_manager never needs per-call branching.
        default_mode = cfg[self.DEFAULT_KEY]
        for name, max_mode in self.MAX_MODE_OVERRIDES.items():
            resolved = cfg.get(name, default_mode)
            if resolved > max_mode:
                cfg[name] = max_mode

        return cfg
