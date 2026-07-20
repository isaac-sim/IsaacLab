# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager call switch for routing manager stage calls through stable/warp/captured paths."""

from __future__ import annotations

import importlib
import json
import logging
import os
import warnings
from collections.abc import Callable
from copy import deepcopy
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from isaaclab.utils.timer import Timer

from isaaclab_experimental.utils.warp_graph_cache import WarpGraphCache

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnvCfg

logger = logging.getLogger(__name__)


class ManagerCallMode(IntEnum):
    """Execution mode for manager stage calls.

    * ``STABLE``  (0): Call stable Python manager implementations from :mod:`isaaclab.managers`.
      This mode is deprecated inside ``ManagerBasedEnvWarp``; use the stable environment instead.
    * ``WARP_NOT_CAPTURED`` (1): Call Warp-compatible implementations without CUDA graph capture.
    * ``WARP_CAPTURED`` (2): Call Warp implementations with CUDA graph capture/replay.
    """

    STABLE = 0
    WARP_NOT_CAPTURED = 1
    WARP_CAPTURED = 2


class ManagerCallSwitch:
    """Compatibility router for stable and Warp manager implementations.

    This temporary layer selects stable or Warp manager classes and forwards
    Warp stages to an environment-owned :class:`WarpGraphCache`. Execution and
    capture policy therefore remain reusable after mixed stable-manager routing
    is removed. Calls may optionally use a :class:`Timer` for profiling.
    """

    # Warp eager is the correctness-first default. Capture remains an explicit
    # optimization while stage state and pointer contracts are validated.
    DEFAULT_CONFIG: dict[str, int] = {"default": 1}
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
        "Scene",
    )
    # Scene stages remain eager until scene, sensor, and actuator graphability is
    # validated together. Warp-first execution does not depend on that later step.
    MAX_MODE_OVERRIDES: dict[str, int] = {"Scene": ManagerCallMode.WARP_NOT_CAPTURED}

    ENV_VAR = "MANAGER_CALL_CONFIG"
    """Environment variable name for the JSON config string.

    Example usage::

        MANAGER_CALL_CONFIG='{"RewardManager": 0, "default": 1}' python train.py ...
    """

    def __init__(
        self,
        cfg_source: dict[str, int] | str | None = None,
        *,
        max_modes: dict[str, int] | None = None,
        graph_cache: WarpGraphCache | None = None,
    ):
        # The environment normally owns this durable executor. Creating one here
        # keeps the compatibility router independently usable in tests and tools.
        self._graph_cache = graph_cache if graph_cache is not None else WarpGraphCache(enabled=True)
        # Merge caller-supplied max_modes with the class-level MAX_MODE_OVERRIDES.
        self._max_modes = dict(self.MAX_MODE_OVERRIDES)
        if max_modes is not None:
            self._max_modes.update(max_modes)
        # Resolve config: prefer explicit cfg_source, fall back to env var.
        if cfg_source is None:
            cfg_source = os.environ.get(self.ENV_VAR)
        self._cfg = self._load_cfg(cfg_source)
        print("[INFO] ManagerCallSwitch configuration:")
        print(f"  - {self.DEFAULT_KEY}: {self._cfg[self.DEFAULT_KEY]}")
        for manager_name in self.MANAGER_NAMES:
            mode = int(self.get_mode_for_manager(manager_name))
            cap = self._max_modes.get(manager_name)
            cap_str = f" (cap={cap})" if cap is not None else ""
            print(f"  - {manager_name}: {mode}{cap_str}")

    # ------------------------------------------------------------------
    # Graph management
    # ------------------------------------------------------------------

    def invalidate_graphs(self) -> None:
        """Invalidate cached capture graphs and their cached return values."""
        self._graph_cache.invalidate()

    def apply_term_cfg_profile(self, cfg: ManagerBasedEnvCfg) -> None:
        """Apply the deprecated mixed stable-manager configuration profile.

        This compatibility hook is intentionally isolated from the Warp
        environment. It can be removed together with stable manager routing
        once all task configurations use the Warp frontend natively.

        Args:
            cfg: Experimental environment configuration to update in place.
        """
        manager_to_cfg_attr = {
            "ActionManager": "actions",
            "ObservationManager": "observations",
            "EventManager": "events",
            "RecorderManager": "recorders",
            "CommandManager": "commands",
            "TerminationManager": "terminations",
            "RewardManager": "rewards",
            "CurriculumManager": "curriculum",
        }
        stable_managers = [
            name for name in manager_to_cfg_attr if self.get_mode_for_manager(name) == ManagerCallMode.STABLE
        ]
        if not stable_managers:
            return

        warnings.warn(
            "Selecting STABLE managers inside ManagerBasedEnvWarp is deprecated. Use ManagerBasedEnv for Torch "
            "managers or WARP_NOT_CAPTURED for the Warp frontend.",
            DeprecationWarning,
            stacklevel=2,
        )
        stable_cfg = self._resolve_stable_cfg_counterpart(cfg)
        if stable_cfg is None:
            logger.warning(
                "Stable managers requested (%s), but no stable cfg counterpart could be resolved. Keeping "
                "experimental term configs.",
                ", ".join(stable_managers),
            )
            return

        for manager_name, cfg_attr in manager_to_cfg_attr.items():
            if self.get_mode_for_manager(manager_name) != ManagerCallMode.STABLE:
                continue
            if hasattr(cfg, cfg_attr) and hasattr(stable_cfg, cfg_attr):
                setattr(cfg, cfg_attr, deepcopy(getattr(stable_cfg, cfg_attr)))

    # ------------------------------------------------------------------
    # Stage dispatch
    # ------------------------------------------------------------------

    def call(
        self,
        stage: str,
        fn: Callable[..., Any],
        /,
        *args: Any,
        _output: Callable[[Any], Any] | None = None,
        _timer: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run a Warp frontend stage eagerly or through its cached CUDA graph.

        The call site always supplies the same callable and arguments. The
        configured manager mode only controls how that call is executed.

        Args:
            stage: Stage identifier in the form ``"ManagerName_function_name"``.
            fn: Callable implementing the stage.
            *args: Positional arguments forwarded to :paramref:`fn`.
            _output: Optional transform applied to the stage result after execution.
            _timer: Whether to time the stage.
            **kwargs: Keyword arguments forwarded to :paramref:`fn`.

        Returns:
            The stage result, optionally transformed by :paramref:`_output`.
        """
        with Timer(name=stage, msg=f"{stage} took:", enable=_timer, time_unit="us"):
            manager_name = self._manager_name_from_stage(stage)
            mode = self.get_mode_for_manager(manager_name)
            result = self._graph_cache.call(
                stage,
                fn,
                args=args,
                kwargs=kwargs,
                capture=mode == ManagerCallMode.WARP_CAPTURED,
                group=manager_name,
            )
        return _output(result) if _output is not None else result

    def call_stage(
        self,
        *,
        stage: str,
        warp_call: dict[str, Any],
        stable_call: dict[str, Any] | None = None,
        timer: bool = False,
    ) -> Any:
        """Run the stage according to configured mode, optionally wrapped in a :class:`Timer`.

        A call spec dict supports the following keys:

        * ``fn`` (required): The callable to invoke.
        * ``args`` (optional): Positional arguments tuple.
        * ``kwargs`` (optional): Keyword arguments dict.
        * ``output`` (optional): A ``Callable[[Any], Any]`` that transforms the raw
          return value into the final output. For captured stages the raw value is
          ``None``. When omitted, the raw return value is used as-is.

        Args:
            stage: Stage identifier in the form ``"ManagerName_function_name"``.
            warp_call: Call spec for the warp path (eager or captured).
            stable_call: Call spec for the stable (torch) path. Defaults to ``None``.
            timer: Whether to wrap execution in a :class:`Timer`. Defaults to ``False``
                (controlled by the global :attr:`Timer.enable` class-level toggle).
                Pass a module-level flag like ``TIMER_ENABLED_STEP`` to make timing
                conditional on that flag.

        Returns:
            The (possibly transformed) return value of the stage.
        """
        mode = self.get_mode_for_manager(self._manager_name_from_stage(stage))
        if mode == ManagerCallMode.STABLE:
            if stable_call is None:
                raise ValueError(f"Stage '{stage}' is configured as STABLE (mode=0) but no stable_call was provided.")
            call = stable_call
        else:
            call = warp_call
        return self.call(
            stage,
            call["fn"],
            *call.get("args", ()),
            _output=call.get("output"),
            _timer=timer,
            **call.get("kwargs", {}),
        )

    # ------------------------------------------------------------------
    # Manager resolution
    # ------------------------------------------------------------------

    def _manager_name_from_stage(self, stage: str) -> str:
        if "_" not in stage:
            raise ValueError(f"Invalid stage '{stage}'. Expected '{{manager_name}}_{{function_name}}'.")
        return stage.split("_", 1)[0]

    def get_mode_for_manager(self, manager_name: str) -> ManagerCallMode:
        """Return the resolved execution mode for the given manager.

        Looks up the manager in the config dict, falls back to the default,
        then caps by :attr:`_max_modes` (static overrides + dynamic registrations).
        """
        default_key = next(iter(self.DEFAULT_CONFIG))
        mode_value = self._cfg.get(manager_name, self._cfg[default_key])
        cap = self._max_modes.get(manager_name)
        if cap is not None:
            mode_value = min(mode_value, cap)
        if mode_value == ManagerCallMode.WARP_CAPTURED and not self._graph_cache.is_capturable(manager_name):
            mode_value = ManagerCallMode.WARP_NOT_CAPTURED
        return ManagerCallMode(mode_value)

    def resolve_manager_class(self, manager_name: str, mode_override: ManagerCallMode | int | None = None) -> type:
        """Import and return the manager class for the configured mode."""
        mode = self.get_mode_for_manager(manager_name) if mode_override is None else ManagerCallMode(mode_override)
        module_name = "isaaclab.managers" if mode == ManagerCallMode.STABLE else "isaaclab_experimental.managers"
        module = importlib.import_module(module_name)
        if not hasattr(module, manager_name):
            raise AttributeError(f"Manager '{manager_name}' not found in module '{module_name}'.")
        return getattr(module, manager_name)

    def register_manager_capturability(self, manager_name: str, capturable: bool) -> None:
        """Register that a manager has non-capturable terms, capping its mode.

        Called by :class:`ManagerBase` during term preparation when a term
        is decorated with ``@warp_capturable(False)``.
        """
        self._graph_cache.register_capturability(manager_name, capturable)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_cfg(self, cfg_source: dict[str, int] | str | None) -> dict[str, int]:
        if cfg_source is None:
            cfg = dict(self.DEFAULT_CONFIG)
        elif isinstance(cfg_source, dict):
            cfg = dict(cfg_source)
            if self.DEFAULT_KEY not in cfg:
                cfg[self.DEFAULT_KEY] = self.DEFAULT_CONFIG[self.DEFAULT_KEY]
        elif isinstance(cfg_source, str):
            if cfg_source.strip() == "":
                cfg = dict(self.DEFAULT_CONFIG)
            else:
                parsed = json.loads(cfg_source)
                if not isinstance(parsed, dict):
                    raise TypeError("manager_call_config must decode to a dict.")
                cfg = dict(parsed)
                if self.DEFAULT_KEY not in cfg:
                    cfg[self.DEFAULT_KEY] = self.DEFAULT_CONFIG[self.DEFAULT_KEY]
        else:
            raise TypeError(f"cfg_source must be a dict, string, or None, got: {type(cfg_source)}")

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

        # Apply max mode caps: bake caps into the resolved config so
        # get_mode_for_manager never needs per-call branching.
        default_mode = cfg[self.DEFAULT_KEY]
        for name, max_mode in self._max_modes.items():
            resolved = cfg.get(name, default_mode)
            if resolved > max_mode:
                cfg[name] = max_mode

        return cfg

    @staticmethod
    def _resolve_stable_cfg_counterpart(cfg: ManagerBasedEnvCfg) -> ManagerBasedEnvCfg | None:
        """Resolve the legacy stable task configuration counterpart."""
        cfg_cls = cfg.__class__
        cfg_module_name = cfg_cls.__module__
        if "isaaclab_tasks_experimental" not in cfg_module_name:
            return None

        stable_module_name = cfg_module_name.replace("isaaclab_tasks_experimental", "isaaclab_tasks", 1)
        try:
            stable_module = importlib.import_module(stable_module_name)
        except Exception as exc:
            logger.warning(
                "Failed to import stable task cfg module '%s' for manager_call_config stable mode: %s",
                stable_module_name,
                exc,
            )
            return None

        stable_cfg_cls = getattr(stable_module, cfg_cls.__name__, None)
        if stable_cfg_cls is None:
            logger.warning("Stable task cfg class '%s' not found in module '%s'.", cfg_cls.__name__, stable_module_name)
            return None

        try:
            return stable_cfg_cls()
        except Exception as exc:
            logger.warning(
                "Failed to instantiate stable task cfg '%s.%s': %s",
                stable_module_name,
                cfg_cls.__name__,
                exc,
            )
            return None
