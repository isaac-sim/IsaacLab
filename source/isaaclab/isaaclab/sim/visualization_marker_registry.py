# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Registry for simulation-scoped visualization marker state."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class VisualizationMarkerRegistry:
    """Tracks visualization marker callbacks and active marker groups."""

    def __init__(self):
        self._callbacks: dict[str, Callable[[Any], None]] = {}
        self._groups: dict[str, Any] = {}

    def add_callback(self, name: str, callback: Callable[[Any], None]) -> str:
        """Register a callback invoked before marker-capable visualizers step each render tick."""
        self._callbacks[name] = callback
        return name

    def remove_callback(self, callback_id: str) -> None:
        """Remove a visualization marker callback if it exists."""
        self._callbacks.pop(callback_id, None)

    def dispatch_callbacks(self, event: Any = None) -> None:
        """Invoke all registered visualization marker callbacks."""
        for callback in self._callbacks.values():
            callback(event)

    def set_group(self, group_id: str, state: Any) -> None:
        """Set or replace one visualization marker group state."""
        self._groups[group_id] = state

    def remove_group(self, group_id: str) -> None:
        """Remove one visualization marker group state if present."""
        self._groups.pop(group_id, None)

    def get_groups(self) -> dict[str, Any]:
        """Return all active visualization marker groups."""
        return self._groups
