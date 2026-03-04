# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleop command handling for IsaacTeleop-based teleoperation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import carb

logger = logging.getLogger(__name__)


class CommandHandler:
    """Handles teleop command callbacks and XR message bus events.

    This class is responsible for:

    1. Registering callbacks for teleop commands (START, STOP, RESET)
    2. Subscribing to the XR message bus for command events
    3. Dispatching callbacks when commands are received

    Teleop commands can be triggered via XR controller buttons or the
    message bus.  The handler normalizes command names (e.g. mapping
    ``"R"`` to ``"RESET"``) and dispatches to registered callbacks.
    """

    TELEOP_COMMAND_EVENT_TYPE = "teleop_command"

    def __init__(self, xr_core: Any | None = None, on_reset: Callable[[], None] | None = None):
        """Initialize the command handler.

        Args:
            xr_core: The XRCore singleton, or ``None`` if XR is not available.
                When provided, the handler subscribes to the message bus for
                teleop command events.
            on_reset: Optional hook called whenever a ``"reset"`` message-bus
                event is received, *in addition to* the user's RESET callback.
                This allows the device to perform internal reset actions (e.g.
                resetting the XR anchor) without coupling the handler to the
                anchor manager.
        """
        self._callbacks: dict[str, Callable] = {}
        self._on_reset = on_reset
        self._xr_core = xr_core
        self._vc_subscription = None

        if self._xr_core is not None:
            self._vc_subscription = self._xr_core.get_message_bus().create_subscription_to_pop_by_type(
                carb.events.type_from_string(self.TELEOP_COMMAND_EVENT_TYPE), self._on_teleop_command
            )

    @property
    def callbacks(self) -> dict[str, Callable]:
        """The registered callbacks dictionary (read-only view)."""
        return self._callbacks

    def add_callback(self, key: str, func: Callable) -> None:
        """Add a callback function for a teleop command.

        Args:
            key: The command type to bind to.  Valid values are
                ``"START"``, ``"STOP"``, ``"RESET"``, and ``"R"``
                (``"R"`` is mapped to ``"RESET"`` for compatibility).
            func: The function to call when the command is received.
                Should take no arguments.
        """
        # Map "R" to "RESET" for compatibility with existing scripts
        if key == "R":
            key = "RESET"
        self._callbacks[key] = func

    def fire(self, command: str) -> None:
        """Dispatch a named command callback if registered.

        Args:
            command: The command name (e.g. ``"START"``, ``"STOP"``, ``"RESET"``).
        """
        if command in self._callbacks:
            self._callbacks[command]()

    def _on_teleop_command(self, event: carb.events.IEvent) -> None:
        """Handle teleop command events from the message bus."""
        msg = event.payload.get("message", "")

        if "start" in msg:
            self.fire("START")
        elif "stop" in msg:
            self.fire("STOP")
        elif "reset" in msg:
            self.fire("RESET")
            if self._on_reset is not None:
                self._on_reset()

    def cleanup(self) -> None:
        """Release event subscriptions."""
        self._vc_subscription = None
