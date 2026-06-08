# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleop command callback registry for IsaacTeleop-based teleoperation."""

from __future__ import annotations

from collections.abc import Callable


class CommandHandler:
    """Lightweight callback registry for teleop commands.

    Scripts can register callbacks for ``START``, ``STOP``, and ``RESET``
    commands via :meth:`add_callback`.  The callbacks are dispatched by
    :meth:`fire` when the corresponding command is received.

    Note:
        In the current architecture control signals arrive through
        TeleopCore's ``teleop_control_pipeline`` and are consumed via
        :func:`~isaaclab_teleop.poll_control_events`.  This registry is
        retained for backward compatibility with scripts that register
        callbacks before the pipeline-based path was introduced.
    """

    def __init__(self) -> None:
        self._callbacks: dict[str, Callable] = {}

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

    def cleanup(self) -> None:
        """Release resources (no-op; retained for API compatibility)."""
