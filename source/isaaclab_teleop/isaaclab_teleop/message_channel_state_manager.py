# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Message-channel-based teleop state manager for TeleopCore's teleop_control_pipeline."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from isaacteleop.retargeting_engine.interface.execution_events import ExecutionEvents, ExecutionState
from isaacteleop.teleop_session_manager.teleop_state_manager_retargeter import TeleopStateManager

from .control_events import ControlEvents

if TYPE_CHECKING:
    from isaacteleop.retargeting_engine.interface import RetargeterIOType
    from isaacteleop.retargeting_engine.interface.retargeter_core_types import ComputeContext, RetargeterIO

_COMMAND_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\breset\b", re.IGNORECASE), "reset"),
    (re.compile(r"\bstop\b", re.IGNORECASE), "stop"),
    (re.compile(r"\bstart\b", re.IGNORECASE), "start"),
]
"""Ordered patterns for classifying a command string.

``reset`` is checked first so that a hypothetical payload containing
both "reset" and "start" is treated as a reset (the more destructive
operation wins).  ``stop`` precedes ``start`` for the same reason.
"""


class MessageChannelTeleopStateManager(TeleopStateManager):
    """Teleop state manager driven by message channel payloads.

    Consumes the ``messages_tracked`` output of a
    :class:`~isaacteleop.retargeting_engine.deviceio_source_nodes.MessageChannelSource`,
    parses the JSON/text payloads (``"start"``, ``"stop"``, ``"reset"``), and
    produces the ``teleop_state`` (one-hot) and ``reset_event`` (bool pulse)
    outputs required by TeleopCore's ``teleop_control_pipeline`` contract.

    Payload formats supported (same as the legacy carb message bus):

    1. **JSON (Quest client format)**::

           {"type": "teleop_command", "message": {"command": "start teleop"}}

    2. **Plain text (fallback)**: raw UTF-8 string matched by word boundary
       (``"start"``, ``"stop"``, ``"reset"``).

    The state machine maps commands to :class:`ExecutionState` as follows:

    * ``"start"`` -> :attr:`ExecutionState.RUNNING`
    * ``"stop"``  -> :attr:`ExecutionState.PAUSED`
    * ``"reset"`` -> no state change, emits ``reset=True`` pulse
    """

    INPUT_MESSAGES = "messages_tracked"

    def __init__(self, name: str) -> None:
        self._state = ExecutionState.STOPPED
        self._control_events = ControlEvents()
        super().__init__(name=name)

    @property
    def last_control_events(self) -> ControlEvents:
        """The most recent :class:`ControlEvents` derived from message payloads."""
        return self._control_events

    def input_spec(self) -> RetargeterIOType:
        from isaacteleop.retargeting_engine.deviceio_source_nodes.deviceio_tensor_types import (
            MessageChannelMessagesTrackedGroup,
        )

        return {
            self.INPUT_MESSAGES: MessageChannelMessagesTrackedGroup(),
        }

    def _compute_execution_events(
        self,
        inputs: RetargeterIO,
        context: ComputeContext,
    ) -> ExecutionEvents:
        del context

        reset = False
        messages_tracked = inputs[self.INPUT_MESSAGES][0]

        data = getattr(messages_tracked, "data", None)
        if data:
            for message in data:
                payload = getattr(message, "payload", None)
                if payload is None:
                    continue
                try:
                    text = bytes(payload).decode("utf-8")
                except (UnicodeDecodeError, TypeError):
                    continue

                command = _extract_command(text)
                if command is None:
                    continue

                kind = _classify_command(command)
                if kind == "start":
                    self._state = ExecutionState.RUNNING
                elif kind == "stop":
                    self._state = ExecutionState.PAUSED
                elif kind == "reset":
                    reset = True

        is_active: bool | None
        if self._state == ExecutionState.RUNNING:
            is_active = True
        elif self._state == ExecutionState.PAUSED:
            is_active = False
        else:
            is_active = None

        self._control_events = ControlEvents(is_active=is_active, should_reset=reset)

        return ExecutionEvents(reset=reset, execution_state=self._state)


def _classify_command(text: str) -> str | None:
    """Return ``"start"``, ``"stop"``, ``"reset"``, or ``None``.

    Uses word-boundary matching so that e.g. ``"stop_and_restart"``
    matches ``"stop"`` (not ``"start"``).
    """
    for pattern, label in _COMMAND_PATTERNS:
        if pattern.search(text):
            return label
    return None


def _extract_command(text: str) -> str | None:
    """Extract the command string from a JSON or plain-text payload.

    Tries JSON parsing first (Quest client format) and falls back to the
    raw text for plain-string payloads.  Non-string JSON scalars (numbers,
    arrays, booleans) are discarded.
    """
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text

    if not isinstance(obj, dict):
        return None
    if obj.get("type") != "teleop_command":
        return None

    msg = obj.get("message")
    if isinstance(msg, dict):
        return msg.get("command", "")
    if isinstance(msg, str):
        return msg
    return None
