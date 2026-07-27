# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Message-channel payload parser for TeleopCore's teleop_control_pipeline.

Provides :class:`TeleopMessageProcessor`, a lightweight
:class:`~isaacteleop.retargeting_engine.interface.BaseRetargeter` that
converts message-channel payloads into boolean pulse signals suitable for
:class:`~isaacteleop.teleop_session_manager.DefaultTeleopStateManager`.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from isaacteleop.retargeting_engine.interface import BaseRetargeter, RetargeterIOType

if TYPE_CHECKING:
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

# Shadow states mirroring DefaultTeleopStateManager's ExecutionState.
_STOPPED = "stopped"
_PAUSED = "paused"
_RUNNING = "running"

# DefaultTeleopStateManager cycles states on run_toggle rising edges:
#   STOPPED -> PAUSED -> RUNNING -> PAUSED -> RUNNING -> ...
# To map imperative "start" (= go to RUNNING) and "stop" (= go to PAUSED)
# we emit the right number of toggle edges based on predicted state.
_START_TOGGLE_SEQUENCES: dict[str, list[bool]] = {
    _STOPPED: [True, False, True],  # 2 edges: STOPPED -> PAUSED -> RUNNING
    _PAUSED: [True],  # 1 edge:  PAUSED -> RUNNING
    _RUNNING: [],  # already running
}
_STOP_TOGGLE_SEQUENCES: dict[str, list[bool]] = {
    _RUNNING: [True],  # 1 edge:  RUNNING -> PAUSED
    _PAUSED: [],  # already paused
    _STOPPED: [],  # already stopped
}
# Shadow state advances on each rising edge (True after False).
_TOGGLE_TRANSITIONS: dict[str, str] = {
    _STOPPED: _PAUSED,
    _PAUSED: _RUNNING,
    _RUNNING: _PAUSED,
}


class TeleopMessageProcessor(BaseRetargeter):
    """Parse message-channel payloads into boolean control signals.

    Consumes the ``messages_tracked`` output of a
    :class:`~isaacteleop.retargeting_engine.deviceio_source_nodes.MessageChannelSource`
    and produces three boolean pulse outputs that drive
    :class:`~isaacteleop.teleop_session_manager.DefaultTeleopStateManager`:

    * ``run_toggle`` -- pulsed ``True`` on rising edges; the number of
      edges depends on the target state (e.g. ``"start"`` from STOPPED
      emits two edges over three frames: STOPPED -> PAUSED -> RUNNING).
    * ``kill`` -- always ``False`` (reserved for fail-safe; ``"stop"``
      uses ``run_toggle`` to reach PAUSED instead of STOPPED).
    * ``reset`` -- pulsed ``True`` for one frame on ``"reset"``.

    An *operator* reset also pauses a running session (emits the RUNNING ->
    PAUSED toggle edge) so teleop lands in a non-driving state, awaiting an
    explicit start. Channel ``"reset"`` messages (from the XR client) are always
    treated as operator resets. A *host* reset (e.g. the environment auto-reset
    after task success) pulses ``reset`` without pausing so recording continues
    into the next episode -- see :meth:`inject_reset`.

    The processor maintains a *shadow state* that mirrors
    ``DefaultTeleopStateManager``'s internal state so it can emit the
    correct toggle sequence for imperative commands.

    Payload formats supported:

    1. **JSON (Quest client format)**::

           {"type": "teleop_command", "message": {"command": "start teleop"}}

    2. **Plain text (fallback)**: raw UTF-8 string matched by word boundary
       (``"start"``, ``"stop"``, ``"reset"``).
    """

    INPUT_MESSAGES = "messages_tracked"

    def __init__(self, name: str) -> None:
        self._inject_reset_pending = False
        self._inject_reset_pause = False
        self._shadow_state = _STOPPED
        self._run_toggle_queue: list[bool] = []
        self._prev_toggle_output = False
        self._pending_commands: list[str] = []
        super().__init__(name=name)

    def inject_reset(self, pause: bool = False) -> None:
        """Schedule a reset pulse on the next pipeline step.

        The ``reset`` output will be ``True`` for exactly one frame, then
        automatically cleared.

        Args:
            pause: When ``True``, also pause a running session (RUNNING ->
                PAUSED), cancelling any in-flight toggle -- the behavior for an
                *operator* reset (keyboard ``R``). Defaults to ``False`` for a
                *host* reset (e.g. environment auto-reset after task success): a
                pure pulse that keeps the session running into the next episode.
        """
        self._inject_reset_pending = True
        if pause:
            self._inject_reset_pause = True

    def inject_command(self, command: str) -> None:
        """Inject a local teleop command as if it arrived on the control channel.

        Lets a host binding (e.g. a keyboard key) drive
        :class:`~isaacteleop.teleop_session_manager.DefaultTeleopStateManager`
        without an XR client: ``"start"`` / ``"stop"`` enqueue the matching
        ``run_toggle`` sequence and ``"reset"`` pulses the reset output. Natural
        language variants (e.g. ``"start teleop"``) are accepted -- the same
        classifier the message channel uses is applied. Unrecognized commands are
        ignored. The command is queued and applied on the next pipeline step, so
        this is safe to call from another thread (mirrors :meth:`inject_reset`).

        Args:
            command: A teleop command string (``"start"``, ``"stop"``, or ``"reset"``).
        """
        kind = _classify_command(command)
        if kind is not None:
            self._pending_commands.append(kind)

    def _apply_command_kind(self, kind: str | None) -> bool:
        """Apply a classified command to the ``run_toggle`` queue.

        Args:
            kind: ``"start"``, ``"stop"``, ``"reset"``, or ``None``.

        Returns:
            ``True`` if the command requests a reset pulse this frame.
        """
        if kind == "start" and not self._run_toggle_queue:
            self._run_toggle_queue = self._make_toggle_sequence(_START_TOGGLE_SEQUENCES[self._shadow_state])
        elif kind == "stop" and not self._run_toggle_queue:
            self._run_toggle_queue = self._make_toggle_sequence(_STOP_TOGGLE_SEQUENCES[self._shadow_state])
        elif kind == "reset":
            return True
        return False

    def _make_toggle_sequence(self, base_sequence: list[bool]) -> list[bool]:
        """Prepend a ``False`` frame if needed to guarantee a clean rising edge.

        ``DefaultTeleopStateManager`` uses edge detection
        (``pressed and not prev_pressed``), so emitting ``True`` when the
        previous output was already ``True`` would not trigger a state
        transition.  This method prepends ``False`` when necessary.
        """
        if not base_sequence:
            return []
        seq = list(base_sequence)
        if self._prev_toggle_output:
            seq.insert(0, False)
        return seq

    def input_spec(self) -> RetargeterIOType:
        from isaacteleop.retargeting_engine.deviceio_source_nodes.deviceio_tensor_types import (
            MessageChannelMessagesTrackedGroup,
        )

        return {self.INPUT_MESSAGES: MessageChannelMessagesTrackedGroup()}

    def output_spec(self) -> RetargeterIOType:
        from isaacteleop.teleop_session_manager.teleop_state_manager_types import bool_signal

        return {
            "run_toggle": bool_signal("run_toggle"),
            "kill": bool_signal("kill"),
            "reset": bool_signal("reset"),
        }

    def _compute_fn(
        self,
        inputs: RetargeterIO,
        outputs: RetargeterIO,
        context: ComputeContext,
    ) -> None:
        del context

        reset = self._inject_reset_pending
        # ``pause_on_reset`` gates the RUNNING -> PAUSED edge below. Injected resets
        # carry their own intent (``inject_reset(pause=...)``); channel and inject_command
        # resets are operator-initiated and always pause.
        pause_on_reset = self._inject_reset_pause
        self._inject_reset_pending = False
        self._inject_reset_pause = False

        # Parse incoming messages and enqueue toggle sequences.
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

                if self._apply_command_kind(_classify_command(command)):
                    reset = True
                    pause_on_reset = True

        # Apply locally injected commands (e.g. keyboard bindings) identically to
        # channel messages, so a host can start/stop/reset without an XR client.
        injected_commands = self._pending_commands
        self._pending_commands = []
        for injected_kind in injected_commands:
            if self._apply_command_kind(injected_kind):
                reset = True
                pause_on_reset = True

        # An operator reset pauses the session: replace the toggle queue with the stop
        # toggle for the current shadow state (RUNNING -> PAUSED; a no-op from PAUSED/
        # STOPPED), which also cancels any in-flight toggle. ``run_toggle`` and ``reset``
        # are independent inputs to DefaultTeleopStateManager, so the pause edge and the
        # reset pulse apply together. A host reset (e.g. task-success auto-reset) is a pure
        # pulse -- it leaves the run state and any pending toggle untouched -- so a non-XR
        # session that reset()s then request_start()s at startup still reaches RUNNING.
        if reset and pause_on_reset:
            self._run_toggle_queue = self._make_toggle_sequence(_STOP_TOGGLE_SEQUENCES[self._shadow_state])

        # Drain the toggle queue (one value per frame).
        if self._run_toggle_queue:
            run_toggle = self._run_toggle_queue.pop(0)
        else:
            run_toggle = False

        # Advance shadow state on rising edges (matches DefaultTeleopStateManager's
        # edge detection: ``pressed and not prev_pressed``).
        if run_toggle and not self._prev_toggle_output:
            self._shadow_state = _TOGGLE_TRANSITIONS[self._shadow_state]
        self._prev_toggle_output = run_toggle

        outputs["run_toggle"][0] = run_toggle
        outputs["kill"][0] = False
        outputs["reset"][0] = reset


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
