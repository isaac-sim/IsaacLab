# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleop control events dataclass, polling helper, and well-known channel UUID."""

from __future__ import annotations

import dataclasses
import uuid
from typing import Protocol, runtime_checkable

TELEOP_CONTROL_CHANNEL_UUID: bytes = uuid.uuid5(uuid.NAMESPACE_DNS, "teleop_command").bytes
"""Well-known 16-byte UUID for the teleop control message channel.

Derived deterministically as ``uuid5(NAMESPACE_DNS, "teleop_command")``
so that both the Isaac Lab server and the Quest client can independently
compute the same channel identifier from the string ``"teleop_command"``.

Pass this value as :attr:`~isaaclab_teleop.IsaacTeleopCfg.control_channel_uuid`
when configuring a teleop session with message-channel-based control.
"""


@dataclasses.dataclass(frozen=True)
class ControlEvents:
    """Result of :func:`poll_control_events`.

    Attributes:
        is_active: ``True`` when the teleop state machine is in RUNNING,
            ``False`` when PAUSED or STOPPED, or ``None`` when no control
            channel is configured (callers should leave their own active
            flag unchanged).
        should_reset: ``True`` when a reset was triggered this frame.
    """

    is_active: bool | None = None
    should_reset: bool = False


_NO_OP_EVENTS = ControlEvents()
"""Shared immutable sentinel returned when no control channel is active."""


@runtime_checkable
class SupportsControlEvents(Protocol):
    """Duck type for teleop devices that expose control events."""

    @property
    def last_control_events(self) -> ControlEvents: ...


def poll_control_events(teleop_interface: SupportsControlEvents | object) -> ControlEvents:
    """Poll control events from any teleop interface.

    Safe to call with any device type (keyboard, spacemouse, etc.).
    Devices that do not expose the message-channel protocol return
    a no-op :class:`ControlEvents`.

    Args:
        teleop_interface: The teleop device to poll.  Devices implementing
            :class:`SupportsControlEvents` provide full type safety; other
            devices are handled gracefully via duck typing.

    Returns:
        A :class:`ControlEvents` with the latest start/stop and reset
        signals.
    """
    events = getattr(teleop_interface, "last_control_events", None)
    if events is None:
        return _NO_OP_EVENTS
    if isinstance(events, ControlEvents):
        return events
    return ControlEvents(
        is_active=getattr(events, "is_active", None),
        should_reset=getattr(events, "should_reset", False),
    )
