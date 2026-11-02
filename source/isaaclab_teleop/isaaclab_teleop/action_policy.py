# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Task-owned action validation shared by live and replay."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .isaac_teleop_device import IsaacTeleopDevice


def prepare_teleop_action(
    env: Any,
    action: Any,
    conditioner: Callable[[Any, Any, dict[str, Any]], Any] | None,
    validator: Callable[[Any], str | None] | None,
    state: dict[str, Any],
) -> tuple[Any, str | None]:
    """Validate and condition one task-owned action without stepping physics."""
    issue = validator(action) if validator is not None else None
    if issue is None and conditioner is not None:
        action = conditioner(env, action, state)
        issue = validator(action) if validator is not None else None
    return action, issue


def initial_hand_tracking_issue(
    teleop_interface: IsaacTeleopDevice, initialized_sides: set[str], *, required: bool = True
) -> str | None:
    """Wait for initial hand presence without changing later tracking-loss behavior."""
    required_sides = {"left", "right"}
    if not required or required_sides.issubset(initialized_sides):
        return None
    presence = teleop_interface.hand_tracking_presence
    if presence is None:
        return "hand-tracking presence is unavailable"
    initialized_sides.update(side for side, present in zip(("left", "right"), presence, strict=True) if present)
    missing = sorted(required_sides - initialized_sides)
    return f"waiting for initial {'/'.join(missing)} hand tracking" if missing else None
