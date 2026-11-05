# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Capture input configuration for an environment teleoperation source."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.teleop_cfg import TeleopSourceCfg
from isaaclab.utils import configclass

from .isaac_teleop_cfg import CLOUDXR_STANDALONE_ENV, IsaacTeleopCfg

if TYPE_CHECKING:
    from .play_source import CaptureTeleopSource


@configclass
class CaptureTeleopSourceCfg(TeleopSourceCfg):
    """A desktop Capture pipeline controlling one environment in a non-Kit runtime.

    Configure ``device`` with a task-compatible pipeline, for example
    :func:`~isaaclab_teleop.se3_keyboard_teleop_cfg`. The source borrows the
    environment's visualizer windows. Keyboard, gamepad, and SpaceMouse input
    are supported; XR input and presentation require a future integration.
    Capture's standalone OpenXR transport does not require a headset.
    """

    class_type: type[CaptureTeleopSource] | str = "{DIR}.play_source:CaptureTeleopSource"

    device: IsaacTeleopCfg = MISSING
    """Capture pipeline whose flattened output uses the environment's action conventions."""

    cloudxr_env_file: str | None = CLOUDXR_STANDALONE_ENV
    """Runtime profile path. ``None`` uses an externally managed OpenXR runtime."""

    auto_launch_cloudxr: bool = True
    """Whether Capture starts the configured runtime when this source is activated."""

    start_active: bool = True
    """Start locally without waiting for an XR client's START command."""
