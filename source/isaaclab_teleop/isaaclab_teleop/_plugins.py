# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Isaac Capture plugins that read the gamepad and the SpaceMouse for their teleop configs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaacteleop.teleop_session_manager import PluginConfig


def bundled_plugin_configs(name: str) -> list[PluginConfig]:
    """The Isaac Capture plugin ``name`` (``"gamepad"`` or ``"spacemouse"``), started with the session.

    It runs from the copy bundled with Isaac Capture and pushes to the collection the pipeline's
    source of the same name reads.
    """
    from isaacteleop.teleop_session_manager import BUNDLED_PLUGINS_DIR, PluginConfig

    return [PluginConfig(plugin_name=name, plugin_root_id=name, search_paths=[BUNDLED_PLUGINS_DIR], required=True)]
