# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal Newton-viewer selector for conveyor transfer goals."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from .conveyor_geometry import CUBE_COLORS
from .mdp.reset_events import LEFT_SIDE

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _mix_color(
    color: tuple[float, float, float],
    target: tuple[float, float, float],
    amount: float,
) -> tuple[float, float, float]:
    """Linearly mix two RGB colors."""
    return tuple(value + amount * (target_value - value) for value, target_value in zip(color, target, strict=True))


class ConveyorGoalSelector:
    """Render four color-matched cube buttons for one displayed environment."""

    def __init__(self, env: ManagerBasedRLEnv, env_id: int) -> None:
        """Create a selector bound to one vectorized environment index."""
        if not 0 <= env_id < env.num_envs:
            raise IndexError(f"Conveyor selector environment {env_id} is out of range.")
        self._env = env
        self._env_id = env_id
        self._command = env.command_manager.get_term("transfer")
        self._target_cube_id = 0
        self._source_side_id = LEFT_SIDE
        self._last_refresh_time = float("-inf")

    def render(self, imgui: Any) -> None:
        """Draw the selector and publish a new transfer command when clicked."""
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header(f"Transfer Goal · Env {self._env_id}"):
            return
        imgui.separator()

        if not self._refresh_command():
            imgui.text_disabled("Waiting for transfer state...")
            return

        style = imgui.get_style()
        available_width = float(imgui.get_content_region_avail().x)
        spacing = float(style.item_spacing.x)
        button_width = max(36.0, (available_width - spacing * (len(CUBE_COLORS) - 1)) / len(CUBE_COLORS))
        button_height = max(34.0, min(46.0, button_width * 0.72))

        selected_cube_id: int | None = None
        for cube_id, color in enumerate(CUBE_COLORS):
            selected = cube_id == self._target_cube_id
            text_color = (0.05, 0.05, 0.05) if sum(color) > 1.45 else (1.0, 1.0, 1.0)
            border_color = (1.0, 0.88, 0.20) if selected else (0.12, 0.12, 0.12)
            label = f"{cube_id + 1}##conveyor_goal_{cube_id}"

            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*color, 1.0))
            imgui.push_style_color(
                imgui.Col_.button_hovered,
                imgui.ImVec4(*_mix_color(color, (1.0, 1.0, 1.0), 0.18), 1.0),
            )
            imgui.push_style_color(
                imgui.Col_.button_active,
                imgui.ImVec4(*_mix_color(color, (0.0, 0.0, 0.0), 0.12), 1.0),
            )
            imgui.push_style_color(imgui.Col_.border, imgui.ImVec4(*border_color, 1.0))
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*text_color, 1.0))
            imgui.push_style_var(imgui.StyleVar_.frame_border_size, 3.0 if selected else 1.0)
            imgui.push_style_var(imgui.StyleVar_.frame_rounding, 5.0)

            if imgui.button(label, imgui.ImVec2(button_width, button_height)):
                selected_cube_id = cube_id

            imgui.pop_style_var(2)
            imgui.pop_style_color(5)
            if cube_id + 1 < len(CUBE_COLORS):
                imgui.same_line()

        if selected_cube_id is not None and selected_cube_id != self._target_cube_id:
            self._command.set_goal(selected_cube_id, env_ids=(self._env_id,))
            self._refresh_command(force=True)

        source_name = "Left" if self._source_side_id == LEFT_SIDE else "Right"
        target_name = "Right" if self._source_side_id == LEFT_SIDE else "Left"
        imgui.text(f"Cube {self._target_cube_id + 1}: {source_name} -> {target_name}")
        imgui.text_disabled("Click a color to change the next transfer.")

    def _refresh_command(self, force: bool = False) -> bool:
        """Refresh the small host-side UI cache at most ten times per second."""
        current_time = time.monotonic()
        if force or current_time - self._last_refresh_time >= 0.1:
            self._target_cube_id = int(self._command.target_cube_ids[self._env_id].item())
            self._source_side_id = int(self._command.source_side_ids[self._env_id].item())
            self._last_refresh_time = current_time
        return True
