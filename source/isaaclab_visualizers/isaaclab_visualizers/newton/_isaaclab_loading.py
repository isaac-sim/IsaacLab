# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab's console greeting drawn directly in the native viewer."""

from __future__ import annotations

import math
import time
from typing import Any


def draw_isaaclab_logo(
    imgui: Any, draw_list: Any, x: float, y: float, size: float, elapsed: float = 0.0
) -> None:
    """Draw the CLI robot inside a square of *size* [px] at (*x*, *y*) [px].

    Antennas and eyes use the console greeting's NVIDIA green and cyan. The face follows the current
    text color so the same mark works in light and dark themes. A positive *elapsed* [s] animates it.
    """
    green = imgui.color_convert_float4_to_u32(imgui.ImVec4(118 / 255, 185 / 255, 0, 1))
    cyan = imgui.color_convert_float4_to_u32(imgui.ImVec4(77 / 255, 217 / 255, 232 / 255, 1))
    outline = imgui.get_color_u32(imgui.Col_.text)
    thickness = max(1.0, size * 0.024)
    lift = math.sin(elapsed * 2.0) * size * 0.012 if elapsed > 0 else 0.0
    y += lift

    for direction in (-1, 1):
        draw_list.add_line(
            imgui.ImVec2(x + size * (0.5 + direction * 0.17), y + size * 0.29),
            imgui.ImVec2(x + size * (0.5 + direction * 0.25), y + size * 0.10),
            green,
            thickness,
        )
    draw_list.add_rect(
        imgui.ImVec2(x + size * 0.12, y + size * 0.29),
        imgui.ImVec2(x + size * 0.88, y + size * 0.78),
        outline,
        rounding=size * 0.075,
        thickness=thickness,
    )
    blinking = elapsed > 0 and 3.78 < elapsed % 4.2 < 3.92
    for direction in (-1, 1):
        eye_x = x + size * (0.5 + direction * 0.18)
        eye_y = y + size * 0.49
        if blinking:
            draw_list.add_line(
                imgui.ImVec2(eye_x - size * 0.04, eye_y),
                imgui.ImVec2(eye_x + size * 0.04, eye_y),
                cyan,
                thickness,
            )
        else:
            draw_list.add_circle_filled(imgui.ImVec2(eye_x, eye_y), size * 0.043, cyan)
    draw_list.add_line(
        imgui.ImVec2(x + size * 0.45, y + size * 0.66),
        imgui.ImVec2(x + size * 0.55, y + size * 0.66),
        outline,
        thickness,
    )


def _draw_centered_text(
    imgui: Any,
    draw_list: Any,
    font: Any,
    font_size: float,
    center_x: float,
    y: float,
    width: float,
    color: int,
    text: str,
) -> None:
    """Center wrapped text in a fixed-width part of the loading card."""
    imgui.push_font(font, font_size)
    try:
        text_size = imgui.calc_text_size(text, wrap_width=width)
        draw_list.add_text(
            imgui.get_font(),
            imgui.get_font_size(),
            imgui.ImVec2(center_x - min(text_size.x, width) * 0.5, y),
            color,
            text,
            wrap_width=width,
        )
    finally:
        imgui.pop_font()


def draw_loading_splash(gui: Any, *, started: float | None = None) -> None:
    """Draw the animated greeting over the complete viewer on its GL thread.

    Args:
        gui: Native Newton GUI whose loading-splash text supplies the current activity.
        started: Monotonic loading start time [s], used for the elapsed clock and animation.
            Without it, the animation follows ImGui's clock and the elapsed label is omitted.
    """
    if not gui.is_available:
        return
    imgui = gui.ui.imgui
    viewport = imgui.get_main_viewport()
    draw_list = imgui.get_foreground_draw_list(viewport)
    elapsed = max(0.0, time.monotonic() - started) if started is not None else imgui.get_time()
    scale = max(0.25, min(imgui.get_font_size() / 14.0, viewport.size.x / 580.0, viewport.size.y / 390.0))
    center_x = viewport.pos.x + viewport.size.x * 0.5
    center_y = viewport.pos.y + viewport.size.y * 0.5
    width, height = 520.0 * scale, 324.0 * scale
    left, top = center_x - width * 0.5, center_y - height * 0.5
    background = imgui.get_style_color_vec4(imgui.Col_.window_bg)
    dark = background.x + background.y + background.z < 1.5
    dim = imgui.ImVec4(0.025, 0.03, 0.035, 0.86) if dark else imgui.ImVec4(0.90, 0.91, 0.92, 0.86)
    green = imgui.color_convert_float4_to_u32(imgui.ImVec4(118 / 255, 185 / 255, 0, 1))
    muted = imgui.get_color_u32(imgui.Col_.text_disabled)
    foreground = imgui.get_color_u32(imgui.Col_.text)

    draw_list.add_rect_filled(
        viewport.pos,
        imgui.ImVec2(viewport.pos.x + viewport.size.x, viewport.pos.y + viewport.size.y),
        imgui.color_convert_float4_to_u32(dim),
    )
    draw_list.add_rect_filled(
        imgui.ImVec2(left + 3 * scale, top + 6 * scale),
        imgui.ImVec2(left + width + 3 * scale, top + height + 6 * scale),
        imgui.color_convert_float4_to_u32(imgui.ImVec4(0, 0, 0, 0.16)),
        rounding=14 * scale,
    )
    draw_list.add_rect_filled(
        imgui.ImVec2(left, top),
        imgui.ImVec2(left + width, top + height),
        imgui.color_convert_float4_to_u32(imgui.ImVec4(background.x, background.y, background.z, 1)),
        rounding=14 * scale,
    )
    draw_list.add_rect(
        imgui.ImVec2(left, top),
        imgui.ImVec2(left + width, top + height),
        imgui.get_color_u32(imgui.Col_.border),
        rounding=14 * scale,
    )
    draw_isaaclab_logo(imgui, draw_list, center_x - 48 * scale, top + 24 * scale, 96 * scale, elapsed)

    font = gui.ui.io.font_default
    heading_font = getattr(gui.ui, "heading_font", None) or font
    for label, label_font, font_size, offset, padding, color in (
        ("Isaac Lab", heading_font, 36, 120, 48, green),
        ("Robotics simulation and learning", font, 13, 169, 48, muted),
        (gui._loading_splash_text or "Starting simulation", font, 15, 213, 72, foreground),
    ):
        _draw_centered_text(
            imgui, draw_list, label_font, font_size * scale, center_x, top + offset * scale, width - padding * scale,
            color, label,
        )

    bar_left, bar_top = left + 44 * scale, top + 260 * scale
    bar_width, bar_height = width - 88 * scale, 5 * scale
    draw_list.add_rect_filled(
        imgui.ImVec2(bar_left, bar_top),
        imgui.ImVec2(bar_left + bar_width, bar_top + bar_height),
        imgui.get_color_u32(imgui.Col_.frame_bg),
        rounding=bar_height * 0.5,
    )
    segment_width = bar_width * 0.24
    segment_left = bar_left + (bar_width - segment_width) * (0.5 - 0.5 * math.cos(elapsed * 2.2))
    draw_list.add_rect_filled(
        imgui.ImVec2(segment_left, bar_top),
        imgui.ImVec2(segment_left + segment_width, bar_top + bar_height),
        green,
        rounding=bar_height * 0.5,
    )
    _draw_centered_text(
        imgui,
        draw_list,
        font,
        12 * scale,
        center_x,
        top + 282 * scale,
        width - 48 * scale,
        muted,
        f"Elapsed  {elapsed:.1f} s" if started is not None else "Getting ready",
    )
