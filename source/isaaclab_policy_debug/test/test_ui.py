# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_policy_debug.ui import _draw_color_swatch


class _Flags:
    no_tooltip = 1
    no_drag_drop = 2


class _ImVec4(tuple):
    def __new__(cls, *values):
        return super().__new__(cls, values)


class _ImVec2(tuple):
    def __new__(cls, *values):
        return super().__new__(cls, values)


class _Imgui:
    ImVec4 = _ImVec4
    ImVec2 = _ImVec2
    ColorEditFlags_ = _Flags

    def __init__(self):
        self.calls = []

    def color_button(self, item_id, color, flags, size):
        self.calls.append((item_id, color, flags, size))


def test_checkpoint_color_swatch_uses_visual_tint():
    imgui = _Imgui()

    _draw_color_swatch(imgui, "##checkpoint", (0.2, 0.4, 0.8))

    assert imgui.calls == [("##checkpoint", (0.2, 0.4, 0.8, 1.0), 3, (0.0, 0.0))]
