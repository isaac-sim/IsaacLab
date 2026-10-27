# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Window-local Isaac Lab colors for Newton's native interface."""

from __future__ import annotations

import weakref
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from newton._src.viewer.gui.viewer_gui import ViewerGui


# Match NVIDIA green used by isaaclab.app.loading_screen's CLI greeting.
_GREEN = (118 / 255, 185 / 255, 0.0, 1.0)
_GREEN_LIGHT = (0.60, 0.85, 0.18, 1.0)
_GREEN_DARK = (0.23, 0.34, 0.10, 1.0)


def install_isaaclab_theme(gui: ViewerGui, *, workspace_enabled: bool = False) -> None:
    """Install a green theme on one GUI, preserving other windows and native attribution.

    Colors are applied before drawing because Newton restores its default palette when display
    scaling changes. Only the supplied GUI and its workspace widgets are customized; Newton's
    widget classes and the ImGui module remain unchanged. Install once when the GUI is created.

    Args:
        gui: Native GUI whose current ImGui context is owned by the viewer.
        workspace_enabled: Whether this Newton version supports the native workspace.
    """
    render_ui = gui._render_ui
    themed_widgets = None
    themed_timeline = None

    @wraps(render_ui)
    def render() -> None:
        nonlocal themed_widgets, themed_timeline
        imgui = gui.ui.imgui
        style = imgui.get_style()
        for name, color in (
            ("check_mark", _GREEN),
            ("slider_grab", _GREEN),
            ("slider_grab_active", _GREEN_LIGHT),
            ("button_hovered", (0.27, 0.34, 0.20, 1.0)),
            ("button_active", _GREEN_DARK),
            ("header", (0.25, 0.30, 0.20, 1.0)),
            ("header_hovered", (0.29, 0.38, 0.19, 1.0)),
            ("header_active", _GREEN_DARK),
            ("tab_hovered", (0.29, 0.38, 0.19, 1.0)),
            ("tab_selected_overline", _GREEN),
            ("tab_dimmed_selected_overline", (0.34, 0.48, 0.16, 1.0)),
            ("separator_hovered", _GREEN),
            ("separator_active", _GREEN_LIGHT),
            ("resize_grip", (*_GREEN[:3], 0.20)),
            ("resize_grip_hovered", (*_GREEN[:3], 0.65)),
            ("resize_grip_active", (*_GREEN[:3], 0.90)),
            ("nav_cursor", _GREEN),
            ("input_text_cursor", _GREEN),
            ("text_selected_bg", (*_GREEN[:3], 0.30)),
            ("docking_preview", (*_GREEN[:3], 0.45)),
        ):
            # Newton versions share ImGui's named colors; newer optional colors may be absent
            # when the viewer uses an older imgui-bundle installation.
            if hasattr(imgui.Col_, name):
                style.set_color_(getattr(imgui.Col_, name), imgui.ImVec4(*color))

        workspace = gui.workspace if workspace_enabled else None
        if workspace is not None and workspace.widgets is not themed_widgets:
            themed_widgets = workspace.widgets
            themed_widgets.accent = _GREEN

            def tab_item(label, *, flags=0, accent=False):
                if not accent:
                    return imgui.begin_tab_item(label, flags=flags)
                imgui.push_style_color(imgui.Col_.tab, imgui.ImVec4(0.20, 0.25, 0.16, 1.0))
                imgui.push_style_color(imgui.Col_.tab_selected, imgui.ImVec4(*_GREEN_DARK))
                imgui.push_style_color(imgui.Col_.tab_hovered, imgui.ImVec4(0.29, 0.41, 0.15, 1.0))
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*_GREEN_LIGHT))
                try:
                    return imgui.begin_tab_item(label, flags=flags)
                finally:
                    imgui.pop_style_color(4)

            themed_widgets.tab_item = tab_item
        if workspace is not None and workspace._active_timeline is not None:
            timeline = workspace._active_timeline
            if themed_timeline is None or timeline is not themed_timeline():
                themed_timeline = weakref.ref(timeline)
                # A new presenter is created when the scene is rebound. Its widgets are local
                # to each frame. Keep history ownership with the workspace when it is detached.
                draw_transport = type(timeline)._draw_transport

                @wraps(draw_transport)
                def transport(ui, viewer, widgets, _draw_transport=draw_transport, _timeline=themed_timeline):
                    widgets.accent = _GREEN
                    return _draw_transport(_timeline(), ui, viewer, widgets)

                timeline._draw_transport = transport
        render_ui()

    gui._render_ui = render
