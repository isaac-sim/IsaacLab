# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import asyncio
import functools
import textwrap
from typing import TYPE_CHECKING, Any, TypeAlias

import omni.kit.commands
import omni.ui as ui
from pxr import Gf

import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from omni.kit.xr.scene_view.utils import UiContainer

Vec3Type: TypeAlias = Gf.Vec3f | Gf.Vec3d

camera_facing_widget_container = {}
camera_facing_widget_timers = {}


class SimpleTextWidget(ui.Widget):
    """A rectangular text label widget for XR overlays.

    The widget renders a centered label over a rectangular background. It keeps
    track of the configured style and an original width value used by
    higher-level helpers to update the text.
    """

    def __init__(
        self,
        text: str | None = "Simple Text",
        style: dict[str, Any] | None = None,
        original_width: float = 0.0,
        **kwargs,
    ):
        """Initialize the text widget.

        Args:
            text (str): Initial text to display.
            style (dict[str, Any]): Optional style dictionary (for example: ``{"font_size": 1, "color": 0xFFFFFFFF}``).
            original_width (float): Width used when updating the text.
            **kwargs: Additional keyword arguments forwarded to ``ui.Widget``.
        """
        super().__init__(**kwargs)
        if style is None:
            style = {"font_size": 1, "color": 0xFFFFFFFF}
        self._text = text
        self._style = style
        self._ui_label = None
        self._original_width = original_width
        self._build_ui()

    def set_label_text(self, text: str):
        """Update the text displayed by the label.

        Args:
            text (str): New label text to display.
        """
        self._text = text
        if self._ui_label:
            self._ui_label.text = self._text

    def get_font_size(self):
        """Return the configured font size.

        Returns:
            float: Font size value.
        """
        return self._style.get("font_size", 1)

    def get_width(self):
        """Return the width used when updating the text.

        Returns:
            float: Width used when updating the text.
        """
        return self._original_width

    def _build_ui(self):
        """Build the UI with a window-like rectangle and centered label."""
        with ui.ZStack():
            ui.Rectangle(style={"Rectangle": {"background_color": 0xFF454545, "border_radius": 0.1}})
            with ui.VStack(alignment=ui.Alignment.CENTER):
                self._ui_label = ui.Label(self._text, style=self._style, alignment=ui.Alignment.CENTER)


def compute_widget_dimensions(
    text: str, font_size: float, max_width: float, min_width: float
) -> tuple[float, float, str]:
    """Estimate widget width/height and wrap the text.

    Args:
        text (str): Raw text to render.
        font_size (float): Font size used for estimating character metrics.
        max_width (float): Maximum allowed widget width.
        min_width (float): Minimum allowed widget width.

    Returns:
        tuple[float, float, str]: A tuple ``(width, height, wrapped_text)`` where
        ``width`` and ``height`` are the computed widget dimensions, and
        ``wrapped_text`` contains the input text broken into newline-separated
        lines to fit within the width constraints.
    """
    # Estimate average character width.
    char_width = 0.6 * font_size
    max_chars_per_line = int(max_width / char_width)
    lines = textwrap.wrap(text, width=max_chars_per_line)
    if not lines:
        lines = [text]
    computed_width = max(len(line) for line in lines) * char_width
    actual_width = max(min(computed_width, max_width), min_width)
    line_height = 1.2 * font_size
    actual_height = len(lines) * line_height
    wrapped_text = "\n".join(lines)
    return actual_width, actual_height, wrapped_text


def show_instruction(
    text: str,
    prim_path_source: str | None = None,
    translation: Gf.Vec3d = Gf.Vec3d(0, 0, 0),
    display_duration: float | None = 5.0,
    max_width: float = 2.5,
    min_width: float = 1.0,  # Prevent widget from being too narrow.
    font_size: float = 0.1,
    text_color: int = 0xFFFFFFFF,
    target_prim_path: str = "/newPrim",
) -> UiContainer | None:
    """Create and display an instruction widget with the given text.

    The widget size is computed from the text and font size, wrapping content
    to respect the width limits. If ``display_duration`` is provided and
    non-zero, the widget is hidden automatically after the duration elapses.

    Args:
        text (str): Instruction text to display.
        prim_path_source (str | None): Optional prim path used as a spatial source for the widget.
        translation (Gf.Vec3d): World translation to apply to the widget.
        display_duration (float | None): Seconds to keep the widget visible. If ``None`` or ``0``,
            the widget remains until hidden manually.
        max_width (float): Maximum widget width used for wrapping.
        min_width (float): Minimum widget width used for wrapping.
        font_size (float): Font size of the rendered text.
        text_color (int): RGBA color encoded as a 32-bit integer.
        target_prim_path (str): Prim path where the widget prim will be created/copied.

    Returns:
        UiContainer | None: The container that owns the instruction widget, or ``None`` if creation failed.
    """
    try:
        from omni.kit.xr.scene_view.utils import UiContainer, WidgetComponent
        from omni.kit.xr.scene_view.utils.spatial_source import SpatialSource
    except ImportError:
        # TODO(isaaclab-3.0): Re-enable once omni.kit.xr.scene_view.utils is available in the kit file
        return None

    global camera_facing_widget_container, camera_facing_widget_timers

    # Check if widget exists and has different text
    if target_prim_path in camera_facing_widget_container:
        container, current_text = camera_facing_widget_container[target_prim_path]
        if current_text == text:
            return container

        # Cancel existing timer if there is one
        if target_prim_path in camera_facing_widget_timers:
            camera_facing_widget_timers[target_prim_path].cancel()
            del camera_facing_widget_timers[target_prim_path]

        container.root.clear()
        del camera_facing_widget_container[target_prim_path]

    # Obtain stage handle
    stage = sim_utils.get_current_stage()
    # Clean up existing widget
    if stage.GetPrimAtPath(target_prim_path).IsValid():
        sim_utils.delete_prim(target_prim_path)

    width, height, wrapped_text = compute_widget_dimensions(text, font_size, max_width, min_width)

    # Create the widget component.
    widget_component = WidgetComponent(
        SimpleTextWidget,
        width=width,
        height=height,
        resolution_scale=300,
        widget_args=[wrapped_text, {"font_size": font_size, "color": text_color}, width],
    )

    copied_prim = omni.kit.commands.execute(
        "CopyPrim",
        path_from=prim_path_source,
        path_to=target_prim_path,
        exclusive_select=False,
        copy_to_introducing_layer=False,
    )

    space_stack = []
    if copied_prim is not None:
        space_stack.append(SpatialSource.new_prim_path_source(target_prim_path))

    space_stack.extend(
        [
            SpatialSource.new_translation_source(translation),
            SpatialSource.new_look_at_camera_source(),
        ]
    )

    # Create the UI container with the widget.
    container = UiContainer(
        widget_component,
        space_stack=space_stack,
    )
    camera_facing_widget_container[target_prim_path] = (container, text)

    # Schedule auto-hide after the specified display_duration if provided.
    if display_duration:
        timer = asyncio.get_event_loop().call_later(
            display_duration, functools.partial(hide_instruction, target_prim_path)
        )
        camera_facing_widget_timers[target_prim_path] = timer

    return container


def hide_instruction(target_prim_path: str = "/newPrim") -> None:
    """Hide and clean up a specific instruction widget.

    Args:
        target_prim_path (str): Prim path of the widget to hide.

    Returns:
        None: This function does not return a value.
    """

    global camera_facing_widget_container, camera_facing_widget_timers

    if target_prim_path in camera_facing_widget_container:
        container, _ = camera_facing_widget_container[target_prim_path]
        container.root.clear()
        del camera_facing_widget_container[target_prim_path]

    if target_prim_path in camera_facing_widget_timers:
        del camera_facing_widget_timers[target_prim_path]


def update_instruction(target_prim_path: str = "/newPrim", text: str = ""):
    """Update the text content of an existing instruction widget.

    Args:
        target_prim_path (str): Prim path of the widget to update.
        text (str): New text content to display.

    Returns:
        bool: ``True`` if the widget existed and was updated, otherwise ``False``.
    """
    global camera_facing_widget_container

    container_data = camera_facing_widget_container.get(target_prim_path)
    if container_data:
        container, current_text = container_data

        # Only update if the text has actually changed
        if current_text != text:
            # Access the widget through the manipulator as shown in ui_container.py
            manipulator = container.manipulator

            # The WidgetComponent is stored in the manipulator's components
            # Try to access the widget component and then the actual widget
            components = getattr(manipulator, "_ComposableManipulator__components")
            if len(components) > 0:
                simple_text_widget = components[0]
                if simple_text_widget and simple_text_widget.component and simple_text_widget.component.widget:
                    width, height, wrapped_text = compute_widget_dimensions(
                        text,
                        simple_text_widget.component.widget.get_font_size(),
                        simple_text_widget.component.widget.get_width(),
                        simple_text_widget.component.widget.get_width(),
                    )
                    simple_text_widget.component.widget.set_label_text(wrapped_text)
                # Update the stored text in the global dictionary
                camera_facing_widget_container[target_prim_path] = (container, text)
                return True

    return False
