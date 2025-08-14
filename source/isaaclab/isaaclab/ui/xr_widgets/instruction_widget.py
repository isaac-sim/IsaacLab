# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import functools
import textwrap
from typing import Any, TypeAlias, Optional
from collections.abc import Callable

import omni.kit.commands
import omni.ui as ui
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path
from omni.kit.xr.scene_view.utils import UiContainer, WidgetComponent
from omni.kit.xr.scene_view.utils.spatial_source import SpatialSource
from pxr import Gf, Usd
from isaacsim.core.prims import XFormPrim
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.stage import get_current_stage

Vec3Type: TypeAlias = Gf.Vec3f | Gf.Vec3d

camera_facing_widget_container = {}
camera_facing_widget_timers = {}


class SimpleTextWidget(ui.Widget):
    def __init__(self, text: str | None = "Simple Text", style: dict[str, Any] | None = None, **kwargs):
        super().__init__(**kwargs)
        if style is None:
            style = {"font_size": 1, "color": 0xFFFFFFFF}
        self._text = text
        self._style = style
        self._ui_label = None
        self._build_ui()

    def set_label_text(self, text: str):
        """Update the text displayed by the label."""
        self._text = text
        if self._ui_label:
            self._ui_label.text = self._text

    def _build_ui(self):
        """Build the UI with a window-like rectangle and centered label."""
        with ui.ZStack():
            ui.Rectangle(style={"Rectangle": {"background_color": 0xFF454545, "border_radius": 0.1}})
            with ui.VStack(alignment=ui.Alignment.CENTER):
                # Ensure text is never None for ui.Label
                display_text = self._text if self._text is not None else "Simple Text"
                self._ui_label = ui.Label(display_text, style=self._style, alignment=ui.Alignment.CENTER)
                # self._ui_label = ui.Label(display_text, style=self._style, alignment=ui.Alignment.LEFT_TOP)


def compute_widget_dimensions(
    text: str, font_size: float, max_width: float, min_width: float
) -> tuple[float, float, list[str]]:
    """
    Estimate widget dimensions based on text content.

    Returns:
        actual_width (float): The width, clamped between min_width and max_width.
        actual_height (float): The computed height based on wrapped text lines.
        lines (List[str]): The list of wrapped text lines.
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
    return actual_width, actual_height, lines

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
    callback_on_hide: Optional[Callable] = None,
    preferred_width: float = 0.0,
    preferred_height: float = 0.0,
    is_billboard: bool = True
) -> UiContainer | None:
    """
    Create and display the instruction widget based on the given text.

    The widget's width and height are computed dynamically based on the input text.
    It automatically wraps text that is too long and adjusts the widget's height
    accordingly. If a display duration is provided (non-zero), the widget is automatically
    hidden after that many seconds.

    Args:
        text (str): The instruction text to display.
        prim_path_source (Optional[str]): The prim path to be used as a spatial sourcey
            for the widget.
        translation (Gf.Vec3d): A translation vector specifying the widget's position.
        display_duration (Optional[float]): The time in seconds to display the widget before
            automatically hiding it. If None or 0, the widget remains visible until manually
            hidden.
        target_prim_path (str): The target path where the copied prim will be created.
            Defaults to "/newPrim".

    Returns:
        UiContainer: The container instance holding the instruction widget.
    """
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

    # Clean up existing widget
    if get_prim_at_path(target_prim_path):
        delete_prim(target_prim_path)

    # Compute dimensions and wrap text.
    if preferred_width > 0.0:
        max_width = preferred_width
        min_width = preferred_width

    width, height, lines = compute_widget_dimensions(text, font_size, max_width, min_width)

    if preferred_height > 0.0:
        height = preferred_height

    wrapped_text = "\n".join(lines)

    # Create the widget component.
    widget_component = WidgetComponent(
        SimpleTextWidget,
        width=width,
        height=height,
        resolution_scale=300,
        widget_args=[wrapped_text, {"font_size": font_size, "color": text_color}],
    )

    copied_prim = omni.kit.commands.execute(
        "CopyPrim",
        path_from=prim_path_source,
        path_to=target_prim_path,
        exclusive_select=False,
        copy_to_introducing_layer=False,
    )

# Todo: stop copy prim, copy transform instead
    # stage = get_current_stage()

    # if stage.GetPrimAtPath(prim_path_source).IsValid(): 
    #     prim = stage.GetPrimAtPath(prim_path_source)

    #     new_prim = XFormPrim(prim_paths_expr=target_prim_path, positions=prim.)
    #     assert new_prim.is_valid()

    space_stack = []
    if copied_prim is not None:
        space_stack.append(SpatialSource.new_prim_path_source(target_prim_path))

    if is_billboard:
        space_stack.extend([
            SpatialSource.new_translation_source(translation),
            SpatialSource.new_look_at_camera_source(),
        ])
    else:
        space_stack.extend([
            SpatialSource.new_translation_source(translation),
        ])

    # Create the UI container with the widget.
    container = UiContainer(
        widget_component,
        space_stack=space_stack,
    )
    camera_facing_widget_container[target_prim_path] = (container, text)

    # Schedule auto-hide after the specified display_duration if provided.
    if display_duration:
        timer = asyncio.get_event_loop().call_later(display_duration, functools.partial(hide_instruction, target_prim_path, callback_on_hide))
        camera_facing_widget_timers[target_prim_path] = timer

    return container

def hide_instruction(target_prim_path: str = "/newPrim", callback: Optional[Callable] = None) -> None:
    """
    Hide and clean up a specific instruction widget.
    Also cleans up associated timer.
    """

    if callback:
        callback()

    global camera_facing_widget_container, camera_facing_widget_timers

    if target_prim_path in camera_facing_widget_container:
        container, _ = camera_facing_widget_container[target_prim_path]
        container.root.clear()
        del camera_facing_widget_container[target_prim_path]

    if target_prim_path in camera_facing_widget_timers:
        del camera_facing_widget_timers[target_prim_path]

def update_instruction(target_prim_path: str = "/newPrim", text: str = ""):
    """
    Update the text content of an existing instruction widget.
    
    Args:
        target_prim_path (str): The path of the widget to update.
        text (str): The new text content to display.
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
            components = getattr(manipulator, '_ComposableManipulator__components')
            if len(components) > 0:
                simple_text_widget = components[0]
                if simple_text_widget and simple_text_widget.component and simple_text_widget.component.widget:
                    simple_text_widget.component.widget.set_label_text(text)
                
                # Update the stored text in the global dictionary
                camera_facing_widget_container[target_prim_path] = (container, text)
                return True
    
    return False
