# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from collections.abc import Callable
from typing import Any

import omni.ui as ui
from isaacsim.gui.components.element_wrappers import Button, FloatField, Frame, IntField
from isaacsim.gui.components.ui_utils import (
    BUTTON_WIDTH,
    LABEL_HEIGHT,
    LABEL_WIDTH,
    add_line_rect_flourish,
    format_tt,
    get_style,
)
from omni.kit.window.filepicker import FilePickerDialog
from omni.ui import CollapsableFrame, Fraction, ScrollBarPolicy


def create_collapsable_frame(title: str):
    return CollapsableFrame(
        title=title,
        width=Fraction(1),
        height=0,
        collapsed=False,
        style=get_style(),
        horizontal_scrollbar_policy=ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
        vertical_scrollbar_policy=ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
    )


def open_folder_picker(callback_fn, window_title="Select a file", apply_button_label="Select"):
    def on_selected(a, b):
        callback_fn(os.path.join(b, a))
        folder_picker.hide()

    def on_canceled(a, b):
        folder_picker.hide()

    folder_picker = FilePickerDialog(
        title=window_title,
        allow_multi_selection=False,
        apply_button_label=apply_button_label,
        click_apply_handler=lambda a, b: on_selected(a, b),
        click_cancel_handler=lambda a, b: on_canceled(a, b),
    )


class ESButton(Button):
    """Create a Button UI Element for EvalSim

    Args:
        label (str): Short descriptive text to the left of the Button
        text (str): Text on the Button
        tooltip (str, optional): Text to appear when the mouse hovers over the Button. Defaults to "".
        on_click_fn (Callable, optional): Callback function that will be called when the button is pressed.
            Function should take no arguments.  The return value will not be used.  Defaults to None.
        use_line_rect_flourish (bool): Use line rect flourish. Defaults to True
    """

    def __init__(self, label: str, text: str, tooltip="", on_click_fn=None, use_line_rect_flourish: bool = True):
        self.use_line_rect_flourish = use_line_rect_flourish
        super().__init__(label, text, tooltip, on_click_fn)

    def _create_ui_widget(self, label: str, text: str, tooltip: str):
        containing_frame = Frame().frame
        with containing_frame:
            with ui.HStack():
                self._label = ui.Label(
                    label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip)
                )
                self._button = ui.Button(
                    text.upper(),
                    name="Button",
                    width=BUTTON_WIDTH,
                    clicked_fn=self._on_clicked_fn_wrapper,
                    style=get_style(),
                    alignment=ui.Alignment.LEFT_CENTER,
                )
                ui.Spacer(width=5)
                if self.use_line_rect_flourish:
                    add_line_rect_flourish(True)
        return containing_frame


class ESIntField(IntField):
    """
    Creates a IntField UI element for EvalSim.

    Args:
        label (str): Short descriptive text to the left of the IntField.
        tooltip (str, optional): Text to appear when the mouse hovers over the IntField. Defaults to "".
        default_value (int, optional): Default value of the IntField. Defaults to 0.
        lower_limit (int, optional): Lower limit of float. Defaults to None.
        upper_limit (int, optional): Upper limit of float. Defaults to None.
        on_value_changed_fn (Callable, optional): Function to be called when the value of the int is changed.
            The function should take an int as an argument.  The return value will not be used. Defaults to None.
    """

    def _create_ui_widget(self, label, tooltip, default_value):
        containing_frame = Frame().frame
        with containing_frame:
            with ui.HStack():
                self._label = ui.Label(
                    label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip)
                )
                self._int_field = ui.IntDrag(
                    name="Field",
                    height=LABEL_HEIGHT,
                    alignment=ui.Alignment.LEFT_CENTER,
                    min=self.get_lower_limit(),
                    max=self.get_upper_limit(),
                )
                self.int_field.model.set_value(default_value)
                add_line_rect_flourish(False)
            self.int_field.model.add_value_changed_fn(self._on_value_changed_fn_wrapper)
        return containing_frame


class ESFloatField(FloatField):
    """
    Creates a FloatField UI element.

    Args:
        label (str): Short descriptive text to the left of the FloatField.
        tooltip (str, optional): Text to appear when the mouse hovers over the FloatField. Defaults to "".
        default_value (float, optional): Default value of the Float Field. Defaults to 0.0.
        step (float, optional): Smallest increment that the user can change the float by when dragging mouse. Defaults to 0.01.
        format (str, optional): Formatting string for float. Defaults to "%.2f".
        lower_limit (float, optional): Lower limit of float. Defaults to None.
        upper_limit (float, optional): Upper limit of float. Defaults to None.
        on_value_changed_fn (Callable, optional): Function to be called when the value of the float is changed.
            The function should take a float as an argument.  The return value will not be used. Defaults to None.
    """

    def __init__(
        self,
        label: str,
        tooltip: str = "",
        default_value: float = 0,
        step: float = 0.01,
        format: str = "%.2f",
        lower_limit: float = None,
        upper_limit: float = None,
        on_value_changed_fn: Callable[..., Any] = None,
        mouse_double_clicked_fn: Callable[..., Any] | None = None,
    ):
        self.mouse_double_clicked_fn = mouse_double_clicked_fn
        super().__init__(label, tooltip, default_value, step, format, lower_limit, upper_limit, on_value_changed_fn)

    def _create_ui_widget(self, label, tooltip, default_value, step, format):
        containing_frame = Frame().frame
        with containing_frame:
            with ui.HStack():
                self._label = ui.Label(
                    label, width=LABEL_WIDTH, alignment=ui.Alignment.LEFT_CENTER, tooltip=format_tt(tooltip)
                )
                self._float_field = ui.FloatDrag(
                    name="FloatField",
                    width=ui.Fraction(1),
                    height=0,
                    alignment=ui.Alignment.LEFT_CENTER,
                    min=self.get_lower_limit(),
                    max=self.get_upper_limit(),
                    step=step,
                    format=format,
                    mouse_double_clicked_fn=self.mouse_double_clicked_fn,
                )
                self.float_field.model.set_value(default_value)
                add_line_rect_flourish(False)

            self.float_field.model.add_value_changed_fn(self._on_value_changed_fn_wrapper)
        return containing_frame
