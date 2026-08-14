# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Logical keyboard layout builders in key-unit coordinates."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from .keyboard_schema import LogicalKeyboardLayout, UnitKey


def sample_layout(family: str, families: tuple[str, ...], rng: random.Random) -> LogicalKeyboardLayout:
    """Sample and build a logical keyboard layout."""

    if family == "random":
        if not families:
            raise ValueError("Random keyboard layout selection requires at least one family.")
        family = rng.choice(families)
    return LogicalKeyboardLayout(family=family, keys=tuple(LAYOUT_BUILDERS[family](rng)))


def _row(
    y: float,
    tokens: Sequence[tuple[str, str, float, str] | tuple[str, str, float, str, float] | float],
    *,
    row: int,
    start_x: float = 0.0,
) -> list[UnitKey]:
    keys: list[UnitKey] = []
    x = start_x
    for token in tokens:
        if isinstance(token, (int, float)):
            x += float(token)
            continue
        if len(token) == 4:
            name, label, width, group = token
            height = 1.0
        else:
            name, label, width, group, height = token
        keys.append(UnitKey(name=name, label=label, x=x, y=y, w=width, h=height, row=row, group=group))
        x += width
    return keys


def _copy_unit_key(key: UnitKey, **updates: Any) -> UnitKey:
    values = {
        "name": key.name,
        "label": key.label,
        "x": key.x,
        "y": key.y,
        "w": key.w,
        "h": key.h,
        "row": key.row,
        "group": key.group,
        "yaw": key.yaw,
        "pitch": key.pitch,
        "roll": key.roll,
    }
    values.update(updates)
    return UnitKey(**values)


def _layout_ansi_60(rng: random.Random) -> list[UnitKey]:
    del rng
    rows = [
        _row(
            4,
            [
                ("grave", "`", 1, "alpha"),
                ("1", "1", 1, "alpha"),
                ("2", "2", 1, "alpha"),
                ("3", "3", 1, "alpha"),
                ("4", "4", 1, "alpha"),
                ("5", "5", 1, "alpha"),
                ("6", "6", 1, "alpha"),
                ("7", "7", 1, "alpha"),
                ("8", "8", 1, "alpha"),
                ("9", "9", 1, "alpha"),
                ("0", "0", 1, "alpha"),
                ("minus", "-", 1, "alpha"),
                ("equal", "=", 1, "alpha"),
                ("backspace", "Backspace", 2, "modifier"),
            ],
            row=4,
        ),
        _row(
            3,
            [
                ("tab", "Tab", 1.5, "modifier"),
                ("q", "Q", 1, "alpha"),
                ("w", "W", 1, "alpha"),
                ("e", "E", 1, "alpha"),
                ("r", "R", 1, "alpha"),
                ("t", "T", 1, "alpha"),
                ("y", "Y", 1, "alpha"),
                ("u", "U", 1, "alpha"),
                ("i", "I", 1, "alpha"),
                ("o", "O", 1, "alpha"),
                ("p", "P", 1, "alpha"),
                ("left_bracket", "[", 1, "alpha"),
                ("right_bracket", "]", 1, "alpha"),
                ("backslash", "\\", 1.5, "modifier"),
            ],
            row=3,
        ),
        _row(
            2,
            [
                ("caps_lock", "Caps", 1.75, "modifier"),
                ("a", "A", 1, "alpha"),
                ("s", "S", 1, "alpha"),
                ("d", "D", 1, "alpha"),
                ("f", "F", 1, "alpha"),
                ("g", "G", 1, "alpha"),
                ("h", "H", 1, "alpha"),
                ("j", "J", 1, "alpha"),
                ("k", "K", 1, "alpha"),
                ("l", "L", 1, "alpha"),
                ("semicolon", ";", 1, "alpha"),
                ("quote", "'", 1, "alpha"),
                ("enter", "Enter", 2.25, "accent"),
            ],
            row=2,
        ),
        _row(
            1,
            [
                ("left_shift", "Shift", 2.25, "modifier"),
                ("z", "Z", 1, "alpha"),
                ("x", "X", 1, "alpha"),
                ("c", "C", 1, "alpha"),
                ("v", "V", 1, "alpha"),
                ("b", "B", 1, "alpha"),
                ("n", "N", 1, "alpha"),
                ("m", "M", 1, "alpha"),
                ("comma", ",", 1, "alpha"),
                ("period", ".", 1, "alpha"),
                ("slash", "/", 1, "alpha"),
                ("right_shift", "Shift", 2.75, "modifier"),
            ],
            row=1,
        ),
        _row(
            0,
            [
                ("left_ctrl", "Ctrl", 1.25, "modifier"),
                ("left_meta", "Meta", 1.25, "modifier"),
                ("left_alt", "Alt", 1.25, "modifier"),
                ("space", "Space", 6.25, "accent"),
                ("right_alt", "Alt", 1.25, "modifier"),
                ("fn", "Fn", 1.25, "modifier"),
                ("menu", "Menu", 1.25, "modifier"),
                ("right_ctrl", "Ctrl", 1.25, "modifier"),
            ],
            row=0,
        ),
    ]
    return [key for row_keys in rows for key in row_keys]


def _layout_ansi_tkl(rng: random.Random) -> list[UnitKey]:
    del rng
    keys = _layout_ansi_60(random.Random(0))
    keys = [_copy_unit_key(key, name=f"main_{key.name}") for key in keys]
    function = _row(
        6,
        [
            ("esc", "Esc", 1, "accent"),
            1.0,
            ("f1", "F1", 1, "alpha"),
            ("f2", "F2", 1, "alpha"),
            ("f3", "F3", 1, "alpha"),
            ("f4", "F4", 1, "alpha"),
            0.5,
            ("f5", "F5", 1, "alpha"),
            ("f6", "F6", 1, "alpha"),
            ("f7", "F7", 1, "alpha"),
            ("f8", "F8", 1, "alpha"),
            0.5,
            ("f9", "F9", 1, "alpha"),
            ("f10", "F10", 1, "alpha"),
            ("f11", "F11", 1, "alpha"),
            ("f12", "F12", 1, "alpha"),
        ],
        row=6,
    )
    nav_x = 16.25
    nav = []
    nav += _row(
        6,
        [
            ("print_screen", "Prt", 1, "modifier"),
            ("scroll_lock", "Scr", 1, "modifier"),
            ("pause", "Pause", 1, "modifier"),
        ],
        row=6,
        start_x=nav_x,
    )
    nav += _row(
        4,
        [("insert", "Ins", 1, "modifier"), ("home", "Home", 1, "modifier"), ("page_up", "PgUp", 1, "modifier")],
        row=4,
        start_x=nav_x,
    )
    nav += _row(
        3,
        [("delete", "Del", 1, "modifier"), ("end", "End", 1, "modifier"), ("page_down", "PgDn", 1, "modifier")],
        row=3,
        start_x=nav_x,
    )
    nav += _row(0, [1.0, ("arrow_up", "Up", 1, "accent")], row=0, start_x=nav_x)
    nav += _row(
        -1.0,
        [
            ("arrow_left", "Left", 1, "accent"),
            ("arrow_down", "Down", 1, "accent"),
            ("arrow_right", "Right", 1, "accent"),
        ],
        row=-1,
        start_x=nav_x,
    )
    return keys + function + nav


def _layout_ansi_full(rng: random.Random) -> list[UnitKey]:
    del rng
    keys: list[UnitKey] = []
    arrow_names = {"arrow_up", "arrow_left", "arrow_down", "arrow_right"}
    for key in _layout_ansi_tkl(random.Random(0)):
        if key.name in arrow_names:
            continue
        updates = {"y": 5.0, "row": 5} if key.y == 6 else {}
        keys.append(_copy_unit_key(key, **updates))

    arrow_x = 16.25
    keys.extend(
        [
            UnitKey("arrow_up", "Up", arrow_x + 1.0, 1.0, group="accent", row=1),
            UnitKey("arrow_left", "Left", arrow_x, 0.0, group="accent", row=0),
            UnitKey("arrow_down", "Down", arrow_x + 1.0, 0.0, group="accent", row=0),
            UnitKey("arrow_right", "Right", arrow_x + 2.0, 0.0, group="accent", row=0),
        ]
    )

    numpad_x = 20.25
    numpad = [
        _copy_unit_key(key, name=f"num_{key.name}", x=key.x + numpad_x) for key in _layout_numpad(random.Random(0))
    ]
    media = _row(
        5,
        [
            ("calc", "Calc", 1, "modifier"),
            ("screenshot", "Shot", 1, "modifier"),
            ("lock", "Lock", 1, "modifier"),
            ("volume", "Vol", 1, "accent"),
        ],
        row=5,
        start_x=numpad_x,
    )
    return keys + numpad + media


def _layout_numpad(rng: random.Random) -> list[UnitKey]:
    del rng
    return [
        UnitKey("num_lock", "Num", 0, 4, group="modifier", row=4),
        UnitKey("divide", "/", 1, 4, group="modifier", row=4),
        UnitKey("multiply", "*", 2, 4, group="modifier", row=4),
        UnitKey("minus", "-", 3, 4, group="modifier", row=4),
        UnitKey("seven", "7", 0, 3, row=3),
        UnitKey("eight", "8", 1, 3, row=3),
        UnitKey("nine", "9", 2, 3, row=3),
        UnitKey("plus", "+", 3, 2, h=2, group="accent", row=2),
        UnitKey("four", "4", 0, 2, row=2),
        UnitKey("five", "5", 1, 2, row=2),
        UnitKey("six", "6", 2, 2, row=2),
        UnitKey("one", "1", 0, 1, row=1),
        UnitKey("two", "2", 1, 1, row=1),
        UnitKey("three", "3", 2, 1, row=1),
        UnitKey("enter", "Enter", 3, 0, h=2, group="accent", row=0),
        UnitKey("zero", "0", 0, 0, w=2, group="alpha", row=0),
        UnitKey("decimal", ".", 2, 0, row=0),
    ]


def _layout_macro_pad(rng: random.Random) -> list[UnitKey]:
    rows, cols = rng.choice([(3, 3), (3, 4), (4, 4), (5, 3), (2, 6)])
    keys: list[UnitKey] = []
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            keys.append(
                UnitKey(
                    f"macro_{row}_{col}",
                    _macro_pad_label(index),
                    col,
                    rows - row - 1,
                    row=row,
                    group="accent" if col == 0 else "alpha",
                )
            )
    if rows >= 3 and cols >= 4 and rng.random() < 0.45:
        keys = [key for key in keys if not (key.x == cols - 1 and key.y == rows - 1)]
    if cols >= 3 and rng.random() < 0.35:
        keys = [key for key in keys if not (key.x in (0, 1) and key.y == 0)]
        keys.append(UnitKey("macro_wide", "Enter", 0, 0, w=2, group="accent", row=0))
    return keys


def _layout_ortholinear(rng: random.Random) -> list[UnitKey]:
    rows, cols = rng.choice([(4, 12), (5, 12), (5, 15)])
    keys: list[UnitKey] = []
    for row in range(rows):
        for col in range(cols):
            group = "modifier" if row == 0 else "alpha"
            if row == 0 and cols == 12 and 4 <= col <= 7:
                group = "accent"
            keys.append(
                UnitKey(
                    f"ortho_{row}_{col}",
                    _ortho_label(row, col, rows, cols),
                    col,
                    row,
                    row=row,
                    group=group,
                )
            )
    if rows >= 4 and cols == 12:
        keys = [key for key in keys if not (key.y == 0 and 4 <= key.x <= 7)]
        keys.append(UnitKey("space_left", "Space", 4, 0, w=2, group="accent", row=0))
        keys.append(UnitKey("space_right", "Space", 6, 0, w=2, group="accent", row=0))
    return keys


def _layout_split_columnar(rng: random.Random) -> list[UnitKey]:
    keys: list[UnitKey] = []
    rows_by_col = [3, 4, 4, 4, 4, 3]
    stagger = [-0.22, 0.0, 0.22, 0.1, -0.08, -0.18]
    x_spacing = rng.uniform(1.26, 1.38)
    y_spacing = rng.uniform(1.28, 1.40)
    split_gap = rng.uniform(2.8, 3.8)
    yaw = rng.uniform(0.10, 0.22)
    tent = rng.uniform(0.0, 0.10)
    for side, mirror in (("left", -1), ("right", 1)):
        x_base = -split_gap if side == "left" else split_gap
        side_yaw = -yaw if side == "left" else yaw
        for col, count in enumerate(rows_by_col):
            x = x_base + mirror * col * x_spacing
            if side == "left":
                x -= x_spacing
            for row in range(count):
                y = row * y_spacing + stagger[col] * y_spacing
                keys.append(
                    UnitKey(
                        f"{side}_c{col}_r{row}",
                        _split_columnar_label(side, col, row),
                        x,
                        y,
                        row=row,
                        group="alpha",
                        yaw=side_yaw,
                        pitch=mirror * tent,
                    )
                )
        thumb_y = -1.45 * y_spacing
        thumb_x0 = x_base + mirror * 1.6 * x_spacing
        if side == "left":
            thumb_x0 -= x_spacing
        for idx in range(3):
            keys.append(
                UnitKey(
                    f"{side}_thumb_{idx}",
                    _split_thumb_label(side, idx),
                    thumb_x0 + mirror * idx * 1.25 * x_spacing,
                    thumb_y - 0.18 * idx,
                    w=1.25 if idx == 1 else 1.0,
                    row=-1,
                    group="accent",
                    yaw=side_yaw + mirror * 0.18,
                    pitch=mirror * tent,
                )
            )
    return keys


def _layout_compact(rng: random.Random) -> list[UnitKey]:
    base = _layout_ansi_60(rng)
    if rng.random() < 0.5:
        compact = [key for key in base if key.name != "space"]
        compact.extend(
            [
                UnitKey("space_left", "Space", 3.75, 0, w=2.75, group="accent", row=0),
                UnitKey("space_right", "Space", 6.5, 0, w=2.75, group="accent", row=0),
            ]
        )
    else:
        compact = base
    if rng.random() < 0.45:
        for y in range(5):
            compact.append(UnitKey(f"left_macro_{y}", _macro_pad_label(y), -1.25, y, group="accent", row=y))
    return compact


def _macro_pad_label(index: int) -> str:
    labels = (
        "Esc",
        "F13",
        "F14",
        "F15",
        "F16",
        "F17",
        "F18",
        "F19",
        "F20",
        "F21",
        "F22",
        "F23",
        "F24",
        "Mute",
        "Vol-",
        "Vol+",
        "Play",
        "Prev",
        "Next",
        "Copy",
        "Paste",
    )
    return labels[index % len(labels)]


def _ortho_label(row: int, col: int, rows: int, cols: int) -> str:
    if cols == 15:
        labels = (
            (
                "Ctrl",
                "Meta",
                "Alt",
                "Lower",
                "Fn",
                "Space",
                "Space",
                "Space",
                "Space",
                "Raise",
                "Left",
                "Down",
                "Up",
                "Right",
                "Del",
            ),
            ("Shift", "Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "Shift", "Home", "End", "PgDn"),
            ("Caps", "A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "'", "Enter", "Ins", "PgUp"),
            ("Tab", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "[", "]", "\\", "Bksp"),
            ("Esc", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "Prt", "Pause"),
        )
    else:
        labels = (
            ("Ctrl", "Meta", "Alt", "Lower", "Space", "Space", "Space", "Space", "Raise", "Left", "Down", "Right"),
            ("Shift", "Z", "X", "C", "V", "B", "N", "M", ",", ".", "/", "Shift"),
            ("Caps", "A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "Enter"),
            ("Tab", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "Bksp"),
            ("Esc", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11"),
        )
    if row >= rows or row >= len(labels) or col >= len(labels[row]):
        return "Fn"
    return labels[row][col]


def _split_columnar_label(side: str, col: int, row: int) -> str:
    left = (
        ("Ctrl", "Shift", "Tab"),
        ("Alt", "Z", "A", "Q"),
        ("Meta", "X", "S", "W"),
        ("Lower", "C", "D", "E"),
        ("Space", "V", "F", "R"),
        ("B", "G", "T"),
    )
    right = (
        ("N", "H", "Y"),
        ("Space", "M", "J", "U"),
        ("Raise", ",", "K", "I"),
        ("Alt", ".", "L", "O"),
        ("Ctrl", "/", ";", "P"),
        ("Bksp", "Enter", "Del"),
    )
    columns = left if side == "left" else right
    if col < len(columns) and row < len(columns[col]):
        return columns[col][row]
    return "Fn"


def _split_thumb_label(side: str, index: int) -> str:
    labels = {
        "left": ("Alt", "Space", "Enter"),
        "right": ("Space", "Bksp", "Del"),
    }
    return labels[side][index]


LAYOUT_BUILDERS = {
    "ansi_60": _layout_ansi_60,
    "ansi_tkl": _layout_ansi_tkl,
    "ansi_full": _layout_ansi_full,
    "numpad": _layout_numpad,
    "macro_pad": _layout_macro_pad,
    "ortholinear": _layout_ortholinear,
    "split_columnar": _layout_split_columnar,
    "compact": _layout_compact,
}
