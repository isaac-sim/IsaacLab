# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard legend text and label mesh generation."""

from __future__ import annotations

from .keyboard_schema import KeyboardStyle, ResolvedKey

_FONT_5X7: dict[str, tuple[str, ...]] = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10011", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "=": ("00000", "11111", "00000", "11111", "00000", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "*": ("00000", "10101", "01110", "11111", "01110", "10101", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    "\\": ("10000", "01000", "01000", "00100", "00010", "00010", "00001"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ",": ("00000", "00000", "00000", "00000", "00000", "01100", "01000"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    ";": ("00000", "01100", "01100", "00000", "01100", "01000", "10000"),
    "'": ("01100", "00100", "01000", "00000", "00000", "00000", "00000"),
    "`": ("01000", "00100", "00010", "00000", "00000", "00000", "00000"),
    "[": ("01110", "01000", "01000", "01000", "01000", "01000", "01110"),
    "]": ("01110", "00010", "00010", "00010", "00010", "00010", "01110"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
}

_COMMON_LEGEND_REPLACEMENTS = {
    "BACKSPACE": "BSPC",
    "CAPS": "CAPS",
    "SHIFT": "SHFT",  # codespell:ignore
    "CONTROL": "CTRL",
    "PRINT": "PRT",
    "PRINTSCREEN": "PRT",
    "SCROLL": "SCR",
    "PAUSE": "PAUS",
    "INSERT": "INS",
    "DELETE": "DEL",
    "PAGEUP": "PGUP",
    "PAGEDOWN": "PGDN",
    "LEFT": "LFT",
    "RIGHT": "RGT",
    "DOWN": "DN",
    "SPACE": "SPACE",
    "SCREENSHOT": "SHOT",
    "VOLUME": "VOL",
}


def legend_text(label: str, style: KeyboardStyle) -> str:
    if style.label_mode == "blank":
        return ""
    text = "".join(ch for ch in label.upper() if ch in _FONT_5X7 or ch.isalnum()).strip()
    if not text:
        return ""
    compact = text.replace(" ", "")
    if compact == "SPACE" and not style.label_spacebar:
        return ""
    text = _COMMON_LEGEND_REPLACEMENTS.get(compact, text)
    if style.label_mode == "minimal" and len(text) > 4:
        return ""
    if style.label_mode == "compact" and len(text) > 4:
        return text[:4]
    return text


def label_mesh_data(
    key: ResolvedKey, style: KeyboardStyle
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int, int]]]:
    text = legend_text(key.label, style)
    if not text:
        return [], []
    glyph_width = 5
    glyph_height = 7
    glyph_gap = 1
    total_cols = sum(glyph_width if ch != " " else 3 for ch in text) + max(len(text) - 1, 0) * glyph_gap
    if total_cols <= 0:
        return [], []

    cap_top_scale = min(max(style.cap_top_scale, 0.25), 1.0)
    available_w = key.size[0] * cap_top_scale * 0.82
    available_h = key.size[1] * cap_top_scale * 0.50
    pixel_step = min(available_w / total_cols, available_h / glyph_height) * style.label_scale
    pixel_size = pixel_step * style.label_pixel_fill
    if pixel_size <= 0.0:
        return [], []

    width = total_cols * pixel_step
    height = glyph_height * pixel_step
    x0 = -width * 0.5
    y0 = height * 0.5
    z = key.size[2] * 0.5 + max(0.00004, key.size[2] * 0.018)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    cursor = 0.0
    for ch in text:
        glyph = _FONT_5X7.get(ch, _FONT_5X7["?"])
        char_cols = 3 if ch == " " else glyph_width
        if ch != " ":
            for row_index, row in enumerate(glyph):
                for col_index, value in enumerate(row):
                    if value != "1":
                        continue
                    cx = x0 + (cursor + col_index + 0.5) * pixel_step
                    cy = y0 - (row_index + 0.5) * pixel_step
                    half = pixel_size * 0.5
                    base = len(vertices)
                    vertices.extend(
                        [
                            (cx - half, cy - half, z),
                            (cx + half, cy - half, z),
                            (cx + half, cy + half, z),
                            (cx - half, cy + half, z),
                        ]
                    )
                    faces.append((base, base + 1, base + 2, base + 3))
        cursor += char_cols + glyph_gap
    return vertices, faces
