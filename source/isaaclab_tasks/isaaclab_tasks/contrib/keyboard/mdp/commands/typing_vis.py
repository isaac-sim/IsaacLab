# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LED dot-matrix visualization helpers for the SO101 keyboard typing command.

The Newton marker backend only renders primitive prototypes (sphere/box/cylinder/...), so letters are
drawn as a grid of small cuboid "pixels" using the shared 5x7 bitmap font (``_FONT_5X7``). One
cuboid prototype per color encodes the typing state (pending / correct / next / wrong) via the marker
prototype index, and a cylinder prototype marks the next key to press.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers.visualization_markers_cfg import VisualizationMarkersCfg

from ...keyboards.keyboard_labels import _FONT_5X7

# Marker prototype indices. MUST match the insertion order in :func:`make_typing_visualizer_cfg`.
PIX_PENDING = 0  # gray: target letter not yet typed
PIX_CORRECT = 1  # green: correctly typed prefix (target and typed rows)
PIX_NEXT = 2  # yellow: next expected target letter (cursor)
PIX_WRONG = 3  # red: typed letter past the correct prefix (needs backspacing)
KEY_NEXT = 4  # halo over the next key to press

GLYPH_ROWS = 7
GLYPH_COLS = 5


def make_typing_visualizer_cfg(
    prim_path: str, pixel_size: float, key_radius: float, key_height: float
) -> VisualizationMarkersCfg:
    """Build the marker set: one cuboid pixel per color state plus a cylinder key-halo."""

    # ``pixel_size`` is the pixel pitch (spacing); draw each dot a bit smaller for an LED-matrix look.
    dot = pixel_size * 0.72

    def _pix(rgb: tuple[float, float, float]) -> sim_utils.CuboidCfg:
        return sim_utils.CuboidCfg(
            size=(dot, dot, dot),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
        )

    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "pending": _pix((0.55, 0.55, 0.60)),
            "correct": _pix((0.15, 0.90, 0.25)),
            "next": _pix((1.00, 0.82, 0.10)),
            "wrong": _pix((0.95, 0.20, 0.20)),
            "key_next": sim_utils.CylinderCfg(
                radius=key_radius,
                height=key_height,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.00, 0.82, 0.10)),
            ),
        },
    )


def build_glyph_table(device: torch.device | str) -> tuple[torch.Tensor, dict[str, int]]:
    """Return ``(glyph_lit, char_to_index)`` where ``glyph_lit`` is ``(num_chars, 7, 5)`` bool."""
    chars = list(_FONT_5X7.keys())
    char_to_index = {ch: i for i, ch in enumerate(chars)}
    glyph_lit = torch.zeros(len(chars), GLYPH_ROWS, GLYPH_COLS, dtype=torch.bool, device=device)
    for i, ch in enumerate(chars):
        for r, row in enumerate(_FONT_5X7[ch]):
            for c, value in enumerate(row):
                if value == "1":
                    glyph_lit[i, r, c] = True
    return glyph_lit, char_to_index


def slot_glyph_indices(
    slot_labels: tuple[str, ...], char_to_index: dict[str, int], device: torch.device | str
) -> torch.Tensor:
    """Map each global key slot to a glyph index (``-1`` if its label has no drawable glyph)."""
    idx = torch.full((len(slot_labels),), -1, dtype=torch.long, device=device)
    for slot, label in enumerate(slot_labels):
        ch = (label or "").strip().upper()[:1]
        if ch in char_to_index:
            idx[slot] = char_to_index[ch]
    return idx


def pixel_offsets(pixel_size: float, right: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Per-pixel offset ``(7, 5, 3)`` from a glyph's top-left origin (cols go ``+right``, rows go ``-up``)."""
    rows = torch.arange(GLYPH_ROWS, device=right.device, dtype=torch.float32)
    cols = torch.arange(GLYPH_COLS, device=right.device, dtype=torch.float32)
    along = cols[None, :, None] * pixel_size * right
    down = -(rows[:, None, None]) * pixel_size * up
    return along + down


def cell_offsets(
    max_len: int, pixel_size: float, letter_gap: float, row_gap: float, right: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    """Top-left origin ``(2, max_len, 3)`` of each letter cell relative to the banner anchor.

    Row 0 is the target row (above), row 1 the typed row (below); both are horizontally centered.
    """
    advance = (GLYPH_COLS + letter_gap) * pixel_size
    j = torch.arange(max_len, device=right.device, dtype=torch.float32)
    u_start = -0.5 * (max_len - 1) * advance - 0.5 * GLYPH_COLS * pixel_size
    u = (u_start + j * advance)[:, None] * right  # (max_len, 3)
    glyph_h = GLYPH_ROWS * pixel_size
    v_target = (0.5 * row_gap + glyph_h) * up
    v_typed = (-0.5 * row_gap) * up
    return torch.stack([v_target[None, :] + u, v_typed[None, :] + u], dim=0)


def compute_letter_markers(
    rows_glyph: torch.Tensor,
    rows_color: torch.Tensor,
    glyph_lit: torch.Tensor,
    cell_origin: torch.Tensor,
    pixel_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand per-letter glyph/color/origin into per-lit-pixel ``(translations, marker_indices)``.

    Args:
        rows_glyph: Glyph index per letter cell, ``-1`` for empty cells. Shape ``(N, R, L)``.
        rows_color: Marker prototype index per letter cell (unused where empty). Shape ``(N, R, L)``.
        glyph_lit: Bitmap font lit-pixel table. Shape ``(num_chars, 7, 5)`` bool.
        cell_origin: World position of each letter cell's top-left origin. Shape ``(N, R, L, 3)``.
        pixel_offset: Per-pixel offset from a glyph origin. Shape ``(7, 5, 3)``.

    Returns:
        ``translations`` of shape ``(M, 3)`` and integer ``marker_indices`` of shape ``(M,)`` for the
        lit pixels only.
    """
    valid = rows_glyph >= 0
    lit = glyph_lit[rows_glyph.clamp(min=0)] & valid[..., None, None]  # (N, R, L, 7, 5)
    pos = cell_origin[..., None, None, :] + pixel_offset  # (N, R, L, 7, 5, 3)
    color = rows_color[..., None, None].expand_as(lit)
    selected = lit.reshape(-1)
    return pos.reshape(-1, 3)[selected], color.reshape(-1)[selected]
