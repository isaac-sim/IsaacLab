# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


class PolicyDebugUiController:
    """ImGui sidebar controls for :class:`PolicyDebugManager`."""

    def __init__(self, manager):
        self.manager = manager

    def draw(self, imgui) -> None:
        imgui.separator()
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Policy Debug"):
            return
        changed, overlay = imgui.checkbox("Transparent overlay", self.manager.overlay)
        if changed:
            self.manager.set_overlay(overlay)
        changed, opacity = imgui.slider_float("Ghost opacity", self.manager.ghost_opacity, 0.0, 1.0)
        if changed:
            self.manager.set_ghost_opacity(opacity)
        imgui.text(f"{len(self.manager.active)} / {self.manager.cfg.max_policies} active")

        began_child = False
        try:
            began_child = bool(imgui.begin_child("PolicyDebugCheckpoints", imgui.ImVec2(0, 260), True))
        except TypeError:
            began_child = bool(imgui.begin_child("PolicyDebugCheckpoints", (0, 260), True))
        if began_child:
            for entry in self.manager.catalog.entries:
                checked = entry.path in self.manager.active
                changed, enabled = imgui.checkbox(f"##{entry.path}", checked)
                imgui.same_line()
                iteration = entry.iteration if entry.iteration is not None else entry.filename_iteration
                status = entry.error or entry.status
                slot = self.manager.active.get(entry.path)
                if slot is not None:
                    tint = self.manager.ghost_tint(slot)
                    if tint is not None:
                        _draw_color_swatch(imgui, f"##tint-{entry.path}", tint)
                        imgui.same_line()
                    marker = f"[{slot.slot} reference]" if self.manager.overlay and tint is None else f"[{slot.slot}]"
                else:
                    marker = "[-]"
                imgui.align_text_to_frame_padding()
                imgui.text(f"{marker} iter {iteration}: {entry.path.name} — {status}")
                if changed:
                    self.manager.set_checkpoint_enabled(entry, enabled)
        imgui.end_child()

        if imgui.button("Rescan"):
            self.manager.rescan()


def _draw_color_swatch(imgui, item_id: str, tint: tuple[float, float, float]) -> None:
    """Draw a compact, non-editable checkpoint color swatch."""
    color = imgui.ImVec4(float(tint[0]), float(tint[1]), float(tint[2]), 1.0)
    size = imgui.ImVec2(0.0, 0.0)
    flags = imgui.ColorEditFlags_.no_tooltip | imgui.ColorEditFlags_.no_drag_drop
    imgui.color_button(item_id, color, flags, size)
