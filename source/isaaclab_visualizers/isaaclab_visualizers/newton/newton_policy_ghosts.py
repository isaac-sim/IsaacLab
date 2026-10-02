# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton GL HUD section, checkpoint discovery window, and ghost tinting for Policy Ghosts."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import warp as wp

from isaaclab.visualizers.policy_ghost import PolicyGhostCheckpoint, PolicyGhostController

MAX_GHOST_OPACITY = 0.95
"""Highest ghost opacity; Newton re-batches shapes that become fully opaque, so the slider stops short of it."""


class NewtonPolicyGhostPanel:
    """Draws the Policy Ghosts HUD for a :class:`~isaaclab.visualizers.policy_ghost.PolicyGhostController`.

    The panel owns the checkpoint discovery window and tints each ghost slot's shapes through the Newton model's
    per-shape color and opacity arrays, which the viewer re-reads every frame.
    """

    def __init__(self, controller: PolicyGhostController):
        """Create the panel.

        Args:
            controller: Playback-loop side of Policy Ghosts.
        """
        self.controller = controller
        self._window_open = False
        self._discovered: list[PolicyGhostCheckpoint] = []
        self._message = ""
        self._base_colors: np.ndarray | None = None
        self._tint_signature: tuple | None = None
        self._active_signature: tuple | None = None

    @property
    def layout(self):
        """Environment-to-slot layout of the controller."""
        return self.controller.layout

    def draw(self, imgui: Any) -> None:
        """Draw the Policy Ghosts section and, when open, the checkpoint discovery window."""
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("Policy Ghosts"):
            imgui.separator()
            layout = self.layout
            imgui.text(f"Simulating {layout.num_slots} slots x {layout.slice_size} envs")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Every slot is simulated, even when empty. With few envs per slot the extra cost is small;"
                    " it grows with the total environment count."
                )
            for slot in layout.slots:
                self._draw_slot(imgui, slot)
            has_empty = any(not slot.active for slot in layout.slots[1:])
            if not has_empty:
                imgui.begin_disabled()
            if imgui.button("Add Checkpoint..."):
                self._open_window()
            if not has_empty:
                imgui.end_disabled()
            if self._message:
                imgui.text_wrapped(self._message)
        if self._window_open:
            self._draw_window(imgui)

    def apply_tint(self, model: Any) -> None:
        """Write each slot's tint and opacity to the shapes of its worlds when anything changed."""
        tint_signature = tuple((slot.active, slot.opacity, slot.tint) for slot in self.layout.slots)
        if tint_signature == self._tint_signature or model.shape_color is None or model.shape_world is None:
            return
        self._tint_signature = tint_signature
        if self._base_colors is None:
            self._base_colors = model.shape_color.numpy().copy()
        worlds = model.shape_world.numpy()
        slot_of_shape = np.where(worlds >= 0, worlds // self.layout.slice_size, -1)
        colors = self._base_colors.copy()
        opacity = np.ones(len(worlds), dtype=np.float32)
        for slot in self.layout.slots[1:]:
            if slot.active:
                mask = slot_of_shape == slot.index
                colors[mask] = self._base_colors[mask] * (1.0 - slot.tint) + np.asarray(slot.color) * slot.tint
                opacity[mask] = slot.opacity
        model.shape_color = wp.array(colors, dtype=wp.vec3, device=model.shape_color.device)
        model.shape_opacity = wp.array(opacity, dtype=wp.float32, device=model.shape_color.device)

    def consume_slots_changed(self) -> bool:
        """Return whether slots were filled or emptied since the last call, which changes the visible worlds."""
        signature = tuple(slot.active for slot in self.layout.slots)
        changed = signature != self._active_signature
        self._active_signature = signature
        return changed

    def _draw_slot(self, imgui: Any, slot: Any) -> None:
        imgui.push_id(slot.index)
        color = slot.color if slot.color is not None else (0.8, 0.8, 0.8)
        imgui.color_button(
            "##swatch", imgui.ImVec4(*color, 1.0), imgui.ColorEditFlags_.no_tooltip, imgui.ImVec2(14, 14)
        )
        imgui.same_line()
        role = "Reference" if slot.index == 0 else f"Ghost {slot.index}"
        imgui.text(f"{role}: {slot.label if slot.active else '(empty)'}")
        if slot.index > 0 and slot.active:
            changed, opacity = imgui.slider_float("Opacity", slot.opacity, 0.1, MAX_GHOST_OPACITY)
            if changed:
                slot.opacity = opacity
            changed, tint = imgui.slider_float("Tint", slot.tint, 0.0, 1.0)
            if changed:
                slot.tint = tint
            if imgui.button("Remove"):
                self.controller.remove_ghost(slot.index)
                self._message = ""
        imgui.pop_id()

    def _open_window(self) -> None:
        self._window_open = True
        self._refresh()

    def _refresh(self) -> None:
        self._discovered = self.controller.discover()

    def _draw_window(self, imgui: Any) -> None:
        """Draw the discovery window: compatible checkpoints of the running task, grouped by run."""
        imgui.set_next_window_pos(imgui.ImVec2(400, 60), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(520, 420), imgui.Cond_.first_use_ever)
        expanded, self._window_open = imgui.begin("Policy Ghost Checkpoints", self._window_open)
        if expanded:
            imgui.text_wrapped(f"Searching: {self.controller.checkpoint_root}")
            if imgui.button("Refresh"):
                self._refresh()
            imgui.separator()
            loaded = {slot.checkpoint.path for slot in self.layout.slots if slot.active}
            if not self._discovered:
                imgui.text_wrapped("No compatible checkpoints found. Runs need a run.json written by training.")
            runs: dict[str, list[PolicyGhostCheckpoint]] = {}
            for checkpoint in self._discovered:
                runs.setdefault(checkpoint.run, []).append(checkpoint)
            for run, checkpoints in runs.items():
                imgui.set_next_item_open(len(runs) == 1, imgui.Cond_.appearing)
                if imgui.tree_node(f"{run}##run"):
                    for checkpoint in checkpoints:
                        taken = checkpoint.path in loaded
                        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(checkpoint.modified))
                        label = f"{checkpoint.name}   {stamp}{'   (loaded)' if taken else ''}##{checkpoint.path}"
                        clicked, _ = imgui.selectable(label, taken)
                        if clicked and not taken:
                            error = self.controller.add_ghost(checkpoint)
                            self._message = error or ""
                            if error is None:
                                self._window_open = False
                    imgui.tree_pop()
        imgui.end()
