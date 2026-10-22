# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton GL HUD section, checkpoint discovery window, and ghost tinting for Policy Ghosts."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import warp as wp

from isaaclab.visualizers.policy_ghost import PolicyGhostCheckpoint, PolicyGhostController

MAX_GHOST_OPACITY = 0.95
"""Highest ghost opacity; Newton re-batches shapes that become fully opaque, so the slider stops short of it."""

TEXTURE_ENABLE = 3
"""Component of a Newton instance material, ``(roughness, metallic, checker, texture)``, that enables the texture."""

METALLIC = 1
"""Component of a Newton instance material that holds the metallic factor."""


class NewtonPolicyGhostPanel:
    """Draws the Policy Ghosts HUD for a :class:`~isaaclab.visualizers.policy_ghost.PolicyGhostController`.

    The panel owns the checkpoint discovery window and tints each ghost slot's shapes through the Newton model's
    per-shape color and opacity arrays, which the viewer re-reads every frame. A ghost at full tint also drops its
    mesh textures and metallic shading, so no native material color shows through.
    """

    def __init__(self, controller: PolicyGhostController, restart: Callable[[bool], None] | None = None):
        """Create the panel.

        Args:
            controller: Playback-loop side of Policy Ghosts.
            restart: Callback that restarts playback with Policy Ghosts enabled (``True``) or disabled (``False``).
                When omitted, the panel has no Disable button.
        """
        self.controller = controller
        self._restart = restart
        self._window_open = False
        self._discovered: list[PolicyGhostCheckpoint] = []
        self._message = ""
        self._base_colors: np.ndarray | None = None
        self._tint_signature: tuple | None = None
        self._shown_signature: tuple | None = None
        self._native_materials: dict[int, tuple[Any, np.ndarray]] = {}
        self._material_signature: tuple | None = None

    @property
    def layout(self):
        """Environment-to-slot layout of the controller."""
        return self.controller.layout

    def draw(self, imgui: Any) -> None:
        """Draw the Policy Ghosts section and, when open, the checkpoint discovery window."""
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        ghosts_open = imgui.collapsing_header("Policy Ghosts")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Compare several checkpoints of this task in one window")
        if ghosts_open:
            imgui.separator()
            layout = self.layout
            imgui.text(f"Simulating {layout.num_slots} slots x {layout.slice_size} envs")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Every slot is simulated, even when empty. With few envs per slot the extra cost is small;"
                    " it grows with the total environment count."
                )
            for slot in layout.slots:
                if slot.active:
                    self._draw_slot(imgui, slot)
            empty = sum(not slot.active for slot in layout.slots[1:])
            if not empty:
                imgui.begin_disabled()
            if imgui.button(f"Add Checkpoint...  ({empty} free)"):
                self._open_window()
            if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                imgui.set_tooltip("Load another checkpoint of this task into a free slot")
            if not empty:
                imgui.end_disabled()
            if self._message:
                imgui.text_wrapped(self._message)
            if self._restart is not None:
                imgui.separator()
                if imgui.button("Disable Policy Ghosts (restarts)"):
                    self._restart(False)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Restarts the viewer with only the reference environments")
        if self._window_open:
            self._draw_window(imgui)

    def apply_tint(self, model: Any) -> None:
        """Write each slot's tint color and opacity to the shapes of its worlds when anything changed."""
        tint_signature = tuple((slot.shown, slot.opacity, slot.tint) for slot in self.layout.slots)
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
            if slot.shown:
                mask = slot_of_shape == slot.index
                colors[mask] = self._base_colors[mask] * (1.0 - slot.tint) + np.asarray(slot.color) * slot.tint
                opacity[mask] = slot.opacity
        model.shape_color = wp.array(colors, dtype=wp.vec3, device=model.shape_color.device)
        model.shape_opacity = wp.array(opacity, dtype=wp.float32, device=model.shape_color.device)

    def apply_materials(self, viewer: Any) -> None:
        """Flatten the shading of ghosts at full tint so that no native texture or metallic look remains.

        Newton rebuilds its shape batches, and with them the native materials, whenever the visible worlds change, so
        the native materials are re-read from every batch the panel has not seen yet.

        Args:
            viewer: Newton GL viewer whose shape batches are modified.
        """
        batches = getattr(viewer, "_shape_instances", None)
        if not batches:
            return
        flat = tuple((slot.shown and slot.tint >= 1.0) for slot in self.layout.slots)
        signature = (flat, tuple(id(batch) for batch in batches.values()))
        if signature == self._material_signature:
            return
        self._material_signature = signature
        known = {id(batch) for batch in batches.values()}
        self._native_materials = {k: v for k, v in self._native_materials.items() if k in known}
        for batch in batches.values():
            if id(batch) not in self._native_materials:
                self._native_materials[id(batch)] = (batch, batch.materials.numpy().copy())
            native = self._native_materials[id(batch)][1]
            worlds = batch.worlds.numpy()
            slot_of_instance = np.where(worlds >= 0, worlds // self.layout.slice_size, -1)
            materials = native.copy()
            for slot in self.layout.slots[1:]:
                if flat[slot.index]:
                    mask = slot_of_instance == slot.index
                    materials[mask, TEXTURE_ENABLE] = 0.0
                    materials[mask, METALLIC] = 0.0
            batch.materials = wp.array(materials, dtype=wp.vec4, device=batch.materials.device)
        # Newton uploads instance materials only while the model is flagged as changed
        viewer.model_changed = True

    def consume_slots_changed(self) -> bool:
        """Return whether slots were filled, emptied, or shown/hidden since the last call."""
        signature = tuple(slot.shown for slot in self.layout.slots)
        changed = signature != self._shown_signature
        self._shown_signature = signature
        return changed

    def _draw_slot(self, imgui: Any, slot: Any) -> None:
        imgui.push_id(slot.index)
        color = slot.color if slot.color is not None else (0.8, 0.8, 0.8)
        imgui.color_button(
            "##swatch", imgui.ImVec4(*color, 1.0), imgui.ColorEditFlags_.no_tooltip, imgui.ImVec2(14, 14)
        )
        imgui.same_line()
        _, slot.visible = imgui.checkbox("##show", slot.visible)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Show or hide this checkpoint")
        imgui.same_line()
        role = "Reference" if slot.index == 0 else f"Ghost {slot.index}"
        imgui.text(f"{role}: {slot.label}")
        if slot.index > 0:
            imgui.set_next_item_open(False, imgui.Cond_.appearing)
            if imgui.tree_node("Settings"):
                _, slot.opacity = imgui.slider_float("Opacity", slot.opacity, 0.1, MAX_GHOST_OPACITY)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Opacity of this ghost")
                _, slot.tint = imgui.slider_float("Tint", slot.tint, 0.0, 1.0)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("1.0 replaces all native colors and textures with the ghost color")
                if imgui.button("Remove"):
                    self.controller.remove_ghost(slot.index)
                    self._message = ""
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Unload this checkpoint and free its slot")
                imgui.tree_pop()
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


def draw_policy_ghost_launcher(imgui: Any, restart: Callable[[bool], None]) -> None:
    """Draw the collapsed Policy Ghosts section of a run that started without the feature.

    Args:
        imgui: The ``imgui`` module.
        restart: Callback that restarts playback with Policy Ghosts enabled (``True``) or disabled (``False``).
    """
    imgui.set_next_item_open(False, imgui.Cond_.appearing)
    ghosts_open = imgui.collapsing_header("Policy Ghosts")
    if imgui.is_item_hovered():
        imgui.set_tooltip("Compare several checkpoints of this task in one window")
    if ghosts_open:
        imgui.separator()
        imgui.text_wrapped(
            "Compare several checkpoints of this task side by side. Enabling restarts the viewer with several"
            " times the environments, which lowers the step rate."
        )
        if imgui.button("Enable Policy Ghosts (restarts)"):
            restart(True)
