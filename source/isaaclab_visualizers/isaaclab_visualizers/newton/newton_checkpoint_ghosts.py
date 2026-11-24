# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton GL HUD section, checkpoint discovery window, and ghost tinting for Checkpoint Ghosts."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import warp as wp

from isaaclab.visualizers.checkpoint_ghost import CheckpointGhostCheckpoint, CheckpointGhostController

SECTION_TOOLTIP = (
    "Checkpoint visualization: replays other checkpoints of this task from the same starts as translucent copies,"
    " to compare their behavior."
)
"""Tooltip of the Checkpoint Ghosts section header, shown whether or not the feature is enabled."""

MAX_GHOST_OPACITY = 1.0
"""Highest value of the opacity slider."""

OPAQUE_LIMIT = 0.998
"""Highest opacity written to the model. Newton re-batches shapes at 0.999 and above, so the slider's top stops here."""


def display_opacity(slider: float) -> float:
    """Return the opacity Newton draws for a slider value.

    The curve eases toward opaque, so the slider's upper half looks nearly solid instead of staying see-through, and
    its top is just under the level where Newton switches a shape to its opaque path.
    """
    return min(1.0 - (1.0 - float(slider)) ** 2, OPAQUE_LIMIT)


TEXTURE_ENABLE = 3
"""Component of a Newton instance material, ``(roughness, metallic, checker, texture)``, that enables the texture."""

METALLIC = 1
"""Component of a Newton instance material that holds the metallic factor."""


class NewtonCheckpointGhostPanel:
    """Draws the Checkpoint Ghosts HUD for a :class:`~isaaclab.visualizers.checkpoint_ghost.CheckpointGhostController`.

    The panel owns the checkpoint discovery window and tints each ghost slot's shapes through the Newton model's
    per-shape color and opacity arrays, which the viewer re-reads every frame. A ghost at full tint also drops its
    mesh textures and metallic shading, so no native material color shows through.
    """

    def __init__(self, controller: CheckpointGhostController):
        """Create the panel.

        Args:
            controller: Playback-loop side of Checkpoint Ghosts.
        """
        self.controller = controller
        self._window_open = False
        self._replacing: int | None = None
        """Ghost slot the checkpoint window is choosing a replacement for, or None when it fills a free slot."""
        self._directory: str | None = None
        """Directory the checkpoint window shows; it keeps the last one across openings and selections."""
        self._folders: list[str] = []
        self._other_tasks: dict[str, str] = {}
        """Task of each listed folder (and of the current one, under ``""``) that is a run of another task."""
        self._files: list[CheckpointGhostCheckpoint] = []
        self._browse_error = ""
        self._message = ""
        self._base_colors: np.ndarray | None = None
        self._base_opacity: np.ndarray | None = None
        self._tint_signature: tuple | None = None
        self._shown_signature: tuple | None = None
        self._native_materials: dict[int, tuple[Any, np.ndarray]] = {}
        self._material_signature: tuple | None = None

    @property
    def layout(self):
        """Environment-to-slot layout of the controller."""
        return self.controller.layout

    def draw(self, imgui: Any, rtx_view: bool = False, owner: Any = None) -> None:
        """Draw the Checkpoint Ghosts section and, when open, the checkpoint discovery window.

        Args:
            imgui: The ImGui module.
            rtx_view: Whether the window shows the RTX view, where the controls are disabled.
            owner: The visualizer, used to report how many environments are drawn.
        """
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        ghosts_open = imgui.collapsing_header("Checkpoint Ghosts")
        if imgui.is_item_hovered():
            imgui.set_tooltip(SECTION_TOOLTIP)
        if ghosts_open:
            imgui.separator()
            layout = self.layout
            if rtx_view:
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.55, 0.7, 0.95, 0.85))
                imgui.text_wrapped("RTX: switch the Renderer to OpenGL to use Checkpoint Ghosts.")
                imgui.pop_style_color()
                imgui.begin_disabled()
            total = layout.num_envs
            loaded = len(layout.active_slots)
            if owner is not None:
                per_slot = owner.visible_env_count
                shown = sum(slot.shown for slot in layout.slots)
                imgui.text(f"Showing {per_slot * shown}/{total} envs: {per_slot} per slot x {shown} ckpts")
            else:
                imgui.text(f"Simulating {total} envs: {layout.slice_size} per slot x {loaded} ckpts")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"The {total} envs are split evenly among the checkpoints you load ({layout.slice_size} each now),"
                    " so adding or removing one resizes every slot and restarts. Lower Visible Envs in Playback to draw"
                    " fewer per checkpoint."
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
            if rtx_view:
                imgui.end_disabled()
        if self._window_open and not rtx_view:
            self._draw_window(imgui)

    def apply_tint(self, model: Any) -> None:
        """Write each slot's tint color and opacity to the shapes of its worlds when anything changed."""
        tint_signature = (
            self._layout_signature(),
            *((slot.shown, slot.opacity, slot.tint, slot.color) for slot in self.layout.slots),
        )
        if tint_signature == self._tint_signature or model.shape_color is None or model.shape_world is None:
            return
        self._tint_signature = tint_signature
        if self._base_colors is None:
            self._base_colors = model.shape_color.numpy().copy()
            opacity = getattr(model, "shape_opacity", None)
            self._base_opacity = None if opacity is None else opacity.numpy().copy()
        worlds = model.shape_world.numpy()
        slot_of_shape = self.layout.slot_of_env(worlds)
        colors = self._base_colors.copy()
        opacity = np.ones(len(worlds), dtype=np.float32) if self._base_opacity is None else self._base_opacity.copy()
        for slot in self.layout.slots[1:]:
            if slot.shown:
                mask = slot_of_shape == slot.index
                colors[mask] = self._base_colors[mask] * (1.0 - slot.tint) + np.asarray(slot.color) * slot.tint
                opacity[mask] = display_opacity(slot.opacity)
        model.shape_color = wp.array(colors, dtype=wp.vec3, device=model.shape_color.device)
        model.shape_opacity = wp.array(opacity, dtype=wp.float32, device=model.shape_color.device)

    def apply_materials(self, viewer: Any) -> None:
        """Drop the native textures of tinted ghosts and fade their metallic shading out as the tint rises.

        Textures multiply the shape color and switch on or off in Newton's shader, so keeping them for a nearly full
        tint made the last step to 1.0 jump. Dropping them for any tint above 0 keeps the whole slider smooth.

        Newton rebuilds its shape batches, and with them the native materials, whenever the visible worlds change, so
        the native materials are re-read from every batch the panel has not seen yet.

        Args:
            viewer: Newton GL viewer whose shape batches are modified.
        """
        batches = getattr(viewer, "_shape_instances", None)
        if not batches:
            return
        flat = tuple((slot.tint if slot.shown and slot.tint > 0.0 else 0.0) for slot in self.layout.slots)
        signature = (flat, self._layout_signature(), tuple(id(batch) for batch in batches.values()))
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
            slot_of_instance = self.layout.slot_of_env(worlds)
            materials = native.copy()
            for slot in self.layout.slots[1:]:
                if flat[slot.index]:
                    mask = slot_of_instance == slot.index
                    materials[mask, TEXTURE_ENABLE] = 0.0
                    materials[mask, METALLIC] = native[mask, METALLIC] * (1.0 - flat[slot.index])
            batch.materials = wp.array(materials, dtype=wp.vec4, device=batch.materials.device)
        # Newton uploads instance materials only while the model is flagged as changed
        viewer.model_changed = True

    def _layout_signature(self) -> tuple:
        """Identify how environments map to slots, which changes whenever a checkpoint is loaded or unloaded."""
        return (self.layout.slice_size, tuple(slot.index for slot in self.layout.active_slots))

    def consume_slots_changed(self) -> bool:
        """Return whether slots were filled, emptied, or shown/hidden since the last call."""
        signature = (self._layout_signature(), tuple(slot.shown for slot in self.layout.slots))
        changed = signature != self._shown_signature
        self._shown_signature = signature
        return changed

    def _draw_slot(self, imgui: Any, slot: Any) -> None:
        imgui.push_id(slot.index)
        if slot.index == 0:
            # The reference keeps its own colors and is always shown, so it gets an outline, not a swatch or checkbox.
            corner = imgui.get_cursor_screen_pos()
            imgui.dummy(imgui.ImVec2(14, 14))
            imgui.get_window_draw_list().add_rect(
                corner,
                imgui.ImVec2(corner.x + 14, corner.y + 14),
                imgui.get_color_u32(imgui.ImVec4(0.85, 0.85, 0.85, 1.0)),
                0.0,
                0,
                1.5,
            )
            if imgui.is_item_hovered():
                imgui.set_tooltip("The reference: drawn in its own colors, opaque, and always shown")
        else:
            color = slot.color if slot.color is not None else (0.8, 0.8, 0.8)
            imgui.color_button(
                "##swatch", imgui.ImVec4(*color, 1.0), imgui.ColorEditFlags_.no_tooltip, imgui.ImVec2(14, 14)
            )
        imgui.same_line()
        role = "Reference" if slot.index == 0 else f"Ghost {slot.index}"
        imgui.text(f"{role}: {slot.label}")
        if slot.index > 0:
            _, slot.visible = imgui.checkbox("##show", slot.visible)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Show or hide this checkpoint")
            imgui.same_line()
            if imgui.button("Change..."):
                self._open_window(slot.index)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Replace this ghost's checkpoint with another one")
            imgui.same_line()
            if imgui.button("Remove"):
                self.controller.remove_ghost(slot.index)
                self._message = ""
            if imgui.is_item_hovered():
                imgui.set_tooltip("Unload this checkpoint and free its slot")
            imgui.set_next_item_open(False, imgui.Cond_.appearing)
            if imgui.tree_node("Appearance"):
                _, slot.opacity = imgui.slider_float("Opacity", slot.opacity, 0.1, MAX_GHOST_OPACITY)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("How solid this ghost looks. 1.0 is nearly opaque")
                _, slot.tint = imgui.slider_float("Tint", slot.tint, 0.0, 1.0)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Blends toward the ghost color. Any tint above 0 drops native textures,"
                        " and metal shading fades out"
                    )
                imgui.tree_pop()
        imgui.pop_id()

    def _open_window(self, replacing: int | None = None) -> None:
        """Open the checkpoint window, to fill a free slot or, with ``replacing``, to change that ghost's checkpoint."""
        self._replacing = replacing
        self._window_open = True
        self._refresh()

    def _refresh(self) -> None:
        """List the window's directory again."""
        if self._directory is None:
            self._directory = self.controller.checkpoint_root
        try:
            self._folders, self._files = self.controller.browse(self._directory)
            self._browse_error = ""
            self._other_tasks = {
                name: task
                for name, task in (
                    ("", self.controller.other_task(self._directory)),
                    *((f, self.controller.other_task(os.path.join(self._directory, f))) for f in self._folders),
                )
                if task is not None
            }
        except OSError as error:
            self._folders, self._files, self._other_tasks = [], [], {}
            self._browse_error = f"Cannot read this folder: {error.strerror or error}"

    def _go(self, directory: str) -> None:
        self._directory = os.path.abspath(directory)
        self._refresh()

    def _draw_window(self, imgui: Any) -> None:
        """Draw the checkpoint window: a folder browser that starts in the experiment's log directory."""
        imgui.set_next_window_pos(imgui.ImVec2(400, 60), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(520, 420), imgui.Cond_.first_use_ever)
        expanded, self._window_open = imgui.begin("Choose Checkpoint", self._window_open)
        if expanded:
            if self._replacing is not None:
                imgui.text(f"Replacing Ghost {self._replacing}")
            if imgui.button("Up"):
                self._go(os.path.dirname(self._directory))
            if imgui.is_item_hovered():
                imgui.set_tooltip("Go to the parent folder")
            imgui.same_line()
            if imgui.button("Refresh"):
                self._refresh()
            if imgui.is_item_hovered():
                imgui.set_tooltip("List this folder again, such as for checkpoints training just saved")
            imgui.same_line()
            if imgui.button("Experiment Folder"):
                self._go(self.controller.checkpoint_root)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Return to the folder this task's runs are saved in")
            imgui.set_next_item_width(-1)
            submitted, text = imgui.input_text("##path", self._directory, imgui.InputTextFlags_.enter_returns_true)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Type a folder anywhere on this machine and press Enter")
            if submitted:
                self._go(os.path.expanduser(text))
            imgui.separator()
            if self._browse_error:
                imgui.text_wrapped(self._browse_error)
            elif not self._folders and not self._files:
                imgui.text_wrapped("No folders or checkpoint files here.")
            if "" in self._other_tasks:
                imgui.text_colored(
                    imgui.ImVec4(1.0, 0.8, 0.2, 1.0),
                    f"This run is {self._other_tasks['']}, not this task. Its checkpoints will not fit.",
                )
            for folder in self._folders:
                other = self._other_tasks.get(folder)
                if other is not None:
                    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.55, 0.55, 0.55, 1.0))
                clicked, _ = imgui.selectable(f"{folder}/" + (f"   ({other})" if other else "") + "##folder", False)
                if other is not None:
                    imgui.pop_style_color()
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("A run of a different task, so its checkpoints will not fit this policy")
                if clicked:
                    self._go(os.path.join(self._directory, folder))
                    break  # the listing just changed
            loaded = {slot.checkpoint.path for slot in self.layout.slots if slot.active}
            for checkpoint in self._files:
                taken = checkpoint.path in loaded
                stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(checkpoint.modified))
                label = f"{checkpoint.name}   {stamp}{'   (loaded)' if taken else ''}##{checkpoint.path}"
                clicked, _ = imgui.selectable(label, taken)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(
                        "Already loaded" if taken else "Load this checkpoint as a tinted ghost over the reference"
                    )
                if clicked and not taken:
                    if self._replacing is None:
                        error = self.controller.add_ghost(checkpoint)
                    else:
                        error = self.controller.replace_ghost(self._replacing, checkpoint)
                    self._message = error or ""
                    if error is None:
                        self._window_open = False
        imgui.end()


def draw_checkpoint_ghost_offer(imgui: Any) -> None:
    """Draw the greyed-out Checkpoint Ghosts section of a play run that started without ``--checkpoint_ghosts``.

    Args:
        imgui: The ``imgui`` module.
    """
    imgui.set_next_item_open(False, imgui.Cond_.appearing)
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.6, 0.6, 0.6, 1.0))
    ghosts_open = imgui.collapsing_header("Checkpoint Ghosts (not enabled)")
    imgui.pop_style_color()
    if imgui.is_item_hovered():
        imgui.set_tooltip(SECTION_TOOLTIP)
    if ghosts_open:
        imgui.separator()
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.6, 0.6, 0.6, 1.0))
        imgui.text_wrapped(
            "Not enabled. Add --checkpoint_ghosts to play to use it."
            " It needs extra environments, so plain play skips it."
        )
        imgui.pop_style_color()
