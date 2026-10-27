# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open and play one ``.viser`` animation next to the simulation, from the Recording section.

The animation is drawn as a tinted ghost over the live robots. It plays and pauses with the Playback section's pause
button, and its timeline can be dragged when no live checkpoint ghost is loaded.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .newton_replay import NewtonReplayPlayer, list_recordings

if TYPE_CHECKING:
    from .newton_visualizer import NewtonGLVisualizer

GHOST_COLOR = (0.1, 0.9, 1.0)
"""Tint color of the animation."""

DEFAULT_OPACITY = 0.65
"""Opacity of a newly opened animation."""


class NewtonAnimationPanel:
    """One loaded animation of a Newton window, and the controls that open and play it."""

    def __init__(self, owner: NewtonGLVisualizer, folder_getter: Any):
        """Create the panel.

        Args:
            owner: Visualizer whose window draws the animation.
            folder_getter: Callable returning the folder the Open window lists by default.
        """
        self._owner = owner
        self._folder_getter = folder_getter
        self.player = NewtonReplayPlayer(lambda: owner._viewer)
        self.player.color, self.player.tint, self.player.opacity = GHOST_COLOR, 1.0, DEFAULT_OPACITY
        self.folder = ""
        """Folder the Open window lists; empty follows ``folder_getter``."""
        self.player.on_loaded = self._on_loaded
        self._window_open = False
        self._files: list[Path] = []

    @property
    def active(self) -> bool:
        """Whether an animation is loaded."""
        return self.player.active

    @property
    def can_scrub(self) -> bool:
        """Whether the timeline can be dragged: only when no live checkpoint ghost is loaded.

        A live ghost is simulated in real time, so it cannot follow a dragged timeline.
        """
        panel = getattr(self._owner, "_policy_ghost_panel", None)
        return panel is None or not any(slot.active for slot in panel.layout.slots[1:])

    def open(self, path: str) -> None:
        """Load ``path``, replacing the current animation once it is decoded."""
        self.player.load(path)

    def close(self) -> None:
        """Remove the animation from the window."""
        self.player.unload()

    def invalidate(self) -> None:
        """Publish the current frame again, for example after the viewer cleared its instance batches."""
        self.player.invalidate()

    def update(self) -> None:
        """Advance the clock and publish the current frame; call once per viewer frame, inside the frame."""
        viewer = self._owner._viewer
        if viewer is None:
            return
        player = self.player
        player.playing = not viewer._paused_training
        player.hide_static = not self._owner.simulation_hidden
        player.update()

    def _on_loaded(self, recording) -> None:
        """Set up the view for a newly loaded recording; a ghost session already holds the reference."""
        self.player.slot_visible.clear()
        if recording.slots:
            self.player.show_markers = False  # One set of arrows per slot would clutter the overlay.
            self._owner.set_simulation_hidden(True)  # The live robots would only get in the way.

    def draw_window(self, imgui: Any) -> None:
        """Draw the file browser listing the ``.viser`` animations below a folder, while it is open."""
        if not self._window_open:
            return
        imgui.set_next_window_pos(imgui.ImVec2(400, 60), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(520, 360), imgui.Cond_.first_use_ever)
        expanded, self._window_open = imgui.begin("Open Animation", self._window_open)
        if expanded:
            changed, self.folder = imgui.input_text("Folder##animation", self._folder())
            if changed or imgui.button("Refresh##animation"):
                self._files = list_recordings(self._folder())
            imgui.separator()
            if not self._files:
                imgui.text_wrapped("No .viser files found. Record one with Animation > Record.")
            current = self.player.recording.path if self.player.active else ""
            for file in self._files:
                stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(file.stat().st_mtime))
                label = f"{file.name}   {file.stat().st_size / 1e6:.1f} MB   {stamp}##{file}"
                clicked, _ = imgui.selectable(label, str(file) == current)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(str(file))
                if clicked:
                    self.open(str(file))
                    self._window_open = False
        imgui.end()

    def draw_open_button(self, imgui: Any) -> None:
        """Draw the Open button, which lists the available animations."""
        if imgui.button("Open...##animation"):
            self._window_open = True
            self._files = list_recordings(self._folder())
        if imgui.is_item_hovered():
            imgui.set_tooltip("Play a .viser animation in this window, over the simulation")

    def draw(self, imgui: Any) -> None:
        """Draw the loaded animation: its name, timeline, and display options."""
        player = self.player
        if player.message and not player.active:
            imgui.text_wrapped(player.message)
        if not player.active:
            return
        if self._owner.rtx_rendering:
            imgui.text_disabled("Animations are drawn in the OpenGL view only")
        imgui.text(f"{player.name}  ({player.recording.duration:.1f} s)")
        imgui.same_line()
        if imgui.small_button("Close##animation"):
            self.close()
            return
        self._draw_timeline(imgui)
        _, self.player.loop = imgui.checkbox("Loop##animation", player.loop)
        imgui.same_line()
        hidden = self._owner.simulation_hidden
        changed, hidden = imgui.checkbox("Hide live simulation", hidden)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Show only the animation. The simulation keeps running; pause it from Playback to stop it too"
            )
        if changed:
            self._owner.set_simulation_hidden(hidden)
            self.invalidate()
        if player.recording.slots:
            self._draw_slots(imgui)
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if imgui.tree_node("Options##animation"):
            self._draw_options(imgui)
            imgui.tree_pop()

    def _draw_slots(self, imgui: Any) -> None:
        """Draw one row per Policy Ghost slot of a recorded ghost session, with a checkbox to show or hide it."""
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.tree_node("Slots##animation"):
            for slot in self.player.recording.slots:
                imgui.push_id(slot.index)
                color = slot.color if slot.color is not None else (0.8, 0.8, 0.8)
                imgui.color_button(
                    "##swatch", imgui.ImVec4(*color, 1.0), imgui.ColorEditFlags_.no_tooltip, imgui.ImVec2(14, 14)
                )
                imgui.same_line()
                shown = self.player.slot_visible.get(slot.index, True)
                changed, shown = imgui.checkbox("##show", shown)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Show or hide this slot")
                if changed:
                    self.player.slot_visible[slot.index] = shown
                imgui.same_line()
                role = "Reference" if slot.index == 0 else f"Ghost {slot.index}"
                imgui.text(f"{role}: {slot.label}")
                imgui.pop_id()
            imgui.tree_pop()

    def _draw_timeline(self, imgui: Any) -> None:
        player, scrubbable = self.player, self.can_scrub
        duration = player.recording.duration
        if not scrubbable:
            imgui.begin_disabled()
        changed, value = imgui.slider_float(
            "Time##animation", player.time, 0.0, max(duration, 1e-3), f"%.2f / {duration:.2f} s"
        )
        if not scrubbable:
            imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "Drag to jump to a time. This pauses playback; Resume in Playback continues it"
                if scrubbable
                else "The timeline can be dragged only when just an animation is loaded. Live checkpoint ghosts are"
                " simulated in real time and cannot be scrubbed. Remove them to scrub."
            )
        if changed and scrubbable:
            player.seek(value)
            self._owner._viewer._paused_training = True

    def _draw_options(self, imgui: Any) -> None:
        player = self.player
        _, player.speed = imgui.slider_float("Speed##animation", player.speed, 0.1, 4.0, "%.2fx")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Playback speed relative to the recording")
        _, player.show_markers = imgui.checkbox("Show markers##animation", player.show_markers)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Show the recorded visualization markers, such as velocity arrows")
        _, player.opacity = imgui.slider_float("Opacity##animation", player.opacity, 0.1, 1.0)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Make the animation translucent. For a recorded ghost session this applies to the ghost slots"
            )
        _, player.tint = imgui.slider_float("Tint##animation", player.tint, 0.0, 1.0)
        if imgui.is_item_hovered():
            imgui.set_tooltip("1.0 replaces the animation's colors and textures with the ghost color; 0.0 keeps them")
        _, offset = imgui.input_float3("Offset (m)##animation", player.offset.tolist())
        player.offset[:] = offset
        if imgui.is_item_hovered():
            imgui.set_tooltip("Move the animation in the world [m]")
        if imgui.button("Frame##animation"):
            self._owner.frame_animation(player)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Point the camera at the animation")

    def _folder(self) -> str:
        return self.folder or self._folder_getter()
