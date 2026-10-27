# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Animation ghosts: ``.viser`` recordings drawn as tinted ghosts in the Newton window, next to Policy Ghosts.

A recording ghost needs no simulated environments. Several can be open at once, each with its own tint, opacity,
and offset, and they share one timeline. Playback follows the Playback section's pause button.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from isaaclab.visualizers.policy_ghost import POLICY_GHOST_COLORS

from .newton_replay import REPLAY_PREFIX, NewtonReplayPlayer, list_recordings

if TYPE_CHECKING:
    from .newton_visualizer import NewtonGLVisualizer

MAX_ANIMATIONS = 8
"""Most animations open at once."""

DEFAULT_OPACITY = 0.65
"""Opacity of a new animation ghost, as for a new Policy Ghost."""


class AnimationGhostSet:
    """The animation ghosts of one Newton window and the controls that edit them."""

    def __init__(self, owner: NewtonGLVisualizer, folder_getter: Any):
        """Create the set.

        Args:
            owner: Visualizer whose window draws the animations.
            folder_getter: Callable returning the folder the Open window lists by default.
        """
        self._owner = owner
        self._folder_getter = folder_getter
        self.players: list[NewtonReplayPlayer] = []
        self.loop = True
        self.speed = 1.0
        self.folder = ""
        """Folder the Open window lists; empty follows ``folder_getter``."""
        self.message = ""
        self._next_id = 0
        self._window_open = False
        self._files: list[Path] = []

    def _viewer(self):
        return self._owner._viewer

    def __len__(self) -> int:
        return len(self.players)

    @property
    def active(self) -> bool:
        """Whether any animation is open, including one that is still loading."""
        return bool(self.players)

    @property
    def duration(self) -> float:
        """Length of the longest loaded animation [s]."""
        return max((p.recording.duration for p in self.players if p.active), default=0.0)

    @property
    def time(self) -> float:
        """Playback time of the longest loaded animation [s]."""
        longest = max((p for p in self.players if p.active), key=lambda p: p.recording.duration, default=None)
        return 0.0 if longest is None else longest.time

    @property
    def can_scrub(self) -> bool:
        """Whether the timeline can be dragged: only when no live checkpoint ghost is loaded.

        A live ghost is simulated in real time, so it cannot follow a dragged timeline.
        """
        panel = getattr(self._owner, "_policy_ghost_panel", None)
        return panel is None or not any(slot.active for slot in panel.layout.slots[1:])

    def add(self, path: str) -> None:
        """Open ``path`` as a new animation ghost, tinted with the first palette color no other ghost uses."""
        if len(self.players) >= MAX_ANIMATIONS:
            self.message = f"At most {MAX_ANIMATIONS} animations can be open."
            return
        used = {player.color for player in self.players}
        color = next((c for c in POLICY_GHOST_COLORS if c not in used), POLICY_GHOST_COLORS[0])
        player = NewtonReplayPlayer(self._viewer, prefix=f"{REPLAY_PREFIX}/{self._next_id}")
        self._next_id += 1
        player.color, player.tint, player.opacity = color, 1.0, DEFAULT_OPACITY
        player.load(path)
        self.players.append(player)
        self.message = ""

    def remove(self, index: int) -> None:
        """Close one animation and remove its shapes from the window."""
        self.players.pop(index).unload()

    def clear(self) -> None:
        """Close every animation."""
        for player in self.players:
            player.unload()
        self.players.clear()

    def invalidate(self) -> None:
        """Publish every animation's current frame again, for example after the viewer cleared its batches."""
        for player in self.players:
            player.invalidate()

    def seek(self, time_s: float) -> None:
        """Move every animation to ``time_s`` [s]."""
        for player in self.players:
            player.seek(time_s)

    def update(self) -> None:
        """Advance the shared clock and publish the current frames; call once per viewer frame, inside the frame."""
        viewer = self._viewer()
        if viewer is None:
            return
        playing = not viewer._paused_training
        hide_static = not self._owner.simulation_hidden
        for player in list(self.players):
            player.playing, player.loop, player.speed, player.hide_static = playing, self.loop, self.speed, hide_static
            player.update()
            if not player.active and not player.loading and player.message.startswith("Could not"):
                self.message = player.message
                self.players.remove(player)

    def draw(self, imgui: Any) -> None:
        """Draw the animation rows, the Add button, and the shared controls inside the Policy Ghosts section."""
        for index, player in enumerate(self.players):
            self._draw_slot(imgui, index, player)
        full = len(self.players) >= MAX_ANIMATIONS
        if full:
            imgui.begin_disabled()
        if imgui.button("Add Animation..."):
            self._open_window()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip("Draw a recorded .viser animation as a ghost. It needs no extra simulated environments")
        if full:
            imgui.end_disabled()
        if self.message:
            imgui.text_wrapped(self.message)
        if any(player.active for player in self.players):
            imgui.separator()
            self._draw_timeline(imgui)
        if self._window_open:
            self._draw_window(imgui)

    def _draw_slot(self, imgui: Any, index: int, player: NewtonReplayPlayer) -> None:
        imgui.push_id(player.prefix)
        color = player.color if player.color is not None else (0.8, 0.8, 0.8)
        imgui.color_button(
            "##swatch", imgui.ImVec4(*color, 1.0), imgui.ColorEditFlags_.no_tooltip, imgui.ImVec2(14, 14)
        )
        imgui.same_line()
        _, player.visible = imgui.checkbox("##show", player.visible)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Show or hide this animation")
        imgui.same_line()
        imgui.text(f"Animation {index + 1}: {player.name or player.message or 'loading...'}")
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if imgui.tree_node("Settings"):
            _, player.opacity = imgui.slider_float("Opacity", player.opacity, 0.1, 1.0)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Opacity of this animation")
            _, player.tint = imgui.slider_float("Tint", player.tint, 0.0, 1.0)
            if imgui.is_item_hovered():
                imgui.set_tooltip("1.0 replaces all native colors and textures with the ghost color")
            _, offset = imgui.input_float3("Offset (m)", player.offset.tolist())
            player.offset[:] = offset
            if imgui.is_item_hovered():
                imgui.set_tooltip("Move the animation in the world [m]")
            if player.active:
                if imgui.button("Frame"):
                    self._owner.frame_animation(player)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Point the camera at this animation")
                imgui.same_line()
            if imgui.button("Remove"):
                self.remove(index)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Close this animation")
            imgui.tree_pop()
        imgui.pop_id()

    def _draw_timeline(self, imgui: Any) -> None:
        """Draw the shared timeline, loop, and speed controls, and the option to hide the live simulation."""
        duration, scrubbable = self.duration, self.can_scrub
        if not scrubbable:
            imgui.begin_disabled()
        changed, value = imgui.slider_float(
            "Time##animations", self.time, 0.0, max(duration, 1e-3), f"%.2f / {duration:.2f} s"
        )
        if not scrubbable:
            imgui.end_disabled()
        if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "Drag to jump to a time. This pauses playback; Resume in Playback continues it"
                if scrubbable
                else "The timeline can be dragged only when just animations are loaded. Live checkpoint ghosts are"
                " simulated in real time and cannot be scrubbed. Remove them to scrub."
            )
        if changed and scrubbable:
            self.seek(value)
            self._viewer()._paused_training = True
        _, self.loop = imgui.checkbox("Loop##animations", self.loop)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Start over when an animation ends")
        _, self.speed = imgui.slider_float("Speed##animations", self.speed, 0.1, 4.0, "%.2fx")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Playback speed relative to the recording")
        hidden = self._owner.simulation_hidden
        changed, hidden = imgui.checkbox("Hide live simulation", hidden)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Show only the animations. The simulation keeps running; pause it from Playback if you want it to stop"
            )
        if changed:
            self._owner.set_simulation_hidden(hidden)
            self.invalidate()

    def _open_window(self) -> None:
        self._window_open = True
        self._files = list_recordings(self._folder())

    def _folder(self) -> str:
        return self.folder or self._folder_getter()

    def _draw_window(self, imgui: Any) -> None:
        """Draw the file browser listing the ``.viser`` animations below a folder."""
        imgui.set_next_window_pos(imgui.ImVec2(400, 60), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(520, 360), imgui.Cond_.first_use_ever)
        expanded, self._window_open = imgui.begin("Add Animation", self._window_open)
        if expanded:
            changed, self.folder = imgui.input_text("Folder##animations", self._folder())
            if changed or imgui.button("Refresh##animations"):
                self._files = list_recordings(self._folder())
            imgui.separator()
            if not self._files:
                imgui.text_wrapped("No .viser files found. Record one in Recording > Animation > Record.")
            open_paths = {player.recording.path for player in self.players if player.active}
            for file in self._files:
                stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(file.stat().st_mtime))
                label = f"{file.name}   {file.stat().st_size / 1e6:.1f} MB   {stamp}##{file}"
                clicked, _ = imgui.selectable(label, str(file) in open_paths)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(str(file))
                if clicked:
                    self.add(str(file))
                    self._window_open = False
        imgui.end()


def draw_animation_section(imgui: Any, animations: AnimationGhostSet) -> None:
    """Draw the Policy Ghosts section of a run that has no live ghosts, with only the animation controls."""
    imgui.set_next_item_open(False, imgui.Cond_.appearing)
    expanded = imgui.collapsing_header("Policy Ghosts")
    if imgui.is_item_hovered():
        imgui.set_tooltip("Draw recorded animations as ghosts next to the simulation")
    if expanded:
        imgui.separator()
        animations.draw(imgui)
