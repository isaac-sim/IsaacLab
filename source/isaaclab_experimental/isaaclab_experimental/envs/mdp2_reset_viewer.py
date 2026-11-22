# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Show and drive the reset states of an MDP2 environment from a visualizer.

:class:`ResetStateViewer` is the controller: it pins slots, draws reset states as training does, rolls the generator
task, and plays episodes with a policy. Panels for Viser and the Newton viewer only queue requests, from their own
threads; :meth:`ResetStateViewer.update` applies them on the main thread and hands the panels plain values.
"""

from __future__ import annotations

import queue
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab_experimental.mdp2 import mdp

from .mdp2_env import MDP2Env

MODES = ("draw", "slot", "generator")
"""Viewer modes: ``draw`` draws reset states as training does, ``slot`` resets every world to one slot, and
``generator`` shows the proposals of the generator task."""


@dataclass
class ResetStateSnapshot:
    """Plain values that panels show, taken on the main thread after each update."""

    mode: str
    slot: int  # The slot that slot mode restores
    playing: bool
    size: int  # Slots per group of worlds
    num_source_states: int  # Slots from the source; the others record
    world: int  # The world whose reset states the table shows
    started_from: np.ndarray  # (num_worlds,) the slot each world started from, -1 for a default reset
    episode_length: np.ndarray  # (num_worlds,)
    success_rate: np.ndarray  # (size,) of the world's reset states
    episode_counts: np.ndarray  # (size,)
    probabilities: np.ndarray  # (size,)
    accepted: np.ndarray | None = None  # (num_worlds,) after a roll of the generator
    message: str = ""
    labels: list[str] = field(default_factory=list)  # One per world


class ResetStateViewer:
    """Controller for viewing the reset states of a ``reset_to_states`` event and the episodes that start from them.

    Args:
        env: An MDP2 environment, or a wrapper of one.
        event: Name of the ``reset_to_states`` event in the task.
        policy: Returns the actions of a step from the observations; None holds zero actions.
    """

    def __init__(self, env: Any, event: str = "reset", policy: Callable[[dict], torch.Tensor] | None = None) -> None:
        self.env: MDP2Env = env.unwrapped
        self.reset_states = mdp.get_reset_states(self.env.state, f"events.{event}")
        source = self.env.cfg.task.events[event].params.get("source")
        self._generator_cfg = source.task if isinstance(source, mdp.FromTask) else None
        self._generator: mdp.StateGenerator | None = None
        zeros = torch.zeros(self.env.num_envs, self.env.mdp.action_dim, device=self.env.device)
        self.policy = policy or (lambda observations: zeros)
        self.mode, self.slot, self.world, self.playing = "draw", 0, 0, False
        self._accepted: torch.Tensor | None = None
        self._message = ""
        self._requests: queue.SimpleQueue = queue.SimpleQueue()
        self._panels: list[Any] = []
        self._frame_end = time.perf_counter()
        self.snapshot = self._snapshot()

    @property
    def has_generator(self) -> bool:
        """Whether the reset states come from a generator task, which the viewer can roll."""
        return self._generator_cfg is not None

    def request(self, action: str, *args: Any) -> None:
        """Queue an action by method name, such as ``request("show", 3)``, from any thread."""
        self._requests.put((action, args))

    def attach(self) -> int:
        """Add a panel to every Viser and Newton viewer of the environment's simulation, and return how many."""
        for visualizer in self.env.sim.visualizers:
            kind = visualizer.cfg.backend.visualizer_type
            if kind == "viser" and visualizer.server is not None:
                self._panels.append(ViserResetStatePanel(self, visualizer.server))
            elif kind == "newton_gl":
                self._panels.append(NewtonResetStatePanel(self, visualizer))
        self._refresh()
        return len(self._panels)

    def update(self) -> None:
        """Apply the queued requests, step the episodes if playing, render, and refresh the panels.

        Episodes play in real time, one step per ``step_dt``, and a paused view renders at 30 frames per second.
        """
        while not self._requests.empty():
            action, args = self._requests.get()
            getattr(self, action)(*args)
        if self.playing:
            self.env.step(self.policy(self.env.observations))
        else:
            self.env.sim.render()
        self._refresh()
        # Unpaced, episodes would play as fast as the simulation steps.
        self._frame_end = max(self._frame_end + (self.env.step_dt if self.playing else 1.0 / 30.0), time.perf_counter())
        time.sleep(max(0.0, self._frame_end - time.perf_counter()))

    def show(self, slot: int) -> None:
        """Reset every world to its slot ``slot``, paused, and keep resetting there while playing."""
        self.slot = int(np.clip(slot, 0, self.reset_states.size - 1))
        self.reset_states.pin(torch.full((self.env.num_envs,), self.slot, device=self.env.device))
        self._reset("slot")

    def draw(self) -> None:
        """Reset every world to a reset state drawn as in training, paused."""
        self.reset_states.unpin()
        self._reset("draw")

    def roll(self) -> None:
        """Propose a state in every world with the generator task, and mark the ones it rejects."""
        if self._generator_cfg is None:
            self._message = "The reset states do not come from a generator task."
            return
        if self._generator is None:
            self._generator = mdp.StateGenerator(self._generator_cfg, self.env.scene_data, self.env.device)
        self.reset_states.skip_current_episodes()  # the episodes are interrupted
        self._accepted = self._generator.roll()
        self.mode, self.playing = "generator", False
        accepted = int(self._accepted.sum())
        self._message = f"{accepted} of {self.env.num_envs} proposals accepted."

    def play(self) -> None:
        """Step the episodes with the policy. Proposals are not episodes: playing first draws reset states."""
        if self.mode == "generator":
            self.draw()
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def step(self) -> None:
        """Advance the episodes by one step, paused."""
        if self.mode == "generator":
            self.draw()
        self.playing = False
        self.env.step(self.policy(self.env.observations))

    def select(self, world: int) -> None:
        """Show the reset states of a world in the table."""
        self.world = int(np.clip(world, 0, self.env.num_envs - 1))

    def save(self, path: str) -> None:
        """Save the reset states, for later runs to load with ``mdp.FromFile``."""
        self.reset_states.save(Path(path))
        self._message = f"Saved {self.reset_states.size} slots to {path}."

    def _reset(self, mode: str) -> None:
        # Viewing interrupts episodes, whose outcomes say nothing about their reset states.
        self.reset_states.skip_current_episodes()
        self.env.reset()
        self.mode, self.playing, self._accepted, self._message = mode, False, None, ""

    def _snapshot(self) -> ResetStateSnapshot:
        states, world = self.reset_states, self.world
        # Per-slot values are per group of worlds, a group per world by default, or one row if shared.
        groups = np.zeros(self.env.num_envs, dtype=int) if states.shared else states.groups.cpu().numpy()

        def row(values: torch.Tensor) -> np.ndarray:
            return values.view(-1, states.size)[groups[world]].cpu().numpy()

        started_from = states.started_from.cpu().numpy()
        rates = states.success_rate.view(-1, states.size).cpu().numpy()
        accepted = None if self._accepted is None else self._accepted.cpu().numpy()
        labels = []
        for w, slot in enumerate(started_from):
            if accepted is not None:
                labels.append("accepted" if accepted[w] else "rejected")
            elif slot < 0:
                labels.append("default")
            else:
                rate = rates[groups[w], slot]
                labels.append(f"slot {slot} | {rate:.0%}")
        return ResetStateSnapshot(
            mode=self.mode,
            slot=self.slot,
            playing=self.playing,
            size=states.size,
            num_source_states=states.num_source_states,
            world=world,
            started_from=started_from,
            episode_length=self.env.episode_length_buf.cpu().numpy(),
            success_rate=row(states.success_rate),
            episode_counts=row(states.episode_counts),
            probabilities=row(states.probabilities),
            accepted=accepted,
            message=self._message,
            labels=labels,
        )

    def _refresh(self) -> None:
        self.snapshot = self._snapshot()
        for panel in self._panels:
            panel.refresh(self.snapshot)


def rows(snapshot: ResetStateSnapshot) -> list[tuple[str, ...]]:
    """Return a row per slot of the shown world's reset states: slot, kind, success, episodes, probability. The slot
    the world started from is marked with ``*``."""
    # Generator proposals start from no slot.
    current = -1 if snapshot.accepted is not None else snapshot.started_from[snapshot.world]
    return [
        (
            f"{slot}{' *' if slot == current else ''}",
            "source" if slot < snapshot.num_source_states else "recorded",
            f"{snapshot.success_rate[slot]:.0%}",
            f"{snapshot.episode_counts[slot]}",
            f"{snapshot.probabilities[slot]:.3f}",
        )
        for slot in range(snapshot.size)
    ]


HEADER = ("slot", "kind", "success", "episodes", "p")


def html_table(snapshot: ResetStateSnapshot) -> str:
    """Return the shown world's reset states as an HTML table; Viser's Markdown does not render tables."""
    cell = "<td style='padding:0 6px;text-align:right'>{}</td>"
    head = "".join(f"<th style='padding:0 6px'>{name}</th>" for name in HEADER)
    body = "".join("<tr>" + "".join(cell.format(value) for value in row) + "</tr>" for row in rows(snapshot))
    return f"<b>World {snapshot.world}</b><table style='font-size:12px'><tr>{head}</tr>{body}</table>"


def status(snapshot: ResetStateSnapshot) -> str:
    """Return one line on the mode and what the worlds show."""
    state = "playing" if snapshot.playing else "paused"
    shows = {"draw": "drawn reset states", "slot": f"slot {snapshot.slot}", "generator": "generator proposals"}
    return f"**{shows[snapshot.mode]}**, {state}. {snapshot.message}"


class ViserResetStatePanel:
    """A Viser sidebar folder for a :class:`ResetStateViewer`, and a label above every shown world."""

    MAX_LABELS = 64

    def __init__(self, viewer: ResetStateViewer, server: Any) -> None:
        self.viewer, self.server = viewer, server
        snapshot = viewer.snapshot
        gui = server.gui
        with gui.add_folder("Reset states"):
            self._status = gui.add_markdown(status(snapshot))
            self._slot = gui.add_slider("Slot", min=0, max=max(snapshot.size - 1, 1), step=1, initial_value=0)
            show = gui.add_button("Show slot in every world", hint="Reset every world to this slot, paused.")
            draw = gui.add_button("Draw reset states", hint="Reset every world to a state drawn as in training.")
            roll = gui.add_button("Roll generator", disabled=not viewer.has_generator)
            play = gui.add_button_group("Episodes", ("Play", "Pause", "Step"))
            self._world = gui.add_number("World", 0, min=0, max=viewer.env.num_envs - 1, step=1)
            self._table = gui.add_html(html_table(snapshot))
            self._path = gui.add_text("File", "reset_states.pt")
            save = gui.add_button("Save reset states")

        show.on_click(lambda _: viewer.request("show", int(self._slot.value)))
        # The panel also sets these controls to the viewer's state, which must not request it again.
        self._slot.on_update(lambda _: self._request_if_changed("show", int(self._slot.value), viewer.slot))
        draw.on_click(lambda _: viewer.request("draw"))
        roll.on_click(lambda _: viewer.request("roll"))
        play.on_click(lambda event: viewer.request(event.target.value.lower()))
        self._world.on_update(lambda _: self._request_if_changed("select", int(self._world.value), viewer.world))
        save.on_click(lambda _: viewer.request("save", self._path.value))

        origins = viewer.env.scene.env_origins.cpu().numpy()
        self._labels = [
            server.scene.add_label(f"/reset_states/world_{w}", "", position=tuple(origins[w] + (0.0, 0.0, 0.8)))
            for w in range(min(len(origins), self.MAX_LABELS))
        ]
        self._last: tuple | None = None

    def refresh(self, snapshot: ResetStateSnapshot) -> None:
        key = (status(snapshot), html_table(snapshot), tuple(snapshot.labels), snapshot.slot, snapshot.world)
        if key == self._last:
            return
        self._last = key
        self._status.content, self._table.content = key[0], key[1]
        self._slot.value, self._world.value = snapshot.slot, snapshot.world
        for label, text in zip(self._labels, snapshot.labels):
            label.text = text

    def _request_if_changed(self, action: str, value: int, current: int) -> None:
        if value != current:
            self.viewer.request(action, value)


class NewtonResetStatePanel:
    """A panel in the Newton viewer's ImGui interface for a :class:`ResetStateViewer`. It needs a display."""

    def __init__(self, viewer: ResetStateViewer, visualizer: Any) -> None:
        self.viewer, self._snapshot, self._slot = viewer, viewer.snapshot, 0
        visualizer.register_ui_callback(self._draw, position="side")

    def refresh(self, snapshot: ResetStateSnapshot) -> None:
        self._snapshot = snapshot

    def _draw(self, imgui: Any) -> None:
        snapshot, viewer = self._snapshot, self.viewer
        if not imgui.collapsing_header("Reset states"):
            return
        imgui.text(status(snapshot).replace("**", ""))
        changed, self._slot = imgui.slider_int("Slot", self._slot, 0, max(snapshot.size - 1, 0))
        if imgui.button("Show slot") or changed:
            viewer.request("show", self._slot)
        imgui.same_line()
        if imgui.button("Draw reset states"):
            viewer.request("draw")
        if viewer.has_generator:
            imgui.same_line()
            if imgui.button("Roll generator"):
                viewer.request("roll")
        for action in ("Play", "Pause", "Step"):
            if imgui.button(action):
                viewer.request(action.lower())
            imgui.same_line()
        imgui.new_line()
        changed, world = imgui.input_int("World", snapshot.world)
        if changed:
            viewer.request("select", world)
        imgui.text("  ".join(HEADER))
        for row in rows(snapshot):
            imgui.text("  ".join(row))
