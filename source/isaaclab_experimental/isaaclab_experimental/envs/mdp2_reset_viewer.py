# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Show and drive the reset states of an MDP2 environment from a visualizer.

:class:`ResetViewer` is the controller: it restores bank slots, samples starts as training does, rolls the bank's
candidates task, and plays episodes with a policy. Panels for Viser and the Newton viewer only queue requests, from
their own threads; :meth:`ResetViewer.update` applies them on the main thread and hands the panels plain values.
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

MODES = ("draw", "slot", "candidates")
"""Viewer modes: ``draw`` samples starts as training does, ``slot`` restores one slot in every world, and
``candidates`` shows the proposals of the bank's candidates task."""


@dataclass
class ResetSnapshot:
    """Plain values that panels show, taken on the main thread after each update."""

    mode: str
    slot: int  # The slot that slot mode restores
    playing: bool
    size: int  # Slots per bank
    filled: int  # Slots that the fill wrote; the others record
    shared: bool
    world: int  # The world whose bank the table shows
    slots: np.ndarray  # (num_worlds,) the slot each world started from, -1 for a nominal start
    episode_length: np.ndarray  # (num_worlds,)
    success_rate: np.ndarray  # (size,) of the world's bank
    episode_counts: np.ndarray  # (size,)
    probabilities: np.ndarray  # (size,)
    accepted: np.ndarray | None = None  # (num_worlds,) after a roll of the candidates
    message: str = ""
    labels: list[str] = field(default_factory=list)  # One per world


class ResetViewer:
    """Controller for viewing the bank of a ``reset_from_bank`` event and the episodes that start from it.

    Args:
        env: An MDP2 environment, or a wrapper of one.
        event: Name of the ``reset_from_bank`` event in the task.
        policy: Returns the actions of a step from the observations; None holds zero actions.
    """

    def __init__(self, env: Any, event: str = "reset", policy: Callable[[dict], torch.Tensor] | None = None) -> None:
        self.env: MDP2Env = env.unwrapped
        self.bank = mdp.reset_bank(self.env.state, f"events.{event}")
        fill = self.env.cfg.task.events[event].params.get("fill")
        self._candidates_cfg = fill.candidates if isinstance(fill, mdp.FillFromCandidates) else None
        self._candidates: mdp.Candidates | None = None
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
    def has_candidates(self) -> bool:
        """Whether the bank fills from a candidates task, which the viewer can roll."""
        return self._candidates_cfg is not None

    def request(self, action: str, *args: Any) -> None:
        """Queue an action by method name, such as ``request("show", 3)``, from any thread."""
        self._requests.put((action, args))

    def attach(self) -> int:
        """Add a panel to every Viser and Newton viewer of the environment's simulation, and return how many."""
        for visualizer in self.env.sim.visualizers:
            kind = visualizer.cfg.backend.visualizer_type
            if kind == "viser" and visualizer.server is not None:
                self._panels.append(ViserResetPanel(self, visualizer.server))
            elif kind == "newton_gl":
                self._panels.append(NewtonResetPanel(self, visualizer))
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
        """Restart every world from its slot ``slot``, paused, and keep restarting there while playing."""
        self.slot = int(np.clip(slot, 0, self.bank.size - 1))
        self.bank.replay(torch.full((self.env.num_envs,), self.slot, device=self.env.device))
        self._reset("slot")

    def draw(self) -> None:
        """Restart every world from a start drawn as in training, paused."""
        self.bank.replay(None)
        self._reset("draw")

    def roll(self) -> None:
        """Propose a state in every world with the bank's candidates task, and mark the ones it rejects."""
        if self._candidates_cfg is None:
            self._message = "The bank does not fill from a candidates task."
            return
        if self._candidates is None:
            self._candidates = mdp.Candidates(self._candidates_cfg, self.env.scene_data, self.env.device)
        self.bank.discard_outcomes()  # the episodes are interrupted
        self._accepted = self._candidates.roll()
        self.mode, self.playing = "candidates", False
        accepted = int(self._accepted.sum())
        self._message = f"{accepted} of {self.env.num_envs} proposals accepted."

    def play(self) -> None:
        """Step the episodes with the policy. Candidates are not episodes: playing first samples starts."""
        if self.mode == "candidates":
            self.draw()
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def step(self) -> None:
        """Advance the episodes by one step, paused."""
        if self.mode == "candidates":
            self.draw()
        self.playing = False
        self.env.step(self.policy(self.env.observations))

    def select(self, world: int) -> None:
        """Show the bank of a world in the table."""
        self.world = int(np.clip(world, 0, self.env.num_envs - 1))

    def save(self, path: str) -> None:
        """Save the bank, to fill later runs with ``mdp.FillFromFile``."""
        self.bank.save(Path(path))
        self._message = f"Saved {self.bank.size} slots to {path}."

    def _reset(self, mode: str) -> None:
        # Viewing interrupts episodes, whose outcomes say nothing about their starts.
        self.bank.discard_outcomes()
        self.env.reset()
        self.mode, self.playing, self._accepted, self._message = mode, False, None, ""

    def _snapshot(self) -> ResetSnapshot:
        bank, world = self.bank, self.world

        def row(values: torch.Tensor) -> np.ndarray:
            return (values if bank.shared else values[world]).cpu().numpy()

        slots = bank.slots.cpu().numpy()
        rates = bank.success_rate.cpu().numpy()
        accepted = None if self._accepted is None else self._accepted.cpu().numpy()
        labels = []
        for w, slot in enumerate(slots):
            if accepted is not None:
                labels.append("accepted" if accepted[w] else "rejected")
            elif slot < 0:
                labels.append("nominal")
            else:
                rate = rates[slot] if bank.shared else rates[w, slot]
                labels.append(f"slot {slot} | {rate:.0%}")
        return ResetSnapshot(
            mode=self.mode,
            slot=self.slot,
            playing=self.playing,
            size=bank.size,
            filled=bank.filled,
            shared=bank.shared,
            world=world,
            slots=slots,
            episode_length=self.env.episode_length_buf.cpu().numpy(),
            success_rate=row(bank.success_rate),
            episode_counts=row(bank.episode_counts),
            probabilities=row(bank.probabilities),
            accepted=accepted,
            message=self._message,
            labels=labels,
        )

    def _refresh(self) -> None:
        self.snapshot = self._snapshot()
        for panel in self._panels:
            panel.refresh(self.snapshot)


def rows(snapshot: ResetSnapshot) -> list[tuple[str, ...]]:
    """Return a row per slot of the shown world's bank: slot, kind, success, episodes, probability. The slot the
    world started from is marked with ``*``."""
    # Candidate proposals start from no slot.
    current = -1 if snapshot.accepted is not None else snapshot.slots[0 if snapshot.shared else snapshot.world]
    return [
        (
            f"{slot}{' *' if slot == current else ''}",
            "filled" if slot < snapshot.filled else "recorded",
            f"{snapshot.success_rate[slot]:.0%}",
            f"{snapshot.episode_counts[slot]}",
            f"{snapshot.probabilities[slot]:.3f}",
        )
        for slot in range(snapshot.size)
    ]


HEADER = ("slot", "kind", "success", "episodes", "p")


def html_table(snapshot: ResetSnapshot) -> str:
    """Return the shown world's bank as an HTML table; Viser's Markdown does not render tables."""
    cell = "<td style='padding:0 6px;text-align:right'>{}</td>"
    head = "".join(f"<th style='padding:0 6px'>{name}</th>" for name in HEADER)
    body = "".join("<tr>" + "".join(cell.format(value) for value in row) + "</tr>" for row in rows(snapshot))
    return f"<b>World {snapshot.world}</b><table style='font-size:12px'><tr>{head}</tr>{body}</table>"


def status(snapshot: ResetSnapshot) -> str:
    """Return one line on the mode and what the worlds show."""
    state = "playing" if snapshot.playing else "paused"
    shows = {"draw": "drawn starts", "slot": f"slot {snapshot.slot}", "candidates": "candidate proposals"}
    return f"**{shows[snapshot.mode]}**, {state}. {snapshot.message}"


class ViserResetPanel:
    """A Viser sidebar folder for a :class:`ResetViewer`, and a label above every shown world."""

    MAX_LABELS = 64

    def __init__(self, viewer: ResetViewer, server: Any) -> None:
        self.viewer, self.server = viewer, server
        snapshot = viewer.snapshot
        gui = server.gui
        with gui.add_folder("Reset states"):
            self._status = gui.add_markdown(status(snapshot))
            self._slot = gui.add_slider("Slot", min=0, max=max(snapshot.size - 1, 1), step=1, initial_value=0)
            show = gui.add_button("Show slot in every world", hint="Restart every world from this slot, paused.")
            draw = gui.add_button("Draw starts", hint="Restart every world from a start drawn as in training.")
            roll = gui.add_button("Roll candidates", disabled=not viewer.has_candidates)
            play = gui.add_button_group("Episodes", ("Play", "Pause", "Step"))
            self._world = gui.add_number("World", 0, min=0, max=viewer.env.num_envs - 1, step=1)
            self._table = gui.add_html(html_table(snapshot))
            self._path = gui.add_text("File", "reset_bank.pt")
            save = gui.add_button("Save bank")

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

    def refresh(self, snapshot: ResetSnapshot) -> None:
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


class NewtonResetPanel:
    """A panel in the Newton viewer's ImGui interface for a :class:`ResetViewer`. It needs a display."""

    def __init__(self, viewer: ResetViewer, visualizer: Any) -> None:
        self.viewer, self._snapshot, self._slot = viewer, viewer.snapshot, 0
        visualizer.register_ui_callback(self._draw, position="side")

    def refresh(self, snapshot: ResetSnapshot) -> None:
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
        if imgui.button("Draw starts"):
            viewer.request("draw")
        if viewer.has_candidates:
            imgui.same_line()
            if imgui.button("Roll candidates"):
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
