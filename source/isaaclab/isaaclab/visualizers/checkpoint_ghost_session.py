# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback of several checkpoints of one task side by side as Checkpoint Ghosts.

The session does not depend on a training library. It needs an environment wrapper whose ``step`` returns
observations, rewards, dones and extras and that has ``get_observations``, ``reset`` and ``num_actions``, plus a
``policy_factory`` that turns a checkpoint file into a callable policy with a ``reset`` method. A library's play entry
point supplies those, so supporting another library means writing that wiring. Only RSL-RL provides it for now.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from isaaclab.envs.utils.checkpoint_ghost_sync import CheckpointGhostSync

from .checkpoint_ghost import CheckpointGhostCheckpoint, CheckpointGhostController, CheckpointGhostLayout

logger = logging.getLogger(__name__)

MAX_CHECKPOINT_GHOST_SLOTS = 8
"""Largest number of slots, including the reference."""


class CheckpointGhostPolicy(Protocol):
    """Inference policy driving one slot."""

    def __call__(self, obs: Any) -> torch.Tensor:
        """Return actions for the slot's observations."""

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Reset recurrent state of the environments that terminated, or of all of them when ``dones`` is None."""


def _task_name(task: str) -> str:
    """Return a task id without its namespace prefix."""
    return task.split(":")[-1]


def _natural_sort_key(value: str) -> list[int | str]:
    """Order file names such as ``model_9.pt`` before ``model_10.pt``."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def _describe_load_error(error: Exception) -> str:
    """Return a one-line reason a checkpoint could not be loaded, naming a size mismatch for what it usually means."""
    text = str(error)
    if "size mismatch" in text:
        return "it does not fit this task's policy (its observation or action sizes differ)"
    return (text.splitlines() or [type(error).__name__])[0]


class CheckpointGhostSession(CheckpointGhostController):
    """Steps one environment whose slices are driven by different checkpoints of the same task.

    Slot 0 is the reference and is loaded at start, so it owns every environment until a ghost is added. Ghost slots
    start empty and are filled from the visualizer HUD. The environments are split evenly among the loaded slots, so
    adding or removing a checkpoint resizes every slot and restarts the environments. Environments left over by an
    uneven split are simulated with zero actions.
    """

    _actions: torch.Tensor | None = None
    """Actions of every environment, reused by every step."""

    def __init__(
        self,
        env: Any,
        *,
        num_slots: int,
        task: str,
        checkpoint_root: str,
        checkpoint_pattern: str,
        policy_factory: Callable[[str], CheckpointGhostPolicy],
        reference: CheckpointGhostCheckpoint,
        run_task: Callable[[str], str | None] | None = None,
    ):
        """Create the session and load the reference checkpoint.

        Args:
            env: RL wrapper around a manager-based environment; its environments are split among the loaded slots.
            num_slots: Largest number of slots, including the reference.
            task: Task of the running environment, used to flag runs of other tasks in the checkpoint window.
            checkpoint_root: Directory the checkpoint window starts in.
            checkpoint_pattern: Regular expression matching checkpoint file names.
            policy_factory: Builds the inference policy of a checkpoint file.
            reference: Checkpoint driving slot 0.
            run_task: Returns the task of the training run saved in a directory, or None when it is not a run. Without
                it, the checkpoint window does not flag runs of other tasks.
        """
        if not 2 <= num_slots <= MAX_CHECKPOINT_GHOST_SLOTS:
            raise ValueError(f"Checkpoint Ghosts support 2 to {MAX_CHECKPOINT_GHOST_SLOTS} slots, got {num_slots}.")
        self.layout = CheckpointGhostLayout(env.unwrapped.num_envs, num_slots)
        self._env = env
        self._root = checkpoint_root
        self._task = _task_name(task)
        self._run_task = run_task
        self._pattern = checkpoint_pattern
        self._policy_factory = policy_factory
        self._policies: list[CheckpointGhostPolicy | None] = [None] * num_slots
        self._restart_pending = False
        self._sync: CheckpointGhostSync | None = None
        self._load(0, reference)
        try:
            self._obs = env.get_observations()
        except BaseException:
            self.close()
            raise
        # start the reference and the empty slots together from the same spawn state
        self._restart_pending = True

    @property
    def checkpoint_root(self) -> str:  # noqa: D102
        return self._root

    def browse(self, directory: str) -> tuple[list[str], list[CheckpointGhostCheckpoint]]:  # noqa: D102
        path = Path(directory)
        folders, files = [], []
        for entry in path.iterdir():
            if entry.is_dir():
                if not entry.name.startswith("."):
                    folders.append(entry)
            elif entry.is_file() and re.fullmatch(self._pattern, entry.name):
                files.append(entry)
        folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # newest run first
        files.sort(key=lambda p: _natural_sort_key(p.name))
        return [p.name for p in folders], [
            CheckpointGhostCheckpoint(str(p.resolve()), path.name, p.name, p.stat().st_mtime) for p in files
        ]

    def other_task(self, directory: str) -> str | None:  # noqa: D102
        task = self._run_task(directory) if self._run_task is not None else None
        return task if isinstance(task, str) and _task_name(task) != self._task else None

    def add_ghost(self, checkpoint: CheckpointGhostCheckpoint) -> str | None:  # noqa: D102
        empty = next((s for s in self.layout.slots[1:] if not s.active), None)
        if empty is None:
            return "All ghost slots are in use."
        try:
            self._load(empty.index, checkpoint)
        except Exception as error:  # a corrupt or mismatched checkpoint must not end the session
            return f"Could not load {checkpoint.name}: {_describe_load_error(error)}"  # nothing changed yet
        empty.visible = True
        try:
            self._resplit()
        except Exception as error:  # the new split failed: go back to the previous one
            self._policies[empty.index] = None
            empty.checkpoint = None
            empty.label = ""
            self._resplit()
            return f"Could not add {checkpoint.name}: {error}"
        return None

    def replace_ghost(self, slot_index: int, checkpoint: CheckpointGhostCheckpoint) -> str | None:  # noqa: D102
        if not 1 <= slot_index < self.layout.num_slots:
            return "Only a ghost slot can be changed."
        slot = self.layout.slots[slot_index]
        if not slot.active:
            return "Only a loaded ghost can be changed; add a checkpoint to an empty slot instead."
        previous = (self._policies[slot_index], slot.checkpoint, slot.label)
        try:
            self._load(slot_index, checkpoint)
        except Exception as error:  # a corrupt or mismatched checkpoint must keep the ghost that was there
            self._policies[slot_index], slot.checkpoint, slot.label = previous
            return f"Could not load {checkpoint.name}: {_describe_load_error(error)}"
        # the changed ghost starts together with the reference from the same spawn state
        self._restart_pending = True
        return None

    def remove_ghost(self, slot_index: int) -> None:  # noqa: D102
        if not 1 <= slot_index < self.layout.num_slots or not self.layout.slots[slot_index].active:
            return  # the reference cannot be removed, and an empty slot has nothing to free
        self._policies[slot_index] = None
        self.layout.slots[slot_index].checkpoint = None
        self.layout.slots[slot_index].label = ""
        self._resplit()

    def env_origin_offsets(self) -> np.ndarray:  # noqa: D102
        if self._sync is None:
            return np.zeros((self._env.unwrapped.num_envs, 3), dtype=np.float32)
        return self._sync.env_origin_offsets()

    def apply_pending_restart(self) -> None:  # noqa: D102
        if not self._restart_pending:
            return
        self._restart_pending = False
        self._obs, _ = self._env.reset()
        for policy in self._policies:
            if policy is not None:
                policy.reset()  # a full reset, since adding or removing a ghost changes the policies' batch size

    def step(self) -> None:
        """Step the environment once, with each loaded slot acting on its own observations."""
        self.apply_pending_restart()
        size = self.layout.slice_size
        if self._actions is None:
            self._actions = torch.zeros(self._env.num_envs, self._env.num_actions, device=self._env.unwrapped.device)
        actions = self._actions
        actions.zero_()  # environments of no loaded slot act with zeros
        for slot in self.layout.active_slots:
            start = self.layout.slot_start(slot.index)
            chunk = slice(start, start + size)
            actions[chunk] = self._policies[slot.index](self._obs[chunk])
        self._obs, _, dones, _ = self._env.step(actions)
        reference_dones = dones[:size]
        for slot in self.layout.active_slots:
            # A ghost resets with its reference, so a ghost that fell alone keeps its memory until then.
            self._policies[slot.index].reset(reference_dones)

    def close(self) -> None:
        """Detach the synchronization hooks from the environment."""
        if self._sync is not None:
            self._sync.close()
            self._sync = None

    def _resplit(self) -> None:
        """Divide the environments among the loaded slots again and restart them from the same spawn state."""
        self.close()  # restores the terrain the previous split overwrote
        loaded = len(self.layout.active_slots)
        if loaded >= 2:
            self._sync = CheckpointGhostSync(self._env.unwrapped, self.layout.slice_size, loaded)
        self._restart_pending = True

    def _load(self, slot_index: int, checkpoint: CheckpointGhostCheckpoint) -> None:
        """Load a checkpoint's policy into a slot."""
        self._policies[slot_index] = self._policy_factory(checkpoint.path)
        slot = self.layout.slots[slot_index]
        slot.checkpoint = checkpoint
        slot.label = f"{checkpoint.name} / {time.strftime('%Y-%m-%d %H:%M', time.localtime(checkpoint.modified))}"
