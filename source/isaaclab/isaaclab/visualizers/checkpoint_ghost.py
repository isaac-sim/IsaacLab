# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract between a policy playback loop and a visualizer that shows Checkpoint Ghosts.

Checkpoint Ghosts overlay several checkpoints of the same task in one viewer. The playback loop owns the
simulation and the policies; a visualizer only reads the controller to draw its HUD and tint the ghosts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

CHECKPOINT_GHOST_COLORS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.1, 0.1),
    (0.1, 1.0, 0.25),
    (1.0, 0.85, 0.05),
    (0.1, 0.9, 1.0),
    (1.0, 0.2, 0.9),
    (1.0, 0.5, 0.05),
    (0.55, 0.3, 1.0),
)
"""Tint colors of the ghost slots in order; the reference slot keeps its native colors."""


@dataclass(frozen=True)
class CheckpointGhostCheckpoint:
    """A checkpoint that can be loaded into a ghost slot."""

    path: str
    """Absolute path to the checkpoint file."""

    run: str
    """Name of the training run directory that holds the checkpoint."""

    name: str
    """Checkpoint file name."""

    modified: float
    """Modification time of the checkpoint file, as seconds since the epoch."""


@dataclass
class CheckpointGhostSlot:
    """One simulated slice of environments driven by a single checkpoint."""

    index: int
    """Slot index. Slot 0 is the reference; the others are ghosts."""

    checkpoint: CheckpointGhostCheckpoint | None = None
    """Checkpoint driving this slot, or None when the slot is empty."""

    opacity: float = 0.4
    """Opacity of the slot's robots in ``[0, 1]``. Ignored for the reference slot."""

    tint: float = 0.7
    """How strongly :attr:`color` replaces the robots' native colors, in ``[0, 1]``.

    At ``1.0`` the color replaces all native colors and textures. Ignored for the reference slot.
    """

    visible: bool = True
    """Whether the slot's robots are drawn. A hidden slot keeps its checkpoint and keeps running."""

    color: tuple[float, float, float] | None = None
    """Tint color of the slot's robots, or None to keep the native colors."""

    label: str = ""
    """Short label shown in the HUD: the checkpoint file name and its modification time."""

    @property
    def active(self) -> bool:
        """Whether a checkpoint is loaded into the slot."""
        return self.checkpoint is not None

    @property
    def shown(self) -> bool:
        """Whether the slot has a checkpoint and is set to be drawn."""
        return self.checkpoint is not None and self.visible


@dataclass
class CheckpointGhostLayout:
    """How environments map to slots.

    The simulated environments are split evenly among the slots that hold a checkpoint, in slot order: the first
    loaded slot (the reference) owns the first ``slice_size`` environments, the next loaded slot the following
    ``slice_size``, and so on. Loading or unloading a checkpoint therefore changes :attr:`slice_size`. Environments
    beyond the last loaded slot's share, when the split leaves a remainder, belong to no slot.
    """

    num_envs: int
    """Number of simulated environments."""

    num_slots: int
    """Largest number of slots, including the reference."""

    slots: list[CheckpointGhostSlot] = field(default_factory=list)
    """Slot state, indexed by slot."""

    def __post_init__(self) -> None:
        if self.num_envs < 1 or self.num_slots < 2:
            raise ValueError("Checkpoint Ghosts need at least one environment and two slots.")
        if not self.slots:
            self.slots = [CheckpointGhostSlot(index=s) for s in range(self.num_slots)]
            for slot in self.slots[1:]:
                slot.color = CHECKPOINT_GHOST_COLORS[(slot.index - 1) % len(CHECKPOINT_GHOST_COLORS)]

    @property
    def active_slots(self) -> list[CheckpointGhostSlot]:
        """Slots that hold a checkpoint, in slot order."""
        return [slot for slot in self.slots if slot.active]

    @property
    def slice_size(self) -> int:
        """Environments per loaded slot: the simulated environments split evenly among them."""
        return max(1, self.num_envs // max(1, len(self.active_slots)))

    def slot_start(self, slot_index: int) -> int:
        """Return the first environment of a loaded slot."""
        ranks = [slot.index for slot in self.active_slots]
        return ranks.index(slot_index) * self.slice_size

    def visible_env_ids(self, per_slot: int) -> list[int]:
        """Return the environments to show when each shown slot shows its first ``per_slot`` environments."""
        size = self.slice_size
        per_slot = max(0, min(int(per_slot), size))
        return [self.slot_start(slot.index) + i for slot in self.slots if slot.shown for i in range(per_slot)]

    def slot_of_env(self, env_ids: np.ndarray) -> np.ndarray:
        """Return the slot index of each environment id, or -1 for ids that belong to no loaded slot."""
        env_ids = np.asarray(env_ids)
        active = [slot.index for slot in self.active_slots]
        result = np.full(env_ids.shape, -1, dtype=np.int64)
        if active:
            rank = env_ids // self.slice_size
            valid = (env_ids >= 0) & (rank < len(active))
            result[valid] = np.asarray(active)[rank[valid]]
        return result


class CheckpointGhostController(ABC):
    """Playback-loop side of Checkpoint Ghosts, read and driven by the visualizer HUD."""

    layout: CheckpointGhostLayout
    """Environment-to-slot layout and slot state."""

    @property
    @abstractmethod
    def checkpoint_root(self) -> str:
        """Directory searched for compatible checkpoints."""

    @abstractmethod
    def browse(self, directory: str) -> tuple[list[str], list[CheckpointGhostCheckpoint]]:
        """List a directory for the checkpoint window.

        Args:
            directory: Directory to list; it may be anywhere on the filesystem.

        Returns:
            The names of its subdirectories and the checkpoint files in it that can be loaded.

        Raises:
            OSError: If the directory cannot be read.
        """

    @abstractmethod
    def other_task(self, directory: str) -> str | None:
        """Return the task of the run saved in ``directory`` when it is not the running task, else None.

        The checkpoint window uses this to flag runs whose checkpoints cannot fit the running policy.
        """

    @abstractmethod
    def add_ghost(self, checkpoint: CheckpointGhostCheckpoint) -> str | None:
        """Load a checkpoint into the first empty ghost slot.

        Returns:
            None on success, otherwise a message explaining why the checkpoint could not be loaded.
        """

    @abstractmethod
    def remove_ghost(self, slot_index: int) -> None:
        """Empty a ghost slot; the reference slot cannot be removed."""

    @abstractmethod
    def replace_ghost(self, slot_index: int, checkpoint: CheckpointGhostCheckpoint) -> str | None:
        """Load a checkpoint into a given ghost slot, in place of the one it holds.

        Returns:
            None on success, otherwise a message explaining why the checkpoint could not be loaded.
        """

    @abstractmethod
    def apply_pending_restart(self) -> None:
        """Reset the environments now when a new ghost asked for a restart that the next step would apply.

        The viewer calls this while the simulation is paused, because no step runs then and a ghost that was just
        added would otherwise stay in whatever pose its idle slot ended in.
        """

    @abstractmethod
    def reference_env_ids(self) -> np.ndarray:
        """Return the reference environment of each environment, shape ``[num_envs]``; references map to themselves."""

    @abstractmethod
    def env_origin_offsets(self) -> np.ndarray:
        """Return per-environment offsets [m], shape ``[num_envs, 3]``, that move each ghost onto the reference."""
