# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract between a policy playback loop and a visualizer that shows Policy Ghosts.

Policy Ghosts overlay several checkpoints of the same task in one viewer. The playback loop owns the
simulation and the policies; a visualizer only reads the controller to draw its HUD and tint the ghosts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

POLICY_GHOST_COLORS: tuple[tuple[float, float, float], ...] = (
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
class PolicyGhostCheckpoint:
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
class PolicyGhostSlot:
    """One simulated slice of environments driven by a single checkpoint."""

    index: int
    """Slot index. Slot 0 is the reference; the others are ghosts."""

    checkpoint: PolicyGhostCheckpoint | None = None
    """Checkpoint driving this slot, or None when the slot is empty."""

    opacity: float = 0.65
    """Opacity of the slot's robots in ``[0, 1]``. Ignored for the reference slot."""

    tint: float = 1.0
    """How strongly :attr:`color` replaces the robots' native colors, in ``[0, 1]``.

    At ``1.0`` the color replaces all native colors and textures. Ignored for the reference slot.
    """

    visible: bool = True
    """Whether the slot's robots are drawn. A hidden slot keeps its checkpoint and keeps running."""

    color: tuple[float, float, float] | None = None
    """Tint color of the slot's robots, or None to keep the native colors."""

    label: str = ""
    """Short label shown in the HUD, such as the run name and checkpoint file."""

    @property
    def active(self) -> bool:
        """Whether a checkpoint is loaded into the slot."""
        return self.checkpoint is not None

    @property
    def shown(self) -> bool:
        """Whether the slot has a checkpoint and is set to be drawn."""
        return self.checkpoint is not None and self.visible


@dataclass(frozen=True)
class PolicyGhostMetrics:
    """Statistics of the episodes a slot finished most recently."""

    episodes: int
    """Number of episodes the statistics average over."""

    mean_return: float
    """Mean undiscounted return of those episodes."""

    mean_length: float
    """Mean episode length [s]."""


@dataclass
class PolicyGhostLayout:
    """How environments map to slots: slot ``s`` owns envs ``[s * slice_size, (s + 1) * slice_size)``."""

    slice_size: int
    """Number of environments per slot."""

    num_slots: int
    """Number of slots, including the reference."""

    slots: list[PolicyGhostSlot] = field(default_factory=list)
    """Slot state, indexed by slot."""

    def __post_init__(self) -> None:
        if self.slice_size < 1 or self.num_slots < 2:
            raise ValueError("Policy Ghosts need a positive slice size and at least two slots.")
        if not self.slots:
            self.slots = [PolicyGhostSlot(index=s) for s in range(self.num_slots)]
            for slot in self.slots[1:]:
                slot.color = POLICY_GHOST_COLORS[(slot.index - 1) % len(POLICY_GHOST_COLORS)]

    @property
    def num_envs(self) -> int:
        """Total number of simulated environments."""
        return self.slice_size * self.num_slots

    def visible_env_ids(self, per_slot: int) -> list[int]:
        """Return the environments to show when each shown slot shows its first ``per_slot`` environments."""
        per_slot = max(0, min(int(per_slot), self.slice_size))
        return [slot.index * self.slice_size + i for slot in self.slots if slot.shown for i in range(per_slot)]

    def slot_of_env(self, env_ids: np.ndarray) -> np.ndarray:
        """Return the slot index of each environment id."""
        return np.asarray(env_ids) // self.slice_size


class PolicyGhostController(ABC):
    """Playback-loop side of Policy Ghosts, read and driven by the visualizer HUD."""

    layout: PolicyGhostLayout
    """Environment-to-slot layout and slot state."""

    @property
    @abstractmethod
    def checkpoint_root(self) -> str:
        """Directory searched for compatible checkpoints."""

    @abstractmethod
    def discover(self) -> list[PolicyGhostCheckpoint]:
        """Return the checkpoints below :attr:`checkpoint_root` that are compatible with the running task."""

    @abstractmethod
    def add_ghost(self, checkpoint: PolicyGhostCheckpoint) -> str | None:
        """Load a checkpoint into the first empty ghost slot.

        Returns:
            None on success, otherwise a message explaining why the checkpoint could not be loaded.
        """

    @abstractmethod
    def remove_ghost(self, slot_index: int) -> None:
        """Empty a ghost slot; the reference slot cannot be removed."""

    def slot_metrics(self, slot_index: int) -> PolicyGhostMetrics | None:
        """Return the episode statistics of a slot, or None when the controller does not track them yet."""
        return None

    @abstractmethod
    def env_origin_offsets(self) -> np.ndarray:
        """Return per-environment offsets [m], shape ``[num_envs, 3]``, that move each ghost onto the reference."""
