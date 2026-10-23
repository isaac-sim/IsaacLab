# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback of several checkpoints of one task side by side as Policy Ghosts."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch

from isaaclab.envs.utils.policy_ghost_sync import PolicyGhostSync
from isaaclab.visualizers.policy_ghost import PolicyGhostCheckpoint, PolicyGhostController, PolicyGhostLayout

from .common import _compatible_runs, _natural_sort_key, normalize_task_name

if TYPE_CHECKING:
    pass

MAX_POLICY_GHOST_SLOTS = 8
"""Largest number of slots, including the reference."""


class PolicyGhostPolicy(Protocol):
    """Inference policy driving one slot."""

    def __call__(self, obs: Any) -> torch.Tensor:
        """Return actions for the slot's observations."""

    def reset(self, dones: torch.Tensor) -> None:
        """Reset recurrent state of the slot's environments that terminated."""


class PolicyGhostRestart:
    """A request from the viewer HUD to restart playback with Policy Ghosts switched on or off.

    Policy Ghosts simulate several times the environments, which the simulation fixes when it starts, so the
    process is relaunched with the changed command line instead of growing the running simulation.
    """

    def __init__(self) -> None:
        self.requested = False
        """Whether the HUD asked for a restart."""
        self.enable = False
        """Whether the restarted process should run with Policy Ghosts."""

    def request(self, enable: bool) -> None:
        """Ask for playback to restart with Policy Ghosts enabled or disabled."""
        self.requested = True
        self.enable = enable


_POLICY_GHOST_VALUE_FLAGS = ("--policy_ghost_slots", "--policy_ghost_root")
"""Policy Ghost options that take a value and are dropped when the feature is disabled."""


def relaunch_argv(argv: list[str], enable: bool) -> list[str]:
    """Return a command line that runs the same playback with Policy Ghosts switched on or off.

    Args:
        argv: Original command line, including the executable at index 0.
        enable: Whether the new command line should enable Policy Ghosts.

    Returns:
        The command line with ``--policy_ghosts`` added or, with all Policy Ghost options, removed.
    """
    if enable:
        return argv if "--policy_ghosts" in argv else [*argv, "--policy_ghosts"]
    result, skip = [], False
    for arg in argv:
        if skip:
            skip = False
        elif arg == "--policy_ghosts" or any(arg.startswith(f"{flag}=") for flag in _POLICY_GHOST_VALUE_FLAGS):
            continue
        elif arg in _POLICY_GHOST_VALUE_FLAGS:
            skip = True
        else:
            result.append(arg)
    return result


def _launch_argv() -> list[str]:
    """Return the command line that started this process, including interpreter and entry point.

    A launcher script may replace ``sys.argv`` with the arguments it forwards to the CLI. Running that script again
    would ignore any flag added to its command line, so the CLI entry point is run with the forwarded arguments.
    """
    original, current = list(sys.orig_argv), list(sys.argv)
    if original[len(original) - len(current) + 1 :] == current[1:]:
        return original
    return [sys.executable, "-c", "from isaaclab.cli import cli; cli()", *current[1:]]


def relaunch(enable: bool) -> None:
    """Replace the running process with the same command line and Policy Ghosts switched on or off."""
    argv = relaunch_argv(_launch_argv(), enable)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, *argv[1:]])


def discover_checkpoints(
    log_root: str, *, library: str, task: str, checkpoint_pattern: str, metadata: dict[str, str] | None = None
) -> list[PolicyGhostCheckpoint]:
    """List the checkpoints of every manifested run in a log root that matches a task.

    Args:
        log_root: Directory containing training run directories.
        library: Reinforcement learning library expected in the run manifests.
        task: Task expected in the run manifests.
        checkpoint_pattern: Regular expression matching checkpoint file names.
        metadata: Additional manifest metadata required for compatibility.

    Returns:
        Checkpoints of the newest run first, each run's checkpoints in natural order.
    """
    runs = _compatible_runs(Path(log_root), library=library, task=task, metadata=metadata or {})
    checkpoints = []
    for _, run_dir in sorted(runs, reverse=True):
        files = [p for p in run_dir.rglob("*") if p.is_file() and re.fullmatch(checkpoint_pattern, p.name)]
        files.sort(key=lambda p: _natural_sort_key(str(p.relative_to(run_dir))))
        checkpoints += [PolicyGhostCheckpoint(str(p.resolve()), run_dir.name, p.name, p.stat().st_mtime) for p in files]
    return checkpoints


class PolicyGhostSession(PolicyGhostController):
    """Steps one environment whose slices are driven by different checkpoints of the same task.

    Slot 0 is the reference and is loaded at start. Ghost slots start empty and are filled from the visualizer
    HUD. Environments of empty slots are simulated with zero actions.
    """

    def __init__(
        self,
        env: Any,
        *,
        slice_size: int,
        num_slots: int,
        checkpoint_root: str,
        task: str,
        library: str,
        checkpoint_pattern: str,
        policy_factory: Callable[[str], PolicyGhostPolicy],
        reference: PolicyGhostCheckpoint,
        metadata: dict[str, str] | None = None,
    ):
        """Create the session and load the reference checkpoint.

        Args:
            env: RL wrapper around a manager-based environment simulated with ``slice_size * num_slots`` worlds.
            slice_size: Number of environments per slot.
            num_slots: Number of slots, including the reference.
            checkpoint_root: Directory searched for compatible checkpoints.
            task: Task of the running environment; only checkpoints of this task are offered.
            library: Reinforcement learning library that wrote the checkpoints.
            checkpoint_pattern: Regular expression matching checkpoint file names.
            policy_factory: Builds the inference policy of a checkpoint file.
            reference: Checkpoint driving slot 0.
            metadata: Additional manifest metadata required for compatibility.
        """
        if not 2 <= num_slots <= MAX_POLICY_GHOST_SLOTS:
            raise ValueError(f"Policy Ghosts support 2 to {MAX_POLICY_GHOST_SLOTS} slots, got {num_slots}.")
        self.layout = PolicyGhostLayout(slice_size, num_slots)
        self._env = env
        self._root = checkpoint_root
        self._task = normalize_task_name(task)
        self._library = library
        self._pattern = checkpoint_pattern
        self._metadata = metadata or {}
        self._policy_factory = policy_factory
        self._sync = PolicyGhostSync(env.unwrapped, slice_size, num_slots)
        self._policies: list[PolicyGhostPolicy | None] = [None] * num_slots
        self._restart_pending = False
        self._load(0, reference)
        self._obs = env.get_observations()

    @property
    def checkpoint_root(self) -> str:  # noqa: D102
        return self._root

    def discover(self) -> list[PolicyGhostCheckpoint]:  # noqa: D102
        return discover_checkpoints(
            self._root,
            library=self._library,
            task=self._task,
            checkpoint_pattern=self._pattern,
            metadata=self._metadata,
        )

    def add_ghost(self, checkpoint: PolicyGhostCheckpoint) -> str | None:  # noqa: D102
        empty = next((s for s in self.layout.slots[1:] if not s.active), None)
        if empty is None:
            return "All ghost slots are in use."
        try:
            self._load(empty.index, checkpoint)
            empty.visible = True
            # a new ghost starts together with the reference from the same spawn state
            self._restart_pending = True
        except Exception as error:  # a corrupt or mismatched checkpoint must not end the session
            self._policies[empty.index] = None
            empty.checkpoint = None
            return f"Could not load {checkpoint.name}: {error}"
        return None

    def remove_ghost(self, slot_index: int) -> None:  # noqa: D102
        if slot_index < 1:
            return
        self._policies[slot_index] = None
        self.layout.slots[slot_index].checkpoint = None
        self.layout.slots[slot_index].label = ""

    def env_origin_offsets(self) -> np.ndarray:  # noqa: D102
        return self._sync.env_origin_offsets()

    def step(self) -> None:
        """Step the environment once, with each active slot acting on its own observations."""
        size = self.layout.slice_size
        if self._restart_pending:
            self._restart_pending = False
            self._obs, _ = self._env.reset()
            for policy in self._policies:
                if policy is not None:
                    policy.reset(torch.ones(size, dtype=torch.bool, device=self._env.unwrapped.device))
        actions = torch.zeros(self._env.num_envs, self._env.num_actions, device=self._env.unwrapped.device)
        for slot in self.layout.slots:
            policy = self._policies[slot.index]
            if policy is not None:
                chunk = slice(slot.index * size, (slot.index + 1) * size)
                actions[chunk] = policy(self._obs[chunk])
        self._obs, _, dones, _ = self._env.step(actions)
        for slot in self.layout.slots:
            policy = self._policies[slot.index]
            if policy is not None:
                policy.reset(dones[slot.index * size : (slot.index + 1) * size])

    def close(self) -> None:
        """Detach the synchronization hooks from the environment."""
        self._sync.close()

    def _load(self, slot_index: int, checkpoint: PolicyGhostCheckpoint) -> None:
        """Load a checkpoint's policy into a slot."""
        self._policies[slot_index] = self._policy_factory(checkpoint.path)
        slot = self.layout.slots[slot_index]
        slot.checkpoint = checkpoint
        slot.label = f"{checkpoint.run} / {checkpoint.name}"
