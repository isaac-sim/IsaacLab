# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback of several checkpoints of one task side by side as Policy Ghosts."""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch

from isaaclab.envs.utils.policy_ghost_sync import PolicyGhostSync
from isaaclab.visualizers.policy_ghost import (
    PolicyGhostCheckpoint,
    PolicyGhostController,
    PolicyGhostLayout,
    PolicyGhostMetrics,
)

from .common import _compatible_runs, _natural_sort_key, normalize_task_name
from .policy_ghost_metrics import PolicyGhostMetricsTracker

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

MAX_POLICY_GHOST_SLOTS = 8
"""Largest number of slots, including the reference."""

DEFAULT_GHOST_ENVS = 4
"""Environments per slot when Policy Ghosts are switched on inside a running playback."""

DEFAULT_GHOST_SLOTS = 3
"""Slots, including the reference, when Policy Ghosts are switched on inside a running playback."""


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
        self.envs: int | None = None
        """Environments per slot the restarted process should use, or None to keep ``--num_envs``."""
        self.slots: int | None = None
        """Slots the restarted process should use, or None to keep ``--policy_ghost_slots``."""

    def request(self, enable: bool, envs: int | None = None, slots: int | None = None) -> None:
        """Ask for playback to restart with Policy Ghosts enabled or disabled.

        Args:
            enable: Whether the restarted process should run with Policy Ghosts.
            envs: Environments per slot for the restarted process, when enabling.
            slots: Slots, including the reference, for the restarted process, when enabling.
        """
        self.requested = True
        self.enable = enable
        self.envs, self.slots = envs, slots


_POLICY_GHOST_VALUE_FLAGS = ("--policy_ghost_slots", "--policy_ghost_envs", "--policy_ghost_root")
"""Policy Ghost options that take a value and are dropped when the feature is disabled."""


def _with_flag(argv: list[str], flag: str, value: int) -> list[str]:
    """Return ``argv`` with ``flag`` set to ``value``, replacing either ``flag value`` or ``flag=value``."""
    result, skip, found = [], False, False
    for arg in argv:
        if skip:
            skip = False
        elif arg == flag:
            result += [flag, str(value)]
            skip, found = True, True
        elif arg.startswith(f"{flag}="):
            result.append(f"{flag}={value}")
            found = True
        else:
            result.append(arg)
    return result if found else [*result, flag, str(value)]


def relaunch_argv(argv: list[str], enable: bool, envs: int | None = None, slots: int | None = None) -> list[str]:
    """Return a command line that runs the same playback with Policy Ghosts switched on or off.

    Args:
        argv: Original command line, including the executable at index 0.
        enable: Whether the new command line should enable Policy Ghosts.
        envs: Environments per slot to run with when enabling, which sets ``--num_envs``.
        slots: Slots to run with when enabling, which sets ``--policy_ghost_slots``.

    Returns:
        The command line with ``--policy_ghosts`` added or, with all Policy Ghost options, removed.
    """
    if enable:
        result = argv if "--policy_ghosts" in argv else [*argv, "--policy_ghosts"]
        if envs is not None:
            result = _with_flag(result, "--num_envs", envs)
        if slots is not None:
            result = _with_flag(result, "--policy_ghost_slots", slots)
        return result
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


def relaunch(enable: bool, envs: int | None = None, slots: int | None = None) -> None:
    """Replace the running process with the same command line and Policy Ghosts switched on or off."""
    argv = relaunch_argv(_launch_argv(), enable, envs, slots)
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


class PolicyGhostMode:
    """Switches Policy Ghosts on and off inside a playback that started without them.

    Switching on uses the first ``slice_size * num_slots`` environments of the running simulation as the slots, so no
    restart is needed when the simulation has that many. The remaining environments are hidden and get zero actions.
    Switching off resets every environment and returns to the plain playback.
    """

    def __init__(
        self,
        env: Any,
        *,
        slice_size: int,
        num_slots: int,
        create_session: Callable[[int, int], PolicyGhostSession],
        step: Callable[[], None],
        resume: Callable[[], None],
        restart: PolicyGhostRestart,
        visualizers: list[Any],
    ):
        """Create the mode.

        Args:
            env: RL wrapper around the running manager-based environment.
            slice_size: Environments per slot.
            num_slots: Slots, including the reference.
            create_session: Builds the Policy Ghost session for a slice size and slot count.
            step: One step of the plain playback.
            resume: Prepares the plain playback to continue after the environments were reset.
            restart: Request used when the simulation has too few environments for the slots.
            visualizers: Visualizers that can show the Policy Ghosts HUD section.
        """
        self._env = env
        self.slice_size = slice_size
        self.num_slots = num_slots
        self._create_session = create_session
        self._step = step
        self._resume = resume
        self.restart = restart
        self._visualizers = visualizers
        self.session: PolicyGhostSession | None = None
        self.message = ""
        """The reason the latest switch failed, or an empty string."""
        self._pending: bool | None = None

    @property
    def pool_size(self) -> int:
        """Environments the running simulation has, which bound ``slice_size * num_slots``."""
        return self._env.unwrapped.num_envs

    def set_layout(self, slice_size: int, num_slots: int) -> None:
        """Choose the environments per slot and the slots to use the next time Policy Ghosts are switched on.

        Args:
            slice_size: Environments per slot, at least 1.
            num_slots: Slots including the reference, from 2 to :data:`MAX_POLICY_GHOST_SLOTS`.
        """
        self.slice_size = max(1, int(slice_size))
        self.num_slots = max(2, min(int(num_slots), MAX_POLICY_GHOST_SLOTS))

    def request_restart(self) -> None:
        """Ask for the playback to restart with Policy Ghosts and the chosen environments per slot and slots."""
        self.restart.request(True, self.slice_size, self.num_slots)

    @property
    def in_place(self) -> bool:
        """Whether the running simulation has enough environments to switch Policy Ghosts on without a restart."""
        return self._env.unwrapped.num_envs >= self.slice_size * self.num_slots

    @property
    def active(self) -> bool:
        """Whether Policy Ghosts are on."""
        return self.session is not None

    def request_enable(self) -> None:
        """Switch Policy Ghosts on before the next step."""
        self._pending = True

    def request_disable(self) -> None:
        """Switch Policy Ghosts off before the next step."""
        self._pending = False

    def step(self) -> None:
        """Apply a pending switch, then step the active mode."""
        pending, self._pending = self._pending, None
        if pending is not None and pending != self.active:
            self._switch(pending)
        (self.session.step if self.session is not None else self._step)()

    def close(self) -> None:
        """Release the Policy Ghost session if it is on."""
        if self.session is not None:
            self.session.close()
            self.session = None

    def _switch(self, enable: bool) -> None:
        self.message = ""
        session = None
        try:
            if enable:
                session = self._create_session(self.slice_size, self.num_slots)
                for visualizer in self._visualizers:
                    visualizer.attach_policy_ghosts(session, self.restart.request, self.request_disable)
                self.session = session
            else:
                for visualizer in self._visualizers:
                    visualizer.detach_policy_ghosts()
                self.close()
                self._env.reset()
                self._resume()
        except Exception as error:  # a failed switch must not end the playback
            logger.exception("Could not switch Policy Ghosts %s.", "on" if enable else "off")
            if enable:
                for visualizer in self._visualizers:
                    visualizer.detach_policy_ghosts()
                if session is not None:
                    session.close()
            self.message = f"Could not switch Policy Ghosts {'on' if enable else 'off'}: {error}"


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
        self._metrics = PolicyGhostMetricsTracker(
            slice_size, num_slots, env.num_envs, env.unwrapped.step_dt, env.unwrapped.device
        )
        try:
            self._load(0, reference)
            self._obs = env.get_observations()
        except BaseException:
            self._sync.close()  # a failed start must not leave the environment's reset and commands patched
            raise
        # start the reference and the empty slots together from the same spawn state
        self._restart_pending = True

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
        self._metrics.set_active(slot_index, False)

    def slot_metrics(self, slot_index: int) -> PolicyGhostMetrics | None:  # noqa: D102
        return self._metrics.metrics(slot_index)

    def env_origin_offsets(self) -> np.ndarray:  # noqa: D102
        return self._sync.env_origin_offsets()

    def step(self) -> None:
        """Step the environment once, with each active slot acting on its own observations."""
        size = self.layout.slice_size
        if self._restart_pending:
            self._restart_pending = False
            self._obs, _ = self._env.reset()
            self._metrics.reset()
            for policy in self._policies:
                if policy is not None:
                    policy.reset(torch.ones(size, dtype=torch.bool, device=self._env.unwrapped.device))
        actions = torch.zeros(self._env.num_envs, self._env.num_actions, device=self._env.unwrapped.device)
        for slot in self.layout.slots:
            policy = self._policies[slot.index]
            if policy is not None:
                chunk = slice(slot.index * size, (slot.index + 1) * size)
                actions[chunk] = policy(self._obs[chunk])
        self._obs, rewards, dones, _ = self._env.step(actions)
        self._metrics.update(rewards, dones)
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
        self._metrics.set_active(slot_index, True)
