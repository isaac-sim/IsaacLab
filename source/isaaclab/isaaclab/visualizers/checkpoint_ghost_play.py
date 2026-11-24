# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The library-independent parts of Checkpoint Ghosts that a ``play`` script calls.

A play script adds the flags with :func:`add_args`, checks and prepares the run with :func:`check` and :func:`prepare`,
shows the greyed-out HUD section of a plain run with :func:`offer`, and hands its policy loading and playback loop to
``run`` when ``--checkpoint_ghosts`` is set. Every function but :func:`add_args`, :func:`offer`, and :func:`run` does
nothing without the flag, so a script that does not use the feature behaves exactly as before.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.utils.checkpoint_ghost_sync import CheckpointGhostSync
from isaaclab.utils.seed import configure_seed

from .checkpoint_ghost import CheckpointGhostCheckpoint
from .checkpoint_ghost_session import MAX_CHECKPOINT_GHOST_SLOTS, CheckpointGhostSession

logger = logging.getLogger(__name__)

_RUN_MANIFEST_FILENAME = "run.json"
"""Name of the manifest that training writes into a run directory (``isaaclab_rl.entrypoints.common``)."""


def _slot_count(value: str) -> int:
    """Parse ``--checkpoint_ghost_slots``, so a bad count fails at argument parsing like any other flag."""
    count = int(value)
    if not 2 <= count <= MAX_CHECKPOINT_GHOST_SLOTS:
        raise argparse.ArgumentTypeError(f"must be between 2 and {MAX_CHECKPOINT_GHOST_SLOTS}.")
    return count


def add_args(parser: argparse.ArgumentParser) -> None:
    """Add the Checkpoint Ghosts flags to a play script's parser."""
    parser.add_argument(
        "--checkpoint_ghosts",
        action="store_true",
        default=False,
        help=(
            "Enable Checkpoint Ghosts, the checkpoint visualization in the Newton GL viewer. The --checkpoint drives"
            " the reference; ghosts are added from the viewer HUD. --num_envs is the total simulated and is split"
            " evenly among the checkpoints shown."
        ),
    )
    parser.add_argument(
        "--checkpoint_ghost_slots",
        type=_slot_count,
        default=None,
        help=(
            f"Most checkpoints shown at once, including the reference (2 to {MAX_CHECKPOINT_GHOST_SLOTS}). Defaults to"
            f" {MAX_CHECKPOINT_GHOST_SLOTS}, capped at --num_envs."
        ),
    )
    parser.add_argument(
        "--checkpoint_ghost_root",
        default=None,
        help="Directory searched for ghost checkpoints. Defaults to the experiment's log directory.",
    )


def check(args_cli: argparse.Namespace, env_cfg: Any) -> None:
    """Reject a configuration Checkpoint Ghosts cannot run, before any simulation starts.

    Does nothing without ``--checkpoint_ghosts``.
    """
    if not args_cli.checkpoint_ghosts:
        return
    if not isinstance(env_cfg, (ManagerBasedRLEnvCfg, DirectRLEnvCfg)):
        raise SystemExit("--checkpoint_ghosts only supports single-agent environments.")
    visualizers = env_cfg.sim.visualizer_cfgs or []
    if not isinstance(visualizers, (list, tuple)):
        visualizers = [visualizers]
    # A shared visualizer configuration names its type on its backend; a deprecated combined one names it directly.
    types = []
    for cfg in visualizers:
        backend = getattr(cfg, "backend", None)
        for chosen in backend if isinstance(backend, (list, tuple)) else [backend]:  # --viz newton,viser selects both
            types.append(getattr(chosen, "visualizer_type", None) or getattr(cfg, "visualizer_type", None))
    if "newton" not in types:
        raise SystemExit("--checkpoint_ghosts requires the Newton GL visualizer; pass --visualizer newton.")
    curriculum = getattr(env_cfg, "curriculum", None)
    for name, term in vars(curriculum).items() if curriculum is not None else ():
        # Checkpoint Ghosts disable curriculum terms, so a task whose reset events read a term's state cannot run.
        if getattr(getattr(term, "func", None), "supports_checkpoint_ghosts", True) is False:
            raise SystemExit(
                f"--checkpoint_ghosts does not support this task: its curriculum term '{name}' sets the reset state."
            )


def prepare(args_cli: argparse.Namespace, env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> None:
    """Resolve the slot count and disable randomization that cannot be mirrored, before the environment is created.

    ``--num_envs`` is the total simulated; the environments are split among the loaded checkpoints, so every slot needs
    at least one. Does nothing without ``--checkpoint_ghosts``.
    """
    if not args_cli.checkpoint_ghosts:
        return
    total = env_cfg.scene.num_envs
    if total < 2:
        raise SystemExit("--checkpoint_ghosts needs --num_envs of at least 2: one environment per checkpoint.")
    slots = args_cli.checkpoint_ghost_slots
    if slots is None:
        slots = MAX_CHECKPOINT_GHOST_SLOTS
    if slots > total:
        if args_cli.checkpoint_ghost_slots is not None:
            logger.warning(f"--checkpoint_ghost_slots {slots} exceeds --num_envs {total}; using {total} slots.")
        slots = total
    args_cli.checkpoint_ghost_slots = min(slots, MAX_CHECKPOINT_GHOST_SLOTS)
    CheckpointGhostSync.configure_env_cfg(env_cfg)
    if isinstance(env_cfg, DirectRLEnvCfg):
        logger.warning(
            "Direct environment: reset randomization is mirrored, but goals or commands a task resamples during an"
            " episode are not, so ghosts may chase different targets."
        )


def offer(env: Any, env_cfg: Any) -> None:
    """Show the greyed-out Checkpoint Ghosts section in the Newton GL HUD of a run that started without the feature."""
    if not isinstance(env_cfg, (ManagerBasedRLEnvCfg, DirectRLEnvCfg)):
        return
    for visualizer in env.unwrapped.sim.visualizers:
        if hasattr(visualizer, "offer_checkpoint_ghosts"):
            visualizer.offer_checkpoint_ghosts()


def run(
    env: Any,
    args_cli: argparse.Namespace,
    *,
    policy_factory: Callable[[str], Any],
    checkpoint_pattern: str,
    reference_path: str,
    log_root_path: str,
    run_playback: Callable[..., None],
) -> None:
    """Play the reference checkpoint with ghost slots that the Newton GL viewer HUD fills.

    Args:
        env: The library's environment wrapper.
        args_cli: Parsed play arguments.
        policy_factory: Builds the inference policy of a checkpoint file.
        checkpoint_pattern: Regular expression matching the library's checkpoint file names.
        reference_path: Checkpoint driving the reference slot.
        log_root_path: The experiment's log directory, where the checkpoint window starts by default.
        run_playback: The play script's playback loop, called with ``step``, ``dt``, ``args_cli`` and ``env_cfg``.
    """
    if args_cli.deterministic:
        configure_seed(env.unwrapped.cfg.seed, torch_deterministic=True)  # no runner is built here, so seed up front
    reference_file = Path(reference_path)
    session = CheckpointGhostSession(
        env.unwrapped,  # the Isaac Lab environment: its observation groups and 5-tuple step, free of library wrappers
        num_slots=args_cli.checkpoint_ghost_slots,
        task=args_cli.task,
        checkpoint_root=os.path.abspath(args_cli.checkpoint_ghost_root or log_root_path),
        checkpoint_pattern=checkpoint_pattern,
        policy_factory=policy_factory,
        reference=CheckpointGhostCheckpoint(
            reference_path, reference_file.parent.name, reference_file.name, reference_file.stat().st_mtime
        ),
        run_task=_run_task,
    )
    try:  # the session patches the environment's reset and commands, which must be undone on any failure
        attached = False
        for visualizer in env.unwrapped.sim.visualizers:
            if hasattr(visualizer, "attach_checkpoint_ghosts"):
                visualizer.attach_checkpoint_ghosts(session)
                attached = True
        if not attached:
            raise RuntimeError("--checkpoint_ghosts needs an initialized Newton GL visualizer.")
        run_playback(session.step, dt=env.unwrapped.step_dt, args_cli=args_cli, env_cfg=env.unwrapped.cfg)
    finally:
        session.close()


class PolicyAdapter:
    """Turns a library's act function into the policy a slot needs: observation slice in, clipped actions out."""

    def __init__(
        self,
        act: Callable[[Any], torch.Tensor],
        reset: Callable[[torch.Tensor | None], None] | None = None,
        clip: float | None = None,
    ):
        self._act, self._reset, self._clip = act, reset, clip

    def __call__(self, obs: Any) -> torch.Tensor:  # noqa: D102
        actions = self._act(obs)
        return actions if self._clip is None else torch.clamp(actions, -self._clip, self._clip)

    def reset(self, dones: torch.Tensor | None = None) -> None:  # noqa: D102
        if self._reset is not None:
            self._reset(dones)


def _run_task(directory: str) -> str | None:
    """Return the task named by the run manifest in ``directory``, or None when it is not a training run."""
    try:
        return json.loads((Path(directory) / _RUN_MANIFEST_FILENAME).read_text(encoding="utf-8"))["task"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
