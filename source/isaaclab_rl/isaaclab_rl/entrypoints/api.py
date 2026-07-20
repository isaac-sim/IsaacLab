# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Programmatic interfaces for Isaac Lab reinforcement learning workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .dispatch import run_play_cli, run_train_cli

BackendName = Literal["rl_games", "rlinf", "rsl_rl", "sb3", "skrl"]


@dataclass(frozen=True)
class TrainingRequest:
    """Parameters shared by the unified training workflows.

    Args:
        backend: Reinforcement learning backend to run.
        task: Registered Gym task identifier.
        checkpoint: Checkpoint path or a backend-supported selector to resume training from.
        agent: Optional task agent configuration entry point.
        num_envs: Number of environments to simulate.
        seed: Environment and agent seed.
        max_iterations: Maximum training iterations.
        device: Simulation device identifier.
        video: Whether to record training video.
        distributed: Whether to enable distributed training.
        backend_args: Backend-specific command-line arguments.
        hydra_args: Hydra overrides and typed preset selectors.
    """

    backend: BackendName
    task: str
    checkpoint: str | None = None
    agent: str | None = None
    num_envs: int | None = None
    seed: int | None = None
    max_iterations: int | None = None
    device: str | None = None
    video: bool = False
    distributed: bool = False
    backend_args: tuple[str, ...] = field(default_factory=tuple)
    hydra_args: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PlaybackRequest:
    """Parameters shared by the unified playback workflows.

    Args:
        backend: Reinforcement learning backend to run.
        task: Registered Gym task identifier.
        checkpoint: Checkpoint path or a backend-supported selector.
        agent: Optional task agent configuration entry point.
        num_envs: Number of environments to simulate.
        seed: Environment and agent seed.
        device: Simulation device identifier.
        video: Whether to record playback video.
        backend_args: Backend-specific command-line arguments.
        hydra_args: Hydra overrides and typed preset selectors.
    """

    backend: BackendName
    task: str
    checkpoint: str | None = None
    agent: str | None = None
    num_envs: int | None = None
    seed: int | None = None
    device: str | None = None
    video: bool = False
    backend_args: tuple[str, ...] = field(default_factory=tuple)
    hydra_args: tuple[str, ...] = field(default_factory=tuple)


def train(request: TrainingRequest) -> int:
    """Run a training workflow for the requested backend.

    Args:
        request: Typed training parameters.

    Returns:
        Process exit code.
    """
    return run_train_cli(_training_argv(request))


def play(request: PlaybackRequest) -> int:
    """Run a playback workflow for the requested backend.

    Args:
        request: Typed playback parameters.

    Returns:
        Process exit code.
    """
    return run_play_cli(_playback_argv(request))


def _training_argv(request: TrainingRequest) -> list[str]:
    argv = ["--rl_library", request.backend, "--task", request.task]
    _append_value(argv, "--model_path" if request.backend == "rlinf" else "--checkpoint", request.checkpoint)
    _append_value(argv, "--agent", request.agent)
    _append_value(argv, "--num_envs", request.num_envs)
    _append_value(argv, "--seed", request.seed)
    _append_value(argv, "--max_epochs" if request.backend == "rlinf" else "--max_iterations", request.max_iterations)
    _append_value(argv, "--device", request.device)
    if request.video:
        argv.append("--video")
    # SB3 does not support distributed training; skip the flag for that backend.
    if request.distributed and request.backend != "sb3":
        argv.append("--distributed")
    return argv + list(request.backend_args) + list(request.hydra_args)


def _playback_argv(request: PlaybackRequest) -> list[str]:
    argv = ["--rl_library", request.backend, "--task", request.task]
    _append_value(argv, "--model_path" if request.backend == "rlinf" else "--checkpoint", request.checkpoint)
    _append_value(argv, "--agent", request.agent)
    _append_value(argv, "--num_envs", request.num_envs)
    _append_value(argv, "--seed", request.seed)
    _append_value(argv, "--device", request.device)
    if request.video:
        argv.append("--video")
    return argv + list(request.backend_args) + list(request.hydra_args)


def _append_value(argv: list[str], option: str, value: str | int | None) -> None:
    """Append an option when its value was provided."""
    if value is not None:
        argv.extend((option, str(value)))
