# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend selection and execution for the unified RL entrypoints."""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import TYPE_CHECKING

import gymnasium as gym

if TYPE_CHECKING:
    from .simple_agents import PolicyName

_BACKEND_MODULES = {
    "export": {
        "rl_games": "isaaclab_rl.entrypoints.backends.export_rl_games",
        "rsl_rl": "isaaclab_rl.entrypoints.backends.export_rsl_rl",
        "sb3": "isaaclab_rl.entrypoints.backends.export_sb3",
        "skrl": "isaaclab_rl.entrypoints.backends.export_skrl",
    },
    "train": {
        "rl_games": "isaaclab_rl.entrypoints.backends.train_rl_games",
        "rlinf": "isaaclab_rl.entrypoints.backends.train_rlinf",
        "rsl_rl": "isaaclab_rl.entrypoints.backends.train_rsl_rl",
        "sb3": "isaaclab_rl.entrypoints.backends.train_sb3",
        "skrl": "isaaclab_rl.entrypoints.backends.train_skrl",
        "torchrl": "isaaclab_rl.entrypoints.backends.train_torchrl",
    },
    "play": {
        "rl_games": "isaaclab_rl.entrypoints.backends.play_rl_games",
        "rlinf": "isaaclab_rl.entrypoints.backends.play_rlinf",
        "rsl_rl": "isaaclab_rl.entrypoints.backends.play_rsl_rl",
        "sb3": "isaaclab_rl.entrypoints.backends.play_sb3",
        "skrl": "isaaclab_rl.entrypoints.backends.play_skrl",
        "torchrl": "isaaclab_rl.entrypoints.backends.play_torchrl",
    },
}


def run_train_cli(argv: list[str] | None = None) -> int:
    """Dispatch unified training command-line arguments to a backend."""
    return run_cli("train", argv)


def run_play_cli(argv: list[str] | None = None) -> int:
    """Dispatch unified playback command-line arguments to a backend."""
    return run_cli("play", argv)


def run_export_cli(argv: list[str] | None = None) -> int:
    """Dispatch unified LEAPP export command-line arguments to a backend."""
    # imported here so that importing the train and play entrypoints stays lightweight
    import torch

    # task registration imports decorated Isaac Lab math helpers, so disable TorchScript before
    # resolving a task's default export backend
    torch.jit._state.disable()
    return run_cli("export", argv)


def run_zero_agent_cli(argv: list[str] | None = None) -> int:
    """Dispatch command-line arguments to the zero-action agent."""
    return _run_simple_agent_cli("zero", argv)


def run_random_agent_cli(argv: list[str] | None = None) -> int:
    """Dispatch command-line arguments to the random-action agent."""
    return _run_simple_agent_cli("random", argv)


def run_cli(action: str, argv: list[str] | None = None) -> int:
    """Dispatch a unified RL command to its selected backend.

    Args:
        action: Workflow to execute, one of ``"train"``, ``"play"``, or ``"export"``.
        argv: Command-line arguments excluding the executable name.

    Returns:
        Process exit code.
    """
    if action not in _BACKEND_MODULES:
        raise ValueError(f"Unsupported RL action {action!r}. Expected one of: {sorted(_BACKEND_MODULES)}.")
    argv = _normalize_argv(argv)
    backends = _BACKEND_MODULES[action]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rl_library", choices=sorted(backends))
    selected, backend_argv = parser.parse_known_args(argv)
    library = selected.rl_library or _resolve_default_library(argv, backends)
    if library is None:
        _print_selector_help(action, sorted(backends))
        if "-h" in argv or "--help" in argv:
            return 0
        print(f"\n{action}: error: the following argument is required: --rl_library", file=sys.stderr)
        return 2
    status = _run_backend(backends[library], backend_argv)
    return status if status is not None else 0


def _run_simple_agent_cli(policy: PolicyName, argv: list[str] | None) -> int:
    """Run a checkpoint-free agent while isolating its command-line arguments.

    Args:
        policy: Action policy to apply, either ``"zero"`` or ``"random"``.
        argv: Command-line arguments excluding the executable name.

    Returns:
        Process exit code.
    """
    # imported here so that importing this module stays lightweight
    from .simple_agents import run

    argv = _normalize_argv(argv)
    original_argv = sys.argv
    try:
        run(argv, policy=policy)
    finally:
        sys.argv = original_argv
    return 0


def _normalize_argv(argv: list[str] | None) -> list[str]:
    """Return the command line to dispatch with space-separated Kit arguments fused.

    The backends parse this explicit list rather than ``sys.argv``, so the fusing that
    :meth:`~isaaclab.app.AppLauncher.add_app_launcher_args` applies to ``sys.argv`` never reaches it.
    """
    # imported here so that importing this module stays lightweight
    from isaaclab.app import AppLauncher

    if argv is None:
        argv = sys.argv[1:]
    return AppLauncher._fuse_kit_args(argv)


def _resolve_default_library(argv: list[str], backends: dict[str, str]) -> str | None:
    """Return the task-registered default RL library requested by the command line."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task")
    args, _ = parser.parse_known_args(argv)
    if args.task is None:
        return None

    # task registration is deferred until a task name asks for it
    import isaaclab_tasks  # noqa: F401

    try:
        default_library = gym.spec(args.task.split(":")[-1]).kwargs.get("default_agent")
    except gym.error.Error:
        return None
    return default_library if default_library in backends else None


def _print_selector_help(action: str, backends: list[str]) -> None:
    """Print help for a unified entrypoint before a backend is selected."""
    parser = argparse.ArgumentParser(description=f"{action.capitalize()} an RL agent with a selected backend.")
    parser.add_argument("--rl_library", choices=backends, required=True, help="Reinforcement learning backend to use.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected backend.")
    parser.print_help()


def _run_backend(module_name: str, argv: list[str]) -> int | None:
    """Run a backend module's ``run(argv)`` while isolating its command-line arguments."""
    module = importlib.import_module(module_name)
    original_argv = sys.argv
    try:
        return module.run(argv)
    finally:
        sys.argv = original_argv
