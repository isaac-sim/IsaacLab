# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend selection and execution for unified RL entrypoints."""

from __future__ import annotations

import argparse
import importlib
import runpy
import sys

_BACKEND_MODULES = {
    "train": {
        "rl_games": "isaaclab_rl.entrypoints.backends.train_rl_games",
        "rlinf": "isaaclab_rl.entrypoints.backends.train_rlinf",
        "rsl_rl": "isaaclab_rl.entrypoints.backends.train_rsl_rl",
        "sb3": "isaaclab_rl.entrypoints.backends.train_sb3",
        "skrl": "isaaclab_rl.entrypoints.backends.train_skrl",
    },
    "play": {
        "rl_games": "isaaclab_rl.entrypoints.backends.play_rl_games",
        "rlinf": "isaaclab_rl.entrypoints.backends.play_rlinf",
        "rsl_rl": "isaaclab_rl.entrypoints.backends.play_rsl_rl",
        "sb3": "isaaclab_rl.entrypoints.backends.play_sb3",
        "skrl": "isaaclab_rl.entrypoints.backends.play_skrl",
    },
}


def run_train_cli(argv: list[str] | None = None) -> int:
    """Dispatch unified training command-line arguments to a backend."""
    return run_cli("train", argv)


def run_play_cli(argv: list[str] | None = None) -> int:
    """Dispatch unified playback command-line arguments to a backend."""
    return run_cli("play", argv)


def run_cli(action: str, argv: list[str] | None = None) -> int:
    """Dispatch a unified RL command to its selected backend.

    Args:
        action: Workflow to execute, either ``"train"`` or ``"play"``.
        argv: Command-line arguments excluding the executable name.

    Returns:
        Process exit code.
    """
    if action not in _BACKEND_MODULES:
        raise ValueError(f"Unsupported RL action {action!r}. Expected one of: {sorted(_BACKEND_MODULES)}.")
    # imported locally so that importing this module stays lightweight
    from isaaclab.app import AppLauncher

    if argv is None:
        argv = sys.argv[1:]
    # the backends parse this explicit list (not sys.argv), so the sys.argv fusing in
    # AppLauncher.add_app_launcher_args never reaches it; normalize here instead
    argv = AppLauncher._fuse_kit_args(argv)
    backends = _BACKEND_MODULES[action]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rl_library", choices=sorted(backends))
    selected, backend_argv = parser.parse_known_args(argv)
    if selected.rl_library is None:
        _print_selector_help(action, sorted(backends))
        if "-h" in argv or "--help" in argv:
            return 0
        print(f"\n{action}: error: the following argument is required: --rl_library", file=sys.stderr)
        return 2
    _run_backend(backends[selected.rl_library], backend_argv, run_as_script=action == "play")
    return 0


def _print_selector_help(action: str, backends: list[str]) -> None:
    """Print help for a unified entrypoint before a backend is selected."""
    parser = argparse.ArgumentParser(description=f"{action.capitalize()} an RL agent with a selected backend.")
    parser.add_argument("--rl_library", choices=backends, required=True, help="Reinforcement learning backend to use.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected backend.")
    parser.print_help()


def _run_backend(module_name: str, argv: list[str], *, run_as_script: bool) -> None:
    """Run a backend module while isolating its command-line arguments."""
    if not run_as_script:
        module = importlib.import_module(module_name)
        runner = getattr(module, "run", None)
        if not callable(runner):
            raise TypeError(f"Training backend {module_name!r} does not define run(argv).")
        original_argv = sys.argv
        try:
            runner(argv)
        finally:
            sys.argv = original_argv
        return
    original_argv = sys.argv
    try:
        sys.argv = [module_name] + argv
        runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = original_argv
