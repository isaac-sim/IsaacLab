# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified training entrypoint for Isaac Lab reinforcement learning workflows."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

LIBRARY_ENTRYPOINTS = {
    "rl_games": SCRIPT_DIR / "rl_games" / "train_rl_games.py",
    "rlinf": SCRIPT_DIR / "rlinf" / "train_rlinf.py",
    "rsl_rl": SCRIPT_DIR / "rsl_rl" / "train_rsl_rl.py",
    "sb3": SCRIPT_DIR / "sb3" / "train_sb3.py",
    "skrl": SCRIPT_DIR / "skrl" / "train_skrl.py",
}


def _parse_library(argv: list[str]) -> tuple[str | None, list[str]]:
    """Parse the selected training library without consuming library-specific arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--library", choices=sorted(LIBRARY_ENTRYPOINTS))
    args_cli, library_args = parser.parse_known_args(argv)
    return args_cli.library, library_args


def _print_top_level_help() -> None:
    """Print help for the unified entrypoint."""
    parser = argparse.ArgumentParser(description="Train an RL agent with a selected reinforcement learning library.")
    parser.add_argument(
        "--library",
        choices=sorted(LIBRARY_ENTRYPOINTS),
        required=True,
        help="Training library to use.",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected library.")
    parser.print_help()


def _load_library_entrypoint(library: str):
    """Load a library-specific training entrypoint from its existing workflow folder."""
    module_path = LIBRARY_ENTRYPOINTS[library]
    module_name = f"isaaclab_rl_train_{library}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training entrypoint for {library!r} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    """Run the selected reinforcement learning training library."""
    if argv is None:
        argv = sys.argv[1:]

    library, library_args = _parse_library(argv)
    if library is None:
        if "-h" in argv or "--help" in argv:
            _print_top_level_help()
            return 0
        _print_top_level_help()
        return 2

    module = _load_library_entrypoint(library)
    module.run(library_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
