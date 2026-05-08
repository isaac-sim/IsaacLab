# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified play entrypoint for Isaac Lab reinforcement learning workflows."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

LIBRARY_ENTRYPOINTS = {
    "rl_games": SCRIPT_DIR / "rl_games" / "play_rl_games.py",
    "rlinf": SCRIPT_DIR / "rlinf" / "play_rlinf.py",
    "rsl_rl": SCRIPT_DIR / "rsl_rl" / "play_rsl_rl.py",
    "sb3": SCRIPT_DIR / "sb3" / "play_sb3.py",
    "skrl": SCRIPT_DIR / "skrl" / "play_skrl.py",
}


def _parse_library(argv: list[str]) -> tuple[str | None, list[str]]:
    """Parse the selected play library without consuming library-specific arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--library", choices=sorted(LIBRARY_ENTRYPOINTS))
    args_cli, library_args = parser.parse_known_args(argv)
    return args_cli.library, library_args


def _print_top_level_help() -> None:
    """Print help for the unified play entrypoint."""
    parser = argparse.ArgumentParser(description="Play an RL agent with a selected reinforcement learning library.")
    parser.add_argument(
        "--library",
        choices=sorted(LIBRARY_ENTRYPOINTS),
        required=True,
        help="Training library used by the checkpoint.",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected library.")
    parser.print_help()


def _run_library_entrypoint(library: str, library_args: list[str]) -> None:
    """Run a library-specific play script from its existing workflow folder."""
    module_path = LIBRARY_ENTRYPOINTS[library]
    original_argv = sys.argv
    original_path = list(sys.path)
    try:
        sys.argv = [str(module_path)] + library_args
        sys.path.insert(0, str(module_path.parent))
        runpy.run_path(str(module_path), run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path


def main(argv: list[str] | None = None) -> int:
    """Run the selected reinforcement learning play library."""
    if argv is None:
        argv = sys.argv[1:]

    library, library_args = _parse_library(argv)
    if library is None:
        if "-h" in argv or "--help" in argv:
            _print_top_level_help()
            return 0
        _print_top_level_help()
        return 2

    _run_library_entrypoint(library, library_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
