# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified training entrypoint for Isaac Lab reinforcement learning workflows."""

from __future__ import annotations

from pathlib import Path

from common import dispatch_library_entrypoint

SCRIPT_DIR = Path(__file__).resolve().parent

LIBRARY_ENTRYPOINTS = {
    "rl_games": SCRIPT_DIR / "rl_games" / "train_rl_games.py",
    "rlinf": SCRIPT_DIR / "rlinf" / "train_rlinf.py",
    "rsl_rl": SCRIPT_DIR / "rsl_rl" / "train_rsl_rl.py",
    "sb3": SCRIPT_DIR / "sb3" / "train_sb3.py",
    "skrl": SCRIPT_DIR / "skrl" / "train_skrl.py",
}


def main(argv: list[str] | None = None) -> int:
    """Run the selected reinforcement learning training library."""
    return dispatch_library_entrypoint(
        argv,
        LIBRARY_ENTRYPOINTS,
        action="train",
        description="Train an RL agent with a selected reinforcement learning library.",
        library_help="Training library to use.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
