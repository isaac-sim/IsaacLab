# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified play entrypoint for Isaac Lab reinforcement learning workflows."""

from __future__ import annotations

from pathlib import Path

from common import dispatch_library_entrypoint

SCRIPT_DIR = Path(__file__).resolve().parent

LIBRARY_ENTRYPOINTS = {
    "rl_games": SCRIPT_DIR / "rl_games" / "play_rl_games.py",
    "rlinf": SCRIPT_DIR / "rlinf" / "play_rlinf.py",
    "rsl_rl": SCRIPT_DIR / "rsl_rl" / "play_rsl_rl.py",
    "sb3": SCRIPT_DIR / "sb3" / "play_sb3.py",
    "skrl": SCRIPT_DIR / "skrl" / "play_skrl.py",
}


def main(argv: list[str] | None = None) -> int:
    """Run the selected reinforcement learning play library."""
    return dispatch_library_entrypoint(
        argv,
        LIBRARY_ENTRYPOINTS,
        action="play",
        description="Play an RL agent with a selected reinforcement learning library.",
        library_help="Training library used by the checkpoint.",
        run_as_script=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
