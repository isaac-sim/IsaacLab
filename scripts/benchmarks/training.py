# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dispatch training benchmarks to the selected RL library adapter."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
# the benchmark adapters import ``scripts.benchmarks.early_stop``, so the repo root must be importable
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from isaaclab_rl.entrypoints.common import dispatch_library_entrypoint

LIBRARY_ENTRYPOINTS = {
    "rsl_rl": _SCRIPT_DIR / "rsl_rl" / "benchmark_rsl_rl_train.py",
    "rl_games": _SCRIPT_DIR / "rl_games" / "benchmark_rl_games_train.py",
    "skrl": _SCRIPT_DIR / "skrl" / "benchmark_skrl_train.py",
    "sb3": _SCRIPT_DIR / "sb3" / "benchmark_sb3_train.py",
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the selected RL library's training-benchmark adapter."""
    return dispatch_library_entrypoint(
        argv,
        LIBRARY_ENTRYPOINTS,
        action="bench",
        description="Benchmark RL training with a selected reinforcement learning library.",
        library_help="Training library to benchmark.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
