# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified play-benchmark dispatcher (mirrors scripts/reinforcement_learning/play.py).

Selects the RL library via ``--rl_library`` and forwards remaining args to the matching
benchmark adapter under ``scripts/benchmarks/<library>/play_benchmark.py``, which rolls
out a checkpointed policy under a ``BenchmarkMonitor`` and emits a ``PlayBundle``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_RL_SCRIPTS = _SCRIPT_DIR.parent / "reinforcement_learning"
if str(_RL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_RL_SCRIPTS))

from common import dispatch_library_entrypoint  # noqa: E402

LIBRARY_ENTRYPOINTS = {
    "rsl_rl": _SCRIPT_DIR / "rsl_rl" / "play_benchmark.py",
    "rl_games": _SCRIPT_DIR / "rl_games" / "play_benchmark.py",
    "skrl": _SCRIPT_DIR / "skrl" / "play_benchmark.py",
    "sb3": _SCRIPT_DIR / "sb3" / "play_benchmark.py",
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the selected RL library's play-benchmark adapter."""
    return dispatch_library_entrypoint(
        argv,
        LIBRARY_ENTRYPOINTS,
        action="bench_play",
        description="Benchmark RL inference (play) with a selected reinforcement learning library.",
        library_help="Inference library to benchmark.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
