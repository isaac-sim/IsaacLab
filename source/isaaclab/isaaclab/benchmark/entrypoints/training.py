# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line entrypoint for RL training benchmarks."""

from __future__ import annotations

import re
from pathlib import Path

from isaaclab.benchmark.dispatch import run_training_cli

_TRAINING_CHECKPOINTS = {
    "rl_games": ("nn", r".*[.]pth$"),
    "skrl": ("checkpoints", r"(?:agent_.*|best_agent)[.]pt$"),
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch training benchmark arguments to the selected RL backend."""
    return run_training_cli(argv)


def _resolve_training_checkpoint_path(log_dir: str, backend: str) -> str | None:
    """Resolve the checkpoint written by a training benchmark, if any."""
    from isaaclab_tasks.utils import get_checkpoint_path

    checkpoint_dir, checkpoint_pattern = _TRAINING_CHECKPOINTS[backend]
    run_path = Path(log_dir)
    try:
        return get_checkpoint_path(
            str(run_path.parent),
            rf"^{re.escape(run_path.name)}$",
            checkpoint_pattern,
            other_dirs=[checkpoint_dir],
        )
    except (FileNotFoundError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
