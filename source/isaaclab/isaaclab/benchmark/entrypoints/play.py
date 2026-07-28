# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line entrypoint for RL checkpoint playback benchmarks."""

from __future__ import annotations

from isaaclab.benchmark.dispatch import run_play_cli


def main(argv: list[str] | None = None) -> int:
    """Dispatch play benchmark arguments to the selected RL backend."""
    return run_play_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
