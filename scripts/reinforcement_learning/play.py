# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified playback executable for Isaac Lab reinforcement learning workflows."""

# Warp captures ``enable_backward`` when a module is created, which happens at import
# time, so it has to be set before importing anything that defines Warp kernels.
# Isaac Lab does not use Warp autodiff; skipping adjoint codegen roughly halves the
# time spent building kernels on a cold kernel cache.
import warp as wp

wp.config.enable_backward = False

from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the selected reinforcement learning play library."""
    return run_play_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
