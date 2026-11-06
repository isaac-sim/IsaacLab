# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay recorded demonstrations; see :mod:`isaaclab_teleop.entrypoints.replay_demos`."""

# Warp captures ``enable_backward`` when a module is created, which happens at import
# time, so it has to be set before importing anything that defines Warp kernels.
# Isaac Lab does not use Warp autodiff; skipping adjoint codegen roughly halves the
# time spent building kernels on a cold kernel cache.
import warp as wp

wp.config.enable_backward = False

from isaaclab_teleop.entrypoints import run_replay_demos_cli  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Replay recorded demonstrations in Isaac Lab environments."""
    return run_replay_demos_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
