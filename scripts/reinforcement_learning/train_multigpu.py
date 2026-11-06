# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-GPU training executable for Isaac Lab reinforcement learning workflows."""

from isaaclab_rl.entrypoints import run_train_multigpu_cli


def main(argv: list[str] | None = None) -> int:
    """Launch multi-GPU training with the selected distributed launcher."""
    return run_train_multigpu_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
