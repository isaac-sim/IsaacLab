# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-GPU training entrypoint for Isaac Lab reinforcement learning workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.cli._multigpu import (
    MultiGpuLauncherSpec,
    build_distributed_command,
    parse_multigpu_args,
    run_multigpu,
)

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train.py"

DISTRIBUTED_LIBRARIES = ("rl_games", "rsl_rl", "skrl")
LAUNCHER_SPEC = MultiGpuLauncherSpec(
    target_script=TRAIN_SCRIPT,
    description="Launch multi-GPU RL training with the selected distributed launcher.",
    supported_libraries=DISTRIBUTED_LIBRARIES,
    epilog=(
        "Examples:\n"
        "  train_multigpu --num_gpus 4 --task Isaac-Cartpole\n"
        "  train_multigpu --rl_library skrl --num_gpus 2 --task Isaac-Cartpole\n"
        "  train_multigpu --rl_library skrl --num_gpus 2 --ml_framework jax --task Isaac-Cartpole\n"
        "\n"
        "All unrecognized arguments are forwarded to the selected training library."
    ),
)


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse multi-GPU launcher arguments and return forwarded training arguments."""
    return parse_multigpu_args(argv, LAUNCHER_SPEC)


def _build_distributed_command(args_cli: argparse.Namespace, train_args: list[str]) -> list[str]:
    """Build the distributed launcher command for multi-GPU training."""
    return build_distributed_command(args_cli, train_args, LAUNCHER_SPEC)


def main(argv: list[str] | None = None) -> int:
    """Launch multi-GPU training with the selected distributed launcher."""
    return run_multigpu(argv, LAUNCHER_SPEC)


if __name__ == "__main__":
    raise SystemExit(main())
