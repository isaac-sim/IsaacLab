# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-GPU launcher for Isaac Lab reinforcement learning training.

:func:`run_train_multigpu_cli` builds a ``torchrun`` (or skrl JAX) command and supervises it. That
command runs this same file once per rank, so the ``__main__`` block below is the per-rank trainer.
"""

from __future__ import annotations

# Warp captures ``enable_backward`` when a module is created, which happens at import time, so it
# has to be set before importing anything that defines Warp kernels.
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

from torch.distributed.elastic.multiprocessing.errors import record  # noqa: E402

from isaaclab.cli.multigpu import (  # noqa: E402
    SKRL_JAX_ARGS,
    SKRL_JAX_TORCHRUN_ONLY_ARGS,
    TORCHRUN_ARGS,
    MultiGpuLauncherCfg,
    build_launch_command,
    parse_launcher_args,
    run_multigpu_cli,
)

# Absolute imports: each rank runs this file as a script, where relative imports have no package.
from isaaclab_rl.entrypoints.api import MULTI_GPU_BACKENDS  # noqa: E402
from isaaclab_rl.entrypoints.dispatch import run_train_cli  # noqa: E402

__all__ = ["SKRL_JAX_ARGS", "SKRL_JAX_TORCHRUN_ONLY_ARGS", "TORCHRUN_ARGS", "WORKER_SCRIPT", "run_train_multigpu_cli"]

WORKER_SCRIPT = str(Path(__file__).resolve())

LAUNCHER_CFG = MultiGpuLauncherCfg(
    prog="train-multigpu",
    description="Launch multi-GPU RL training with the selected distributed launcher.",
    worker_script=WORKER_SCRIPT,
    rl_libraries=MULTI_GPU_BACKENDS,
    epilog=(
        "Examples:\n"
        "  train-multigpu --num_gpus 4 --task Isaac-Cartpole\n"
        "  train-multigpu --rl_library skrl --num_gpus 2 --task Isaac-Cartpole\n"
        "  train-multigpu --rl_library skrl --num_gpus 2 --ml_framework jax --task Isaac-Cartpole\n"
        "\n"
        "All unrecognized arguments are forwarded to the selected training library."
    ),
)


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse multi-GPU launcher arguments and return forwarded training arguments."""
    return parse_launcher_args(argv, LAUNCHER_CFG)


def _build_distributed_command(args_cli: argparse.Namespace, train_args: list[str]) -> list[str]:
    """Build the distributed launcher command for multi-GPU training."""
    return build_launch_command(args_cli, train_args, LAUNCHER_CFG)


def run_train_multigpu_cli(argv: list[str] | None = None) -> int:
    """Launch multi-GPU training with the selected distributed launcher.

    Args:
        argv: Command-line arguments excluding the executable name.

    Returns:
        Process exit code.
    """
    return run_multigpu_cli(argv, LAUNCHER_CFG)


if __name__ == "__main__":
    # Reached once per rank. ``record`` reports the failing rank's traceback to torchrun, which is
    # otherwise lost when console output is filtered to rank 0.
    raise SystemExit(record(run_train_cli)())
