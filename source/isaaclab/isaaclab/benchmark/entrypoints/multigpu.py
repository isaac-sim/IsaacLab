# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""Multi-GPU launcher for the Isaac Lab benchmark workflows.

:func:`run_multigpu_benchmark_cli` builds a ``torchrun`` command and supervises it. That command
runs this same file once per rank, so the ``__main__`` block below is the per-rank worker: it
strips ``--workflow`` and hands the remaining arguments to the ordinary single-process workflow.

Every rank creates its own Isaac Lab instance, so ``--num_envs`` is the number of environments
*per rank*. Global rank 0 owns the emitted bundle; see
:mod:`isaaclab.benchmark.distributed` for how per-rank work is reported.

Usage example::

    uv run isaaclab benchmark startup_multigpu \\
        --task Isaac-Cartpole-Direct \\
        --num_gpus 2 \\
        presets=newton_mjwarp

    uv run isaaclab benchmark training_multigpu \\
        --rl_library rsl_rl \\
        --num_gpus 2 \\
        --task Isaac-Cartpole-Direct \\
        --max_iterations 100
"""

from __future__ import annotations

import sys
from pathlib import Path

from isaaclab.cli.multigpu import MultiGpuLauncherCfg, run_multigpu_cli

WORKER_SCRIPT = str(Path(__file__).resolve())

MULTIGPU_SUFFIX = "_multigpu"
"""Suffix that turns a benchmark workflow name into its multi-GPU variant."""

LEGACY_MULTIGPU_SUFFIX = "-multigpu"
"""Deprecated suffix retained for command-line compatibility."""

MULTIGPU_WORKFLOWS = ("startup", "runtime", "training")
"""Benchmark workflows that offer a multi-GPU variant."""

DISTRIBUTED_RL_LIBRARIES = ("rl_games", "rsl_rl", "skrl")
"""Training backends whose benchmark adapter supports synchronized multi-GPU training."""

_DESCRIPTIONS = {
    "startup": "Profile Isaac Lab startup phases while every rank launches concurrently.",
    "runtime": "Benchmark environment runtime with one independent workload per rank.",
    "training": "Benchmark synchronized multi-GPU RL training.",
}


def launcher_cfg(workflow: str) -> MultiGpuLauncherCfg:
    """Build the launcher configuration for a multi-GPU benchmark workflow.

    Args:
        workflow: Base benchmark workflow name, one of :data:`MULTIGPU_WORKFLOWS`.

    Returns:
        Configuration driving :func:`~isaaclab.cli.multigpu.run_multigpu_cli`.

    Raises:
        ValueError: If *workflow* has no multi-GPU variant.
    """
    if workflow not in MULTIGPU_WORKFLOWS:
        raise ValueError(f"Unsupported multi-GPU benchmark workflow: {workflow!r}")
    prog = f"benchmark {workflow}{MULTIGPU_SUFFIX}"
    return MultiGpuLauncherCfg(
        prog=prog,
        description=_DESCRIPTIONS[workflow],
        worker_script=WORKER_SCRIPT,
        worker_prefix_args=("--workflow", workflow),
        # skrl's JAX launcher exports its own rank variables and is not wired into the
        # benchmark adapters, so every workflow here launches through torchrun.
        rl_libraries=DISTRIBUTED_RL_LIBRARIES if workflow == "training" else (),
        supports_jax=False,
        epilog=(
            "Examples:\n"
            f"  {prog} --num_gpus 2 --task Isaac-Cartpole-Direct\n"
            f"  {prog} --nnodes 2 --node_rank 0 --num_gpus 8 --rdzv_backend c10d"
            " --rdzv_endpoint host0:29400 --rdzv_id bench --task Isaac-Cartpole-Direct\n"
            "\n"
            "--num_envs is per rank. All unrecognized arguments are forwarded to the workflow."
        ),
    )


def run_multigpu_benchmark_cli(workflow: str, argv: list[str] | None = None) -> int:
    """Launch a multi-GPU benchmark workflow.

    Args:
        workflow: Base benchmark workflow name, one of :data:`MULTIGPU_WORKFLOWS`.
        argv: Command-line arguments excluding the executable name. Uses ``sys.argv`` when ``None``.

    Returns:
        Process exit code.
    """
    return run_multigpu_cli(argv, launcher_cfg(workflow))


def run_worker(argv: list[str]) -> int:
    """Run one rank of a multi-GPU benchmark.

    Args:
        argv: Per-rank arguments, starting with ``--workflow <name>``.

    Returns:
        Process exit code.
    """
    import argparse
    import importlib

    from isaaclab.benchmark.dispatch import WORKFLOW_MODULES, run_training_cli

    parser = argparse.ArgumentParser(prog="benchmark multigpu worker", add_help=False)
    parser.add_argument("--workflow", required=True, choices=MULTIGPU_WORKFLOWS)
    args, workflow_argv = parser.parse_known_args(argv)

    if args.workflow == "training":
        return run_training_cli(workflow_argv)
    importlib.import_module(WORKFLOW_MODULES[args.workflow]).run(workflow_argv)
    return 0


if __name__ == "__main__":
    # Reached once per rank. ``record`` reports the failing rank's traceback to torchrun, which is
    # otherwise lost when console output is filtered to rank 0.
    from torch.distributed.elastic.multiprocessing.errors import record

    raise SystemExit(record(run_worker)(sys.argv[1:]))
