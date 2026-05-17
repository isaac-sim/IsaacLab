# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-GPU training entrypoint for Isaac Lab reinforcement learning workflows."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train.py"

DISTRIBUTED_LIBRARIES = ("rl_games", "rsl_rl", "skrl")


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse multi-GPU launcher arguments and return forwarded training arguments."""
    parser = argparse.ArgumentParser(
        description="Launch multi-GPU RL training with torch.distributed.run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog=(
            "Examples:\n"
            "  train_multigpu --num_gpus 4 --task Isaac-Cartpole-v0 --headless\n"
            "  train_multigpu --rl_library skrl --num_gpus 2 --task Isaac-Cartpole-v0 --headless\n"
            "\n"
            "All unrecognized arguments are forwarded to the selected training library."
        ),
    )
    parser.add_argument(
        "--rl_library",
        choices=DISTRIBUTED_LIBRARIES,
        default="rsl_rl",
        help="Distributed-capable training library to use. Defaults to rsl_rl.",
    )
    parser.add_argument(
        "--num_gpus",
        "--nproc_per_node",
        dest="nproc_per_node",
        default="gpu",
        help=(
            "Number of trainer processes to launch on each node. Accepts an integer or torchrun values "
            "'gpu', 'cpu', and 'auto'. Defaults to 'gpu'."
        ),
    )
    parser.add_argument("--nnodes", default=None, help="Number of nodes to use for distributed training.")
    parser.add_argument("--node_rank", default=None, help="Rank of this node in a multi-node job.")
    parser.add_argument("--master_addr", default=None, help="Master node address for static rendezvous.")
    parser.add_argument("--master_port", default=None, help="Master node port for static rendezvous.")
    parser.add_argument("--rdzv_backend", default=None, help="Rendezvous backend used by torchrun.")
    parser.add_argument("--rdzv_endpoint", default=None, help="Rendezvous endpoint used by torchrun.")
    parser.add_argument("--rdzv_id", default=None, help="User-defined rendezvous id used by torchrun.")
    parser.add_argument("--max_restarts", default=None, help="Maximum worker group restarts before failing.")
    parser.add_argument("--monitor_interval", default=None, help="Worker monitor interval [s].")
    parser.add_argument(
        "--start_method",
        choices=("spawn", "fork", "forkserver"),
        default=None,
        help="Multiprocessing start method used by torchrun.",
    )
    parser.add_argument("--role", default=None, help="User-defined worker role used by torchrun.")
    parser.add_argument("--tee", default=None, help="Tee selected worker stdout/stderr streams.")
    parser.add_argument("--redirects", default=None, help="Redirect selected worker stdout/stderr streams.")
    parser.add_argument("--local_ranks_filter", default=None, help="Only show logs from the listed local ranks.")
    parser.add_argument("--log_dir", default=None, help="Directory used by torchrun for worker logs.")
    parser.add_argument("--dry_run", action="store_true", help="Print the torchrun command without launching it.")

    args_cli, train_args = parser.parse_known_args(argv)
    if train_args[:1] == ["--"]:
        train_args = train_args[1:]
    return args_cli, train_args


def _append_optional_torchrun_arg(command: list[str], args_cli: argparse.Namespace, name: str) -> None:
    """Append a torchrun argument when it was provided."""
    value = getattr(args_cli, name)
    if value is not None:
        command.extend([f"--{name}", str(value)])


def _with_distributed_arg(train_args: list[str]) -> list[str]:
    """Ensure the selected training library receives the distributed flag."""
    if "--distributed" in train_args:
        return train_args
    return ["--distributed", *train_args]


def _build_torchrun_command(args_cli: argparse.Namespace, train_args: list[str]) -> list[str]:
    """Build the torchrun command for multi-GPU training."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(args_cli.nproc_per_node),
    ]
    for name in (
        "nnodes",
        "node_rank",
        "master_addr",
        "master_port",
        "rdzv_backend",
        "rdzv_endpoint",
        "rdzv_id",
        "max_restarts",
        "monitor_interval",
        "start_method",
        "role",
        "tee",
        "redirects",
        "local_ranks_filter",
        "log_dir",
    ):
        _append_optional_torchrun_arg(command, args_cli, name)

    command.extend(
        [
            str(TRAIN_SCRIPT),
            "--rl_library",
            args_cli.rl_library,
            *_with_distributed_arg(train_args),
        ]
    )
    return command


def main(argv: list[str] | None = None) -> int:
    """Launch multi-GPU training with ``torch.distributed.run``."""
    if argv is None:
        argv = sys.argv[1:]

    args_cli, train_args = _parse_args(argv)
    command = _build_torchrun_command(args_cli, train_args)

    if args_cli.dry_run:
        print(shlex.join(command))
        return 0

    print(f"[INFO] Launching distributed training with: {shlex.join(command)}")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
