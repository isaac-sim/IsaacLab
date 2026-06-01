# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-GPU training entrypoint for Isaac Lab reinforcement learning workflows."""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from types import FrameType

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train.py"

DISTRIBUTED_LIBRARIES = ("rl_games", "rsl_rl", "skrl")
SKRL_JAX_TORCHRUN_ONLY_ARGS = (
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
)


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse multi-GPU launcher arguments and return forwarded training arguments."""
    parser = argparse.ArgumentParser(
        description="Launch multi-GPU RL training with the selected distributed launcher.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog=(
            "Examples:\n"
            "  train_multigpu --num_gpus 4 --task Isaac-Cartpole\n"
            "  train_multigpu --rl_library skrl --num_gpus 2 --task Isaac-Cartpole\n"
            "  train_multigpu --rl_library skrl --num_gpus 2 --ml_framework jax "
            "--task Isaac-Cartpole\n"
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
            "'gpu', 'cpu', and 'auto'. skrl JAX training requires an integer. Defaults to 'gpu'."
        ),
    )
    parser.add_argument("--nnodes", default=None, help="Number of nodes to use for distributed training.")
    parser.add_argument("--node_rank", default=None, help="Rank of this node in a multi-node job.")
    parser.add_argument(
        "--coordinator_address",
        default=None,
        help="IP address and port where skrl JAX process 0 starts the JAX coordinator service.",
    )
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
    parser.add_argument(
        "--dry_run", action="store_true", help="Print the distributed launcher command without running it."
    )

    args_cli, train_args = parser.parse_known_args(argv)
    if train_args[:1] == ["--"]:
        train_args = train_args[1:]
    _validate_launcher_args(parser, args_cli, train_args)
    return args_cli, train_args


def _append_optional_launcher_arg(command: list[str], args_cli: argparse.Namespace, name: str) -> None:
    """Append a launcher argument when it was provided."""
    value = getattr(args_cli, name)
    if value is not None:
        command.extend([f"--{name}", str(value)])


def _with_distributed_arg(train_args: list[str]) -> list[str]:
    """Ensure the selected training library receives the distributed flag."""
    if "--distributed" in train_args:
        return train_args
    return [*train_args, "--distributed"]


def _get_forwarded_arg_value(args: list[str], name: str) -> str | None:
    """Return the last value of a forwarded command-line option."""
    value = None
    prefix = f"{name}="
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            value = args[index + 1]
        elif arg.startswith(prefix):
            value = arg[len(prefix) :]
    return value


def _is_skrl_jax_launcher(args_cli: argparse.Namespace, train_args: list[str]) -> bool:
    """Return whether the launch should use skrl's JAX distributed launcher."""
    ml_framework = _get_forwarded_arg_value(train_args, "--ml_framework")
    return args_cli.rl_library == "skrl" and ml_framework == "jax"


def _get_visible_cuda_device_count() -> int | None:
    """Return the number of visible CUDA devices on this node, or ``None`` if undetermined."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        entries = [entry for entry in visible_devices.split(",") if entry.strip()]
        return len(entries)
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    except Exception:
        return None


def _validate_num_gpus_against_visible_devices(parser: argparse.ArgumentParser, args_cli: argparse.Namespace) -> None:
    """Error early when fewer CUDA devices are visible than --num_gpus requests."""
    try:
        requested = int(str(args_cli.nproc_per_node))
    except (TypeError, ValueError):
        return  # torchrun keywords like "gpu"/"cpu"/"auto" are resolved by the launcher itself.
    visible = _get_visible_cuda_device_count()
    if visible is None:
        return
    if visible == 0:
        parser.error(
            f"--num_gpus/--nproc_per_node={requested} was requested but no CUDA devices are visible. "
            "Verify the CUDA installation and CUDA_VISIBLE_DEVICES."
        )
    if requested > visible:
        parser.error(
            f"--num_gpus/--nproc_per_node={requested} exceeds the {visible} CUDA device(s) visible to this "
            "process. Lower --num_gpus or expose more devices via CUDA_VISIBLE_DEVICES."
        )


def _validate_launcher_args(
    parser: argparse.ArgumentParser, args_cli: argparse.Namespace, train_args: list[str]
) -> None:
    """Validate launcher-specific argument combinations."""
    if _is_skrl_jax_launcher(args_cli, train_args):
        unsupported_args = [f"--{name}" for name in SKRL_JAX_TORCHRUN_ONLY_ARGS if getattr(args_cli, name) is not None]
        if unsupported_args:
            parser.error(
                f"{', '.join(unsupported_args)} are torchrun-only options and cannot be used with skrl JAX "
                "multi-GPU training. Use --coordinator_address <host:port> to configure the JAX coordinator."
            )
        try:
            nproc_per_node = int(str(args_cli.nproc_per_node))
        except ValueError:
            parser.error(
                "skrl JAX multi-GPU training requires an integer --num_gpus/--nproc_per_node value; "
                "torchrun values 'gpu', 'cpu', and 'auto' are not supported by skrl.utils.distributed.jax."
            )
        if nproc_per_node < 1:
            parser.error("skrl JAX multi-GPU training requires --num_gpus/--nproc_per_node to be at least 1.")
    elif args_cli.coordinator_address is not None:
        parser.error("--coordinator_address is only supported with --rl_library skrl --ml_framework jax.")

    _validate_num_gpus_against_visible_devices(parser, args_cli)


def _run_distributed_command(command: list[str]) -> int:
    """Run the distributed launcher and forward termination signals to the child process."""
    proc = subprocess.Popen(command)

    def _terminate_child(_signum: int, _frame: FrameType | None) -> None:
        proc.terminate()

    previous_sigterm = signal.signal(signal.SIGTERM, _terminate_child)
    previous_sigint = signal.signal(signal.SIGINT, _terminate_child)
    try:
        return proc.wait()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)


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
        _append_optional_launcher_arg(command, args_cli, name)

    command.extend(
        [
            str(TRAIN_SCRIPT),
            "--rl_library",
            args_cli.rl_library,
            *_with_distributed_arg(train_args),
        ]
    )
    return command


def _build_skrl_jax_command(args_cli: argparse.Namespace, train_args: list[str]) -> list[str]:
    """Build the skrl JAX distributed command for multi-GPU training."""
    command = [
        sys.executable,
        "-m",
        "skrl.utils.distributed.jax",
        "--nproc_per_node",
        str(args_cli.nproc_per_node),
    ]
    for name in ("nnodes", "node_rank", "coordinator_address"):
        _append_optional_launcher_arg(command, args_cli, name)

    command.extend(
        [
            str(TRAIN_SCRIPT),
            "--rl_library",
            args_cli.rl_library,
            *_with_distributed_arg(train_args),
        ]
    )
    return command


def _build_distributed_command(args_cli: argparse.Namespace, train_args: list[str]) -> list[str]:
    """Build the distributed launcher command for multi-GPU training."""
    if _is_skrl_jax_launcher(args_cli, train_args):
        return _build_skrl_jax_command(args_cli, train_args)
    return _build_torchrun_command(args_cli, train_args)


def main(argv: list[str] | None = None) -> int:
    """Launch multi-GPU training with the selected distributed launcher."""
    if argv is None:
        argv = sys.argv[1:]

    args_cli, train_args = _parse_args(argv)
    command = _build_distributed_command(args_cli, train_args)

    if args_cli.dry_run:
        print(shlex.join(command))
        return 0

    print(f"[INFO] Launching distributed training with: {shlex.join(command)}")
    return _run_distributed_command(command)


if __name__ == "__main__":
    raise SystemExit(main())
