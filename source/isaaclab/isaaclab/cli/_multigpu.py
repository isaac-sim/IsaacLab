# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private helpers shared by Isaac Lab multi-GPU launcher commands."""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

_TORCHRUN_OPTION_NAMES = (
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
)
_SKRL_JAX_TORCHRUN_ONLY_ARGS = _TORCHRUN_OPTION_NAMES[2:]

ForwardedArgsValidator = Callable[[argparse.ArgumentParser, argparse.Namespace, list[str]], None]


@dataclass(frozen=True)
class MultiGpuLauncherSpec:
    """Configuration for a private multi-GPU launcher."""

    target_script: Path
    description: str
    supported_libraries: tuple[str, ...]
    default_library: str = "rsl_rl"
    allow_skrl_jax: bool = True
    forwarded_args: tuple[str, ...] = ("--distributed",)
    validate_forwarded_args: ForwardedArgsValidator | None = None
    epilog: str = "All unrecognized arguments are forwarded to the selected training library."


def parse_multigpu_args(
    argv: list[str], spec: MultiGpuLauncherSpec
) -> tuple[argparse.Namespace, list[str]]:
    """Parse launcher arguments and return arguments forwarded to training."""
    parser = argparse.ArgumentParser(
        description=spec.description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog=spec.epilog,
    )
    parser.add_argument(
        "--rl_library",
        choices=spec.supported_libraries,
        default=spec.default_library,
        help=f"Distributed-capable training library to use. Defaults to {spec.default_library}.",
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
        "--log_all_ranks",
        action="store_true",
        help="Show console output from every rank instead of only local rank 0.",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print the distributed launcher command without running it."
    )

    args_cli, forwarded_args = parser.parse_known_args(argv)
    if forwarded_args[:1] == ["--"]:
        forwarded_args = forwarded_args[1:]
    _validate_launcher_args(parser, args_cli, forwarded_args, spec)
    if spec.validate_forwarded_args is not None:
        spec.validate_forwarded_args(parser, args_cli, forwarded_args)
    return args_cli, forwarded_args


def _get_forwarded_arg_value(args: list[str], name: str) -> str | None:
    value = None
    prefix = f"{name}="
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            value = args[index + 1]
        elif arg.startswith(prefix):
            value = arg[len(prefix) :]
    return value


def _is_skrl_jax(args_cli: argparse.Namespace, forwarded_args: list[str]) -> bool:
    return args_cli.rl_library == "skrl" and _get_forwarded_arg_value(forwarded_args, "--ml_framework") == "jax"


def _get_visible_cuda_device_count() -> int | None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        return len([entry for entry in visible_devices.split(",") if entry.strip()])
    try:
        import torch
    except ImportError:
        return None
    try:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return None


def _validate_num_gpus(parser: argparse.ArgumentParser, args_cli: argparse.Namespace) -> None:
    try:
        requested = int(str(args_cli.nproc_per_node))
    except (TypeError, ValueError):
        return
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
    parser: argparse.ArgumentParser,
    args_cli: argparse.Namespace,
    forwarded_args: list[str],
    spec: MultiGpuLauncherSpec,
) -> None:
    is_jax = _is_skrl_jax(args_cli, forwarded_args)
    if is_jax and not spec.allow_skrl_jax:
        parser.error("skrl JAX is not supported by this multi-GPU launcher.")
    if is_jax:
        unsupported = [f"--{name}" for name in _SKRL_JAX_TORCHRUN_ONLY_ARGS if getattr(args_cli, name) is not None]
        if unsupported:
            parser.error(
                f"{', '.join(unsupported)} are torchrun-only options and cannot be used with skrl JAX "
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
    _validate_num_gpus(parser, args_cli)


def _append_optional(command: list[str], args_cli: argparse.Namespace, name: str) -> None:
    value = getattr(args_cli, name)
    if value is not None:
        command.extend((f"--{name}", str(value)))


def _with_required_forwarded_args(args: list[str], required: tuple[str, ...]) -> list[str]:
    result = list(args)
    for option in required:
        if option not in result:
            result.append(option)
    return result


def _build_torchrun_command(
    args_cli: argparse.Namespace, forwarded_args: list[str], spec: MultiGpuLauncherSpec
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(args_cli.nproc_per_node),
    ]
    for name in _TORCHRUN_OPTION_NAMES:
        _append_optional(command, args_cli, name)
    if not args_cli.log_all_ranks and args_cli.local_ranks_filter is None:
        command.extend(("--local_ranks_filter", "0"))
    command.extend(
        (
            str(spec.target_script),
            "--rl_library",
            args_cli.rl_library,
            *_with_required_forwarded_args(forwarded_args, spec.forwarded_args),
        )
    )
    return command


def _build_skrl_jax_command(
    args_cli: argparse.Namespace, forwarded_args: list[str], spec: MultiGpuLauncherSpec
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "skrl.utils.distributed.jax",
        "--nproc_per_node",
        str(args_cli.nproc_per_node),
    ]
    for name in ("nnodes", "node_rank", "coordinator_address"):
        _append_optional(command, args_cli, name)
    command.extend(
        (
            str(spec.target_script),
            "--rl_library",
            args_cli.rl_library,
            *_with_required_forwarded_args(forwarded_args, spec.forwarded_args),
        )
    )
    return command


def build_distributed_command(
    args_cli: argparse.Namespace, forwarded_args: list[str], spec: MultiGpuLauncherSpec
) -> list[str]:
    """Build the configured distributed launcher command."""
    if _is_skrl_jax(args_cli, forwarded_args):
        return _build_skrl_jax_command(args_cli, forwarded_args, spec)
    return _build_torchrun_command(args_cli, forwarded_args, spec)


def _run_distributed_command(command: list[str]) -> int:
    proc = subprocess.Popen(command)

    def terminate_child(_signum: int, _frame: FrameType | None) -> None:
        proc.terminate()

    previous_sigterm = signal.signal(signal.SIGTERM, terminate_child)
    previous_sigint = signal.signal(signal.SIGINT, terminate_child)
    try:
        return proc.wait()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)


def run_multigpu(argv: list[str] | None, spec: MultiGpuLauncherSpec) -> int:
    """Parse, build, and run a configured multi-GPU command."""
    if argv is None:
        argv = sys.argv[1:]
    args_cli, forwarded_args = parse_multigpu_args(argv, spec)
    command = build_distributed_command(args_cli, forwarded_args, spec)
    if args_cli.dry_run:
        print(shlex.join(command))
        return 0
    print(f"[INFO] Launching distributed training with: {shlex.join(command)}")
    return _run_distributed_command(command)
