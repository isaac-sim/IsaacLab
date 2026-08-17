# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared ``torchrun`` launcher used by Isaac Lab multi-GPU commands.

A multi-GPU command is a supervisor process that builds a distributed launcher
command and runs one worker per rank. :class:`MultiGpuLauncherCfg` describes the
worker and the accepted rendezvous options; :func:`run_multigpu_cli` does the rest.
Both the reinforcement-learning training launcher and the multi-GPU benchmark
launcher are configurations of this module.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from types import FrameType

TORCHRUN_ARGS = (
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
"""Rendezvous and supervision options forwarded to ``torchrun``."""

SKRL_JAX_ARGS = ("nnodes", "node_rank", "coordinator_address")
"""Options accepted by ``skrl.utils.distributed.jax``."""

SKRL_JAX_TORCHRUN_ONLY_ARGS = tuple(name for name in TORCHRUN_ARGS if name not in SKRL_JAX_ARGS)
"""``torchrun`` options that have no skrl JAX equivalent."""

# torchelastic gives its own workers 30 s before it SIGKILLs them, so the graceful window stays
# above that rather than preempting a shutdown that is already making progress.
_POLL_INTERVAL_S = 0.2
_GRACEFUL_SHUTDOWN_S = 40.0
_FORCED_SHUTDOWN_S = 15.0
_STRAGGLER_GRACE_S = 10.0


@dataclass(frozen=True)
class MultiGpuLauncherCfg:
    """Static description of one multi-GPU launcher command.

    Args:
        prog: Program name shown in usage output.
        description: One-line command description shown in help output.
        worker_script: Absolute path of the script each rank runs.
        worker_prefix_args: Arguments placed before the forwarded arguments on every rank,
            used to select the workload the worker should run.
        worker_flags: Arguments appended to every rank's command unless already present.
        rl_libraries: Distributed-capable reinforcement-learning libraries offered by
            ``--rl_library``. An empty tuple omits the option entirely.
        default_rl_library: Library used when ``--rl_library`` is not given.
        supports_jax: Whether ``--rl_library skrl --ml_framework jax`` may use skrl's JAX
            launcher instead of ``torchrun``.
        epilog: Trailing help text.
    """

    prog: str
    description: str
    worker_script: str
    worker_prefix_args: tuple[str, ...] = ()
    worker_flags: tuple[str, ...] = ("--distributed",)
    rl_libraries: tuple[str, ...] = ()
    default_rl_library: str = "rsl_rl"
    supports_jax: bool = True
    epilog: str = "All unrecognized arguments are forwarded to the worker."


def build_parser(cfg: MultiGpuLauncherCfg) -> argparse.ArgumentParser:
    """Build the argument parser for a multi-GPU launcher.

    Args:
        cfg: Launcher configuration.

    Returns:
        Parser accepting the launcher options; every other argument is forwarded to the worker.
    """
    parser = argparse.ArgumentParser(
        prog=cfg.prog,
        description=cfg.description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog=cfg.epilog,
    )
    if cfg.rl_libraries:
        parser.add_argument(
            "--rl_library",
            choices=cfg.rl_libraries,
            default=cfg.default_rl_library,
            help=f"Distributed-capable training library to use. Defaults to {cfg.default_rl_library}.",
        )
    parser.add_argument(
        "--num_gpus",
        "--nproc_per_node",
        dest="nproc_per_node",
        default="gpu",
        help=(
            "Number of worker processes to launch on each node. Accepts an integer or torchrun values "
            "'gpu', 'cpu', and 'auto'. skrl JAX training requires an integer. Defaults to 'gpu'."
        ),
    )
    parser.add_argument("--nnodes", default=None, help="Number of nodes to use for distributed execution.")
    parser.add_argument("--node_rank", default=None, help="Rank of this node in a multi-node job.")
    if cfg.supports_jax:
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
        help=(
            "Show console output from every rank. By default only local rank 0 on each node is shown, because "
            "each rank otherwise repeats the same startup, warning, and model-summary output. Tracebacks from "
            "failing ranks are reported either way."
        ),
    )
    parser.add_argument(
        "--virtual_local_rank",
        action="store_true",
        help=(
            "Give every worker a single GPU as ``cuda:0`` instead of exposing all of them. Works around "
            "OVPhysX <= 0.5.10 selecting the wrong CUDA device when OVRTX shares the process, which hangs "
            "``presets=ovphysx,ovrtx`` runs on more than one GPU (nvbug 6573426). Every worker reports "
            "``LOCAL_RANK=0``, so use the global rank to name per-rank files."
        ),
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print the distributed launcher command without running it."
    )
    return parser


def parse_launcher_args(argv: list[str], cfg: MultiGpuLauncherCfg) -> tuple[argparse.Namespace, list[str]]:
    """Parse launcher options and return the arguments forwarded to every rank.

    Args:
        argv: Command-line arguments excluding the executable name.
        cfg: Launcher configuration.

    Returns:
        Parsed launcher options and the verbatim worker arguments.
    """
    parser = build_parser(cfg)
    args_cli, worker_args = parser.parse_known_args(argv)
    if worker_args[:1] == ["--"]:
        worker_args = worker_args[1:]
    _validate_launcher_args(parser, args_cli, worker_args, cfg)
    return args_cli, worker_args


def build_launch_command(args_cli: argparse.Namespace, worker_args: list[str], cfg: MultiGpuLauncherCfg) -> list[str]:
    """Build the distributed launcher command for a multi-GPU run.

    Args:
        args_cli: Parsed launcher options.
        worker_args: Arguments forwarded to every rank.
        cfg: Launcher configuration.

    Returns:
        Full command line, starting with the Python executable.
    """
    if _is_skrl_jax_launcher(args_cli, worker_args, cfg):
        command = [sys.executable, "-m", "skrl.utils.distributed.jax", "--nproc_per_node", str(args_cli.nproc_per_node)]
        for name in SKRL_JAX_ARGS:
            _append_optional_arg(command, args_cli, name)
        return command + _worker_argv(args_cli, worker_args, cfg)

    command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(args_cli.nproc_per_node)]
    for name in TORCHRUN_ARGS:
        _append_optional_arg(command, args_cli, name)

    # Every rank repeats the same startup, warning, and model-summary output. Failures still surface:
    # torchrun names the failing rank and reports the traceback that ``record`` captures per rank.
    if not args_cli.log_all_ranks and args_cli.local_ranks_filter is None:
        command.extend(("--local_ranks_filter", "0"))

    if args_cli.virtual_local_rank:
        command.append("--virtual_local_rank")

    return command + _worker_argv(args_cli, worker_args, cfg)


def run_multigpu_cli(argv: list[str] | None, cfg: MultiGpuLauncherCfg) -> int:
    """Parse, build, and supervise a multi-GPU launch.

    Args:
        argv: Command-line arguments excluding the executable name. Uses ``sys.argv`` when ``None``.
        cfg: Launcher configuration.

    Returns:
        Process exit code.
    """
    if argv is None:
        argv = sys.argv[1:]

    args_cli, worker_args = parse_launcher_args(argv, cfg)
    command = build_launch_command(args_cli, worker_args, cfg)

    if args_cli.dry_run:
        print(shlex.join(command))
        return 0

    print(f"[INFO] Launching distributed workers with: {shlex.join(command)}")
    return run_launch_command(command)


def run_launch_command(command: list[str]) -> int:
    """Run the distributed launcher, supervising its whole worker group where the platform allows.

    Args:
        command: Full command line to run.

    Returns:
        Process exit code.
    """
    if hasattr(os, "killpg"):
        return _run_supervised(command)
    return _run_terminating_child(command)


def _worker_argv(args_cli: argparse.Namespace, worker_args: list[str], cfg: MultiGpuLauncherCfg) -> list[str]:
    """Return the per-rank command tail: the worker script plus its arguments."""
    library_argv = ["--rl_library", args_cli.rl_library] if cfg.rl_libraries else []
    flags = [flag for flag in cfg.worker_flags if flag not in worker_args]
    return [cfg.worker_script, *cfg.worker_prefix_args, *library_argv, *worker_args, *flags]


def _append_optional_arg(command: list[str], args_cli: argparse.Namespace, name: str) -> None:
    """Append a launcher option to *command* when it was provided."""
    value = getattr(args_cli, name)
    if value is not None:
        command.extend((f"--{name}", str(value)))


def _forwarded_arg_value(args: list[str], name: str) -> str | None:
    """Return the last value of a forwarded command-line option."""
    value = None
    prefix = f"{name}="
    for index, arg in enumerate(args):
        if arg == name and index + 1 < len(args):
            value = args[index + 1]
        elif arg.startswith(prefix):
            value = arg[len(prefix) :]
    return value


def _is_skrl_jax_launcher(args_cli: argparse.Namespace, worker_args: list[str], cfg: MultiGpuLauncherCfg) -> bool:
    """Return whether the launch should use skrl's JAX distributed launcher."""
    if not cfg.rl_libraries or args_cli.rl_library != "skrl":
        return False
    return _forwarded_arg_value(worker_args, "--ml_framework") == "jax"


def _visible_cuda_device_count() -> int | None:
    """Return the number of visible CUDA devices on this node, or ``None`` if undetermined."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        return len([entry for entry in visible_devices.split(",") if entry.strip()])
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


def _validate_num_gpus(parser: argparse.ArgumentParser, args_cli: argparse.Namespace) -> None:
    """Error early when fewer CUDA devices are visible than ``--num_gpus`` requests."""
    try:
        requested = int(str(args_cli.nproc_per_node))
    except (TypeError, ValueError):
        return  # torchrun keywords like "gpu"/"cpu"/"auto" are resolved by the launcher itself.
    visible = _visible_cuda_device_count()
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
    worker_args: list[str],
    cfg: MultiGpuLauncherCfg,
) -> None:
    """Validate launcher-specific argument combinations."""
    wants_jax = _is_skrl_jax_launcher(args_cli, worker_args, cfg)
    if wants_jax and not cfg.supports_jax:
        parser.error(f"{cfg.prog} does not support skrl JAX; use --ml_framework torch.")
    if wants_jax:
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
    elif getattr(args_cli, "coordinator_address", None) is not None:
        parser.error("--coordinator_address is only supported with --rl_library skrl --ml_framework jax.")

    _validate_num_gpus(parser, args_cli)


def _signal_group(pgid: int, sig: int) -> None:
    """Signal every process in the worker group, ignoring a group that already exited."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


def _group_is_alive(pgid: int) -> bool:
    """Return whether any process remains in the worker group."""
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _reap_group(pgid: int) -> None:
    """Kill workers that outlived the launcher.

    A worker wedged in a native CUDA, NCCL, or renderer call never returns to Python to observe a
    signal, so it survives torchrun and keeps holding GPU memory.
    """
    deadline = time.monotonic() + _STRAGGLER_GRACE_S
    while _group_is_alive(pgid):
        if time.monotonic() >= deadline:
            print("[WARNING] Killing distributed workers that outlived the launcher.", file=sys.stderr)
            _signal_group(pgid, signal.SIGKILL)
            return
        time.sleep(_POLL_INTERVAL_S)


def _run_supervised(command: list[str]) -> int:
    """Run the launcher in its own process group and tear that group down however the run ends.

    Sharing this process's group lets the terminal deliver Ctrl-C to torchrun and every worker at the
    same moment this handler forwards a signal of its own. The extra signal lands inside
    torchelastic's shutdown handler, which then aborts before reaping the workers it spawned.
    """
    proc = subprocess.Popen(command, start_new_session=True)
    pgid = os.getpgid(proc.pid)
    deadlines: dict[str, float] = {}

    def _forward(signum: int, _frame: FrameType | None) -> None:
        if deadlines:
            print("[WARNING] Second interrupt received; killing distributed workers now.", file=sys.stderr)
            _signal_group(pgid, signal.SIGKILL)
            return
        now = time.monotonic()
        deadlines.update(terminate=now + _GRACEFUL_SHUTDOWN_S, kill=now + _GRACEFUL_SHUTDOWN_S + _FORCED_SHUTDOWN_S)
        print(
            f"\n[INFO] Received {signal.Signals(signum).name}; shutting down distributed workers."
            " Press Ctrl-C again to kill them immediately.",
            file=sys.stderr,
        )
        _signal_group(pgid, signum)

    previous = [(sig, signal.signal(sig, _forward)) for sig in (signal.SIGINT, signal.SIGTERM)]
    try:
        while True:
            try:
                return proc.wait(timeout=_POLL_INTERVAL_S)
            except subprocess.TimeoutExpired:
                pass
            if not deadlines:
                continue
            now = time.monotonic()
            if now >= deadlines["kill"]:
                _signal_group(pgid, signal.SIGKILL)
                deadlines["kill"] = now + _FORCED_SHUTDOWN_S
            elif now >= deadlines["terminate"]:
                _signal_group(pgid, signal.SIGTERM)
                deadlines["terminate"] = now + _FORCED_SHUTDOWN_S
    finally:
        for sig, handler in previous:
            signal.signal(sig, handler)
        _reap_group(pgid)


def _run_terminating_child(command: list[str]) -> int:
    """Run the launcher and forward termination to it, for platforms without process groups."""
    proc = subprocess.Popen(command)

    def _terminate(_signum: int, _frame: FrameType | None) -> None:
        proc.terminate()

    previous = [(sig, signal.signal(sig, _terminate)) for sig in (signal.SIGINT, signal.SIGTERM)]
    try:
        return proc.wait()
    finally:
        for sig, handler in previous:
            signal.signal(sig, handler)
