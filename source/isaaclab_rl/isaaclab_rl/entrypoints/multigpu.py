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
import contextlib  # noqa: E402
import os  # noqa: E402
import shlex  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import FrameType  # noqa: E402

from torch.distributed.elastic.multiprocessing.errors import record  # noqa: E402

# Absolute imports: each rank runs this file as a script, where relative imports have no package.
from isaaclab_rl.entrypoints.api import MULTI_GPU_BACKENDS  # noqa: E402
from isaaclab_rl.entrypoints.dispatch import run_train_cli  # noqa: E402

WORKER_SCRIPT = str(Path(__file__).resolve())

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
SKRL_JAX_ARGS = ("nnodes", "node_rank", "coordinator_address")
SKRL_JAX_TORCHRUN_ONLY_ARGS = tuple(name for name in TORCHRUN_ARGS if name not in SKRL_JAX_ARGS)

# torchelastic gives its own workers 30 s before it SIGKILLs them, so the graceful window stays
# above that rather than preempting a shutdown that is already making progress.
_POLL_INTERVAL_S = 0.2
_GRACEFUL_SHUTDOWN_S = 40.0
_FORCED_SHUTDOWN_S = 15.0
_STRAGGLER_GRACE_S = 10.0


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse multi-GPU launcher arguments and return forwarded training arguments."""
    parser = argparse.ArgumentParser(
        prog="train_multigpu",
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
        choices=MULTI_GPU_BACKENDS,
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
        "--log_all_ranks",
        action="store_true",
        help=(
            "Show console output from every rank. By default only local rank 0 on each node is shown, because "
            "each rank otherwise repeats the same startup, warning, and model-summary output. Tracebacks from "
            "failing ranks are reported either way."
        ),
    )
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


def _worker_argv(args_cli: argparse.Namespace, train_args: list[str]) -> list[str]:
    """Return the per-rank command tail: this file plus its training arguments."""
    return [WORKER_SCRIPT, "--rl_library", args_cli.rl_library, *_with_distributed_arg(train_args)]


def _build_torchrun_command(args_cli: argparse.Namespace, train_args: list[str]) -> list[str]:
    """Build the torchrun command for multi-GPU training."""
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(args_cli.nproc_per_node)]
    for name in TORCHRUN_ARGS:
        _append_optional_launcher_arg(command, args_cli, name)

    # Every rank repeats the same startup, warning, and model-summary output. Failures still surface:
    # torchrun names the failing rank and reports the traceback that ``record`` captures per rank.
    if not args_cli.log_all_ranks and args_cli.local_ranks_filter is None:
        command.extend(["--local_ranks_filter", "0"])

    return command + _worker_argv(args_cli, train_args)


def _build_skrl_jax_command(args_cli: argparse.Namespace, train_args: list[str]) -> list[str]:
    """Build the skrl JAX distributed command for multi-GPU training."""
    command = [sys.executable, "-m", "skrl.utils.distributed.jax", "--nproc_per_node", str(args_cli.nproc_per_node)]
    for name in SKRL_JAX_ARGS:
        _append_optional_launcher_arg(command, args_cli, name)

    return command + _worker_argv(args_cli, train_args)


def _build_distributed_command(args_cli: argparse.Namespace, train_args: list[str]) -> list[str]:
    """Build the distributed launcher command for multi-GPU training."""
    if _is_skrl_jax_launcher(args_cli, train_args):
        return _build_skrl_jax_command(args_cli, train_args)
    return _build_torchrun_command(args_cli, train_args)


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


def _run_distributed_command(command: list[str]) -> int:
    """Run the distributed launcher, supervising its whole worker group where the platform allows."""
    if hasattr(os, "killpg"):
        return _run_supervised(command)
    return _run_terminating_child(command)


def run_train_multigpu_cli(argv: list[str] | None = None) -> int:
    """Launch multi-GPU training with the selected distributed launcher.

    Args:
        argv: Command-line arguments excluding the executable name.

    Returns:
        Process exit code.
    """
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
    # Reached once per rank. ``record`` reports the failing rank's traceback to torchrun, which is
    # otherwise lost when console output is filtered to rank 0.
    raise SystemExit(record(run_train_cli)())
