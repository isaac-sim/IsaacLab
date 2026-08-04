# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line entrypoint for multi-GPU RL training benchmarks."""

from __future__ import annotations

import argparse

from isaaclab.cli._multigpu import (
    MultiGpuLauncherSpec,
    build_distributed_command,
    parse_multigpu_args,
    run_multigpu,
)
from isaaclab.cli.utils import ISAACLAB_ROOT


def _option_value(args: list[str], name: str) -> str | None:
    """Return an option value while treating Kit arguments as opaque."""
    value = None
    index = 0
    prefix = f"{name}="
    while index < len(args):
        token = args[index]
        if token == "--kit_args":
            index += 2
            continue
        if token.startswith("--kit_args="):
            index += 1
            continue
        if token == name and index + 1 < len(args):
            value = args[index + 1]
            index += 2
            continue
        if token.startswith(prefix):
            value = token[len(prefix) :]
        index += 1
    return value


def _has_option(args: list[str], name: str) -> bool:
    """Return whether an option occurs outside an opaque Kit argument value."""
    sentinel = "__isaaclab_present__"
    rewritten = [f"{name}={sentinel}" if token == name else token for token in args]
    return _option_value(rewritten, name) is not None


def _validate_benchmark_args(
    parser: argparse.ArgumentParser, _args_cli: argparse.Namespace, forwarded_args: list[str]
) -> None:
    """Reject benchmark features that are unsafe across worker ranks."""
    if _has_option(forwarded_args, "--video"):
        parser.error("Video recording is not supported by multi-GPU training benchmarks.")
    capture_value = _option_value(forwarded_args, "--capture_env_sensors")
    if capture_value is not None:
        try:
            capture_count = int(capture_value)
        except ValueError:
            parser.error("--capture_env_sensors must be an integer.")
        if capture_count > 0:
            parser.error("Environment sensor capture is not supported by multi-GPU training benchmarks.")
    if _has_option(forwarded_args, "--check_success"):
        parser.error("Success-based early stopping is not supported by multi-GPU training benchmarks.")


LAUNCHER_SPEC = MultiGpuLauncherSpec(
    target_script=ISAACLAB_ROOT / "scripts" / "benchmarks" / "training.py",
    description="Launch a multi-GPU Isaac Lab training benchmark.",
    supported_libraries=("rl_games", "rsl_rl", "skrl"),
    allow_skrl_jax=False,
    forwarded_args=("--distributed", "--benchmark_multigpu"),
    validate_forwarded_args=_validate_benchmark_args,
)


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse launcher arguments and return benchmark arguments."""
    return parse_multigpu_args(argv, LAUNCHER_SPEC)


def _build_distributed_command(args_cli: argparse.Namespace, benchmark_args: list[str]) -> list[str]:
    """Build the distributed training benchmark command."""
    return build_distributed_command(args_cli, benchmark_args, LAUNCHER_SPEC)


def main(argv: list[str] | None = None) -> int:
    """Launch the multi-GPU training benchmark."""
    return run_multigpu(argv, LAUNCHER_SPEC)


def run(argv: list[str] | None = None) -> int:
    """Launch the multi-GPU training benchmark from benchmark dispatch."""
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
