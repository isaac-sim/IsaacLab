# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line argument utilities for benchmark scripts."""

import argparse
from collections.abc import Sequence


def parse_non_negative_int(value: str) -> int:
    """Parse a non-negative integer command-line argument."""
    parsed_value = int(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed_value


def parse_positive_int(value: str) -> int:
    """Parse a positive integer command-line argument."""
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed_value


def add_sensor_benchmark_args(
    parser: argparse.ArgumentParser,
    *,
    physics_variants: Sequence[str],
    default_physics_variant: str,
    add_device: bool,
) -> None:
    """Add common workload and output arguments for sensor benchmarks.

    Args:
        parser: Parser receiving the arguments.
        physics_variants: Supported physics variants.
        default_physics_variant: Physics variant selected by default.
        add_device: Whether to add the device argument.
    """
    parser.add_argument(
        "--physics_variant",
        choices=physics_variants,
        default=default_physics_variant,
        help="Physics implementation to benchmark.",
    )
    parser.add_argument("--num_envs", type=parse_positive_int, default=4096, help="Number of environments.")
    parser.add_argument("--num_steps", type=parse_positive_int, default=500, help="Number of measured steps.")
    parser.add_argument(
        "--warmup_steps",
        type=parse_non_negative_int,
        default=50,
        help="Number of unmeasured warm-up steps.",
    )
    parser.add_argument("--label", default="current", help="Label stored in benchmark metadata.")
    parser.add_argument("--output_path", default=".", help="Directory for benchmark results.")
    parser.add_argument(
        "--benchmark_formatter",
        choices=("json", "osmo", "omniperf", "summary"),
        default="summary",
        help="Benchmark result formatter.",
    )
    if add_device:
        parser.add_argument("--device", default="cuda:0", help="Device used by the benchmark.")


def validate_warmup_steps(warmup_steps: int, available_steps: int) -> None:
    """Validate that training warm-up leaves at least one measured environment step."""
    if warmup_steps >= available_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be less than resolved training environment steps "
            f"({available_steps})"
        )
