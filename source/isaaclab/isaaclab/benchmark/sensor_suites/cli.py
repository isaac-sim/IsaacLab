# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line arguments shared by sensor micro-benchmarks."""

import argparse
from collections.abc import Sequence

from .._cli import parse_non_negative_int, parse_positive_int


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
