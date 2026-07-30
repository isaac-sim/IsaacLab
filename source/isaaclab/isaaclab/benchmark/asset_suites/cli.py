# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI bridge used by per-backend asset benchmark scripts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .._cli import parse_non_negative_int, parse_positive_int
from ..method_benchmark import MethodBenchmarkRunnerConfig
from .dispatch import get_asset_benchmark_adapter
from .runner import run_asset_benchmark
from .types import AssetBenchmarkRequest


def run_asset_benchmark_cli(
    physics: str,
    component: str,
    argv: Sequence[str] | None = None,
    *,
    include_app_launcher_args: bool = True,
) -> tuple[Path, ...]:
    """Parse an asset script's arguments and run both benchmark phases."""
    selector_parser = argparse.ArgumentParser(add_help=False)
    selector_parser.add_argument("--physics_variant", default=physics, help="Exact physics variant")
    selector_args, _ = selector_parser.parse_known_args(argv)
    adapter = get_asset_benchmark_adapter(selector_args.physics_variant, component)
    parser = argparse.ArgumentParser(
        description=f"Benchmark {component} methods and data ({selector_args.physics_variant} backend).",
        parents=[selector_parser],
    )
    parser.add_argument("--num_iterations", type=parse_positive_int, default=1000, help="Number of iterations")
    parser.add_argument("--warmup_steps", type=parse_non_negative_int, default=10, help="Number of warmup steps")
    parser.add_argument("--num_instances", type=parse_positive_int, default=4096, help="Number of instances")
    parser.add_argument(
        "--num_bodies", type=parse_positive_int, default=adapter.default_num_bodies, help="Number of bodies"
    )
    parser.add_argument(
        "--num_joints", type=parse_non_negative_int, default=adapter.default_num_joints, help="Number of joints"
    )
    parser.add_argument("--mode", type=str, default="all", help="Benchmark input mode")
    parser.add_argument(
        "--output_path",
        "--output_dir",
        dest="output_path",
        type=Path,
        default=Path("."),
        help="Output directory for both artifacts",
    )
    parser.add_argument(
        "--benchmark_formatter",
        "--backend",
        dest="benchmark_formatter",
        default="json",
        choices=("json", "osmo", "omniperf", "summary"),
        help="Metrics formatter",
    )
    parser.add_argument("--no_shape_checks", action="store_true", help="Disable shape and dtype assertions")
    if include_app_launcher_args:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    else:
        parser.add_argument("--device", type=str, default="cuda:0", help="Tensor device")
    args = parser.parse_args(argv)
    config = MethodBenchmarkRunnerConfig(
        num_iterations=args.num_iterations,
        warmup_steps=args.warmup_steps,
        num_instances=args.num_instances,
        num_bodies=args.num_bodies,
        num_joints=args.num_joints,
        device=args.device,
        mode=args.mode,
    )
    request = AssetBenchmarkRequest(
        config=config,
        physics_variant=args.physics_variant,
        formatter_type=args.benchmark_formatter,
        output_path=args.output_path,
        check_shapes=not args.no_shape_checks,
        launcher_args=args if include_app_launcher_args else None,
    )
    return run_asset_benchmark(request, adapter)
