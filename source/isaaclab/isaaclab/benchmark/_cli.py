# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line argument utilities for benchmark scripts."""

from __future__ import annotations

import argparse


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


def validate_warmup_steps(warmup_steps: int, available_steps: int) -> None:
    """Validate that training warm-up leaves at least one measured environment step."""
    if warmup_steps >= available_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be less than resolved training environment steps ({available_steps})"
        )


def add_benchmark_output_args(parser: argparse.ArgumentParser, *, include_learning_args: bool = False) -> None:
    """Register the output and environment-step timing arguments shared by the RL benchmark adapters.

    Args:
        parser: Parser receiving the arguments.
        include_learning_args: Whether to also register the learning-curve arguments used by training benchmarks.
    """
    parser.add_argument("--output_path", type=str, default=".", help="Directory to write the output JSON.")
    parser.add_argument(
        "--measure_sync_step",
        action="store_true",
        help="Measure a serialized synchronized simulation and outside-simulation step breakdown.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=parse_non_negative_int,
        default=1,
        help="Exclude the first N env.step() calls from environment-step timing. Default 1 removes cold start.",
    )
    parser.add_argument(
        "--benchmark_formatter",
        type=str,
        default="schema",
        help=(
            "Output format(s): comma-separated list of 'schema' (default, the typed benchmark bundle),"
            " 'omniperf', 'osmo', 'json', 'summary'"
            " Example: 'schema,omniperf'."
        ),
    )
    if not include_learning_args:
        return
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.1,
        help="EMA smoothing factor for learning curves (higher = more recent weight).",
    )
    parser.add_argument(
        "--no_series",
        action="store_true",
        default=False,
        help="Omit per-iteration series data from the bundle to reduce file size.",
    )


def add_play_args(
    parser: argparse.ArgumentParser, argv: list[str], *, agent_default: str | None, agent_help: str
) -> None:
    """Register the rollout arguments shared by the RL play benchmark adapters.

    Args:
        parser: Parser receiving the arguments.
        argv: Raw command-line arguments, used to relax ``--task`` when help is requested.
        agent_default: Default agent configuration entry point.
        agent_help: Help text of the ``--agent`` argument.
    """
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
    parser.add_argument("--video_length", type=int, default=None, help="Recorded video length in environment steps.")
    help_requested = "-h" in argv or "--help" in argv
    parser.add_argument("--task", type=str, required=not help_requested, help="Gym task id to benchmark.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument(
        "--num_steps", type=parse_positive_int, default=100, help="Number of inference steps to benchmark."
    )
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Local or Nucleus checkpoint path to roll out; falls back to the published checkpoint when omitted.",
    )
    parser.add_argument("--agent", type=str, default=agent_default, help=agent_help)
