# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-width console summaries printed when a benchmark workflow finishes.

These are overviews meant to be read in a terminal. The JSON output written next to
them holds the full result; nothing here is a substitute for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from isaaclab.benchmark.schema import MeanStd, RuntimeBundle, StartupBundle, TrainingBundle

_WIDTH = 64
"""Total width of the report block."""

_LABEL_WIDTH = 26
"""Width of the label column, so rows line up across workflows."""

_BAR_WIDTH = 20
"""Width of the longest bar in a proportional row."""


def print_startup_report(bundle: StartupBundle, output_paths: tuple[Path, ...], detail: dict[str, float]) -> None:
    """Print the per-phase wall-clock breakdown of a startup benchmark.

    Args:
        bundle: Completed startup bundle.
        output_paths: Files written by the benchmark formatters.
        detail: Elapsed seconds of the timers that ran during ``env_creation``.
    """
    if not bundle.phases:
        return
    # Phases are stored in the order they ran, so no re-ordering is needed here.
    total_s = sum(phase.total_time_s for phase in bundle.phases.values())
    longest_s = max(phase.total_time_s for phase in bundle.phases.values())

    rows: list[tuple[str, str] | None] = []
    for name, phase in bundle.phases.items():
        seconds = phase.total_time_s
        share = seconds / total_s if total_s > 0 else 0.0
        bar = "#" * round(_BAR_WIDTH * seconds / longest_s) if longest_s > 0 else ""
        rows.append((name, f"{seconds:7.2f} s  {share:5.1%}  {bar}"))
        if name != "env_creation":
            continue
        # Nested timers: they overlap each other and the phase, so no share is shown.
        for timer_name, timer_s in sorted(detail.items(), key=lambda item: -item[1]):
            rows.append((f"  {timer_name}", f"{timer_s:7.2f} s"))
    rows.append(None)
    rows.append(("total", f"{total_s:7.2f} s"))

    _print_report(_title("Startup summary", bundle), rows, _footer(output_paths))


def print_runtime_report(bundle: RuntimeBundle, output_paths: tuple[Path, ...]) -> None:
    """Print the throughput and resource overview of a runtime benchmark.

    Args:
        bundle: Completed runtime bundle.
        output_paths: Files written by the benchmark formatters.
    """
    _print_report(_title("Runtime summary", bundle), _runtime_rows(bundle), _footer(output_paths))


def print_training_report(bundle: TrainingBundle, output_paths: tuple[Path, ...]) -> None:
    """Print the throughput, resource, and learning overview of a training benchmark.

    Args:
        bundle: Completed training bundle.
        output_paths: Files written by the benchmark formatters.
    """
    learning = bundle.learning
    rows = _runtime_rows(bundle)
    rows.append(None)
    rows.append(("reward", _curve(learning.reward)))
    rows.append(("episode_length", _curve(learning.ep_length)))
    if learning.success_rate is not None:
        rows.append(("success_rate", _curve(learning.success_rate)))

    footer = _footer(output_paths)
    if bundle.checkpoint_path:
        footer.append(f"checkpoint: {bundle.checkpoint_path}")
    if bundle.video_path:
        footer.append(f"video: {bundle.video_path}")

    _print_report(_title("Training summary", bundle), rows, footer)


def _print_report(title: str, rows: list[tuple[str, str] | None], footer: list[str]) -> None:
    """Print a titled block of label/value rows, where a ``None`` row draws a separator."""
    print()
    print("=" * _WIDTH)
    print(f" {title}")
    print("=" * _WIDTH)
    for row in rows:
        if row is None:
            print("-" * _WIDTH)
            continue
        label, value = row
        print(f" {label:<{_LABEL_WIDTH}}{value}".rstrip())
    print("-" * _WIDTH)
    for line in footer:
        print(f" {line}")
    print("=" * _WIDTH)


def _title(workflow: str, bundle: StartupBundle | RuntimeBundle | TrainingBundle) -> str:
    """Build the report title from the run identity."""
    run = bundle.run
    envs = f"{run.num_envs} envs" if run.num_envs is not None else "default envs"
    framework = f", {run.framework}" if run.framework else ""
    return f"{workflow}: {run.task} ({envs}, {run.config.physics_backend}{framework})"


def _footer(output_paths: tuple[Path, ...]) -> list[str]:
    """Build the footer lines listing the written result files."""
    return [f"report: {path}" for path in output_paths]


def _runtime_rows(bundle: RuntimeBundle | TrainingBundle) -> list[tuple[str, str] | None]:
    """Build the throughput and resource rows shared by runtime and training reports."""
    runtime = bundle.runtime
    resources = bundle.resources
    startup = runtime.startup_time_s
    return [
        ("total_fps", _metric(runtime.total_fps, "steps/s", decimals=0)),
        ("collection_fps", _metric(runtime.collection_fps, "steps/s", decimals=0)),
        ("iteration_time", _metric(runtime.iteration_time_s, "s", decimals=3)),
        ("iterations", f"{runtime.iterations_completed:,}"),
        ("steps_per_iteration", f"{runtime.steps_per_iteration:,}"),
        ("total_wall_time", f"{runtime.total_wall_time_s:,.2f} s"),
        None,
        ("startup (launch/env/step)", f"{startup.app_launch:.2f} / {startup.env_creation:.2f} /"
         f" {startup.first_step:.2f} s"),
        ("gpu_util", _metric(resources.gpu_util_pct, "%", decimals=1)),
        ("gpu_mem", _metric(resources.gpu_mem_gb, "GB", decimals=2)),
    ]


def _metric(value: MeanStd, unit: str, decimals: int = 2) -> str:
    """Format a mean/std/peak aggregate, omitting the parts that carry no information."""
    text = f"{value.mean:,.{decimals}f} {unit}".strip()
    if value.std:
        text += f"  std {value.std:,.{decimals}f}"
    if value.peak is not None:
        text += f"  peak {value.peak:,.{decimals}f}"
    return text


def _curve(curve) -> str:
    """Format the final raw and EMA-smoothed value of a learning curve."""
    return f"{curve.final_raw:,.3f}  ema {curve.final_ema:,.3f}"
