# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Result assembly shared by the benchmark workflow entrypoints.

Imports of the benchmark runtime stay inside the helpers so the entrypoints can still attribute
them to their ``python_imports`` startup phase.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..benchmark_core import BaseIsaacLabBenchmark
    from ..schema import Hardware, Resources, RunConfig, RunIdentity, Runtime, StartupTime, Versions
    from ..stepping import EnvironmentStepTimingRecorder


def step_timing_metadata(args: argparse.Namespace) -> list[dict[str, object]]:
    """Return the workflow metadata entries describing the environment-step timing mode."""
    return [
        {
            "name": "environment_step_measurement_mode",
            "data": "serialized_synchronized" if args.measure_sync_step else "host_return",
        },
        {"name": "environment_step_warmup_steps", "data": args.warmup_steps},
    ]


def create_benchmark(
    benchmark_name: str, args: argparse.Namespace, *, output_prefix: str, metadata: list[dict[str, object]]
) -> BaseIsaacLabBenchmark:
    """Create the benchmark that collects a workflow's measurements.

    Args:
        benchmark_name: Workflow name stored in the outputs.
        args: Parsed arguments carrying ``benchmark_formatter`` and ``output_path``.
        output_prefix: Prefix of the written result files.
        metadata: Workflow metadata entries with ``name`` and ``data`` keys.

    Returns:
        Benchmark with recorders enabled and frametime recorders for the flat formatters.
    """
    from ..benchmark_core import BaseIsaacLabBenchmark

    formatter_types = [value.strip() for value in args.benchmark_formatter.split(",") if value.strip()] or ["omniperf"]
    return BaseIsaacLabBenchmark(
        benchmark_name=benchmark_name,
        formatter_type=formatter_types,
        output_path=args.output_path,
        use_recorders=True,
        frametime_recorders=any(t in ("summary", "omniperf") for t in formatter_types),
        output_prefix=output_prefix,
        workflow_metadata={"metadata": metadata},
    )


def capture_snapshots(benchmark: BaseIsaacLabBenchmark) -> tuple[Versions, Hardware, Resources]:
    """Read the version, hardware, and resource snapshots off a finished benchmark."""
    from .. import capture

    return capture.capture_versions(benchmark), capture.capture_hardware(benchmark), capture.capture_resources(benchmark)


def finish_run_identity(
    *,
    framework: str | None,
    config: RunConfig,
    task: str,
    seed: int,
    start_utc: str,
    num_envs: int,
    max_iterations: int | None = None,
) -> RunIdentity:
    """Close a run at the current time and assemble its identity.

    Args:
        framework: RL library, or ``None`` for non-learning runs.
        config: Physics and rendering configuration of the run.
        task: Gym task id.
        seed: Environment/agent seed.
        start_utc: ISO-8601 UTC start timestamp.
        num_envs: Number of parallel environments.
        max_iterations: Training iteration budget, or ``None`` for non-training runs.

    Returns:
        Run identity whose end time is now.
    """
    from .. import builders, capture

    end_utc = capture.now_utc_iso()
    stamp = end_utc.translate(str.maketrans("", "", ":-"))[:15]
    return builders.build_run_identity(
        run_id=capture.synth_run_id(framework, config.physics_backend, task, seed, stamp),
        framework=framework,
        config=config,
        task=task,
        seed=seed,
        start_utc=start_utc,
        end_utc=end_utc,
        num_envs=num_envs,
        max_iterations=max_iterations,
    )


def build_play_runtime(
    *,
    startup: StartupTime,
    step_times_s: Sequence[float],
    num_envs: int,
    warmup_steps: int,
    timer: EnvironmentStepTimingRecorder,
) -> Runtime:
    """Aggregate a play rollout, where every environment step is one iteration.

    Args:
        startup: Startup phase durations [s].
        step_times_s: Wall time of every rollout step including warm-up [s].
        num_envs: Number of parallel environments.
        warmup_steps: Leading steps excluded from the aggregates.
        timer: Environment-step timing recorder that wrapped the rollout.

    Returns:
        Aggregated runtime metrics.
    """
    from .. import builders

    measured = list(step_times_s[warmup_steps:])
    fps = [num_envs / t for t in measured if t > 0]
    return builders.build_runtime(
        startup_time_s=startup,
        iteration_times_s=measured,
        collection_fps=fps,
        total_fps=fps,
        steps_per_iteration=num_envs,
        frames_per_environment_step=num_envs,
        environment_step_warmup_steps=warmup_steps,
        environment_step_times_s=timer.step_times_s,
        simulation_step_times_s=timer.simulation_step_times_s,
        simulation_step_calls=timer.simulation_step_calls,
    )
