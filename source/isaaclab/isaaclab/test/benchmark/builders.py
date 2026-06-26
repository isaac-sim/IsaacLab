# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure assembly functions for benchmark bundle dataclasses.

Converts already-extracted per-iteration series and scalar measurements into
the frozen :mod:`~isaaclab.test.benchmark.schema` dataclasses that are then
serialised by :func:`~isaaclab.test.benchmark.serialize.write_bundle_file`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from isaaclab.test.benchmark.metrics import ema, mean_std_peak
from isaaclab.test.benchmark.schema import (
    Hardware,
    Learning,
    LearningCurve,
    Resources,
    RunConfig,
    RunIdentity,
    Runtime,
    RuntimeBundle,
    StartupBundle,
    StartupConfig,
    StartupPhase,
    StartupTime,
    TrainingBundle,
    Versions,
)


def build_run_config(
    physics_backend: str,
    rendering_backend: str = "none",
    presets: Sequence[str] | None = None,
) -> RunConfig:
    """Assemble a :class:`~isaaclab.test.benchmark.schema.RunConfig`.

    Args:
        physics_backend: Physics solver preset used by the run
            (e.g. ``"physx"``, ``"newton_mjwarp"``).
        rendering_backend: Rendering backend, or ``"none"`` for headless runs
            with no camera sensors.
        presets: Active Hydra preset tokens (e.g. ``["rgb"]``).  ``None``
            is treated as an empty list.

    Returns:
        Populated :class:`~isaaclab.test.benchmark.schema.RunConfig`.
    """
    return RunConfig(
        physics_backend=physics_backend,
        rendering_backend=rendering_backend,
        presets=list(presets) if presets else [],
    )


def build_run_identity(
    *,
    run_id: str,
    framework: str | None,
    config: RunConfig,
    task: str,
    seed: int,
    start_utc: str,
    end_utc: str,
    status: str = "completed",
    num_envs: int | None = None,
    max_iterations: int | None = None,
) -> RunIdentity:
    """Assemble a :class:`~isaaclab.test.benchmark.schema.RunIdentity`.

    The wall-clock duration is derived from the ISO-8601 timestamps; it is
    clamped to zero so clock skew cannot produce a negative value.

    Args:
        run_id: Stable identifier for the run.
        framework: RL library (e.g. ``"rsl_rl"``), or ``None`` for non-learning
            runs.
        config: Physics/rendering/sensor configuration.
        task: Gym task id.
        seed: Environment/agent seed.
        start_utc: ISO-8601 UTC start timestamp.
        end_utc: ISO-8601 UTC end timestamp.
        status: Terminal status of the run.
        num_envs: Number of parallel environments, or ``None`` for startup runs.
        max_iterations: Training iteration budget, or ``None`` for non-training
            runs.

    Returns:
        Populated :class:`~isaaclab.test.benchmark.schema.RunIdentity` with
        ``duration_s`` computed from the timestamps [s].
    """
    duration_s = max(
        0.0,
        (datetime.fromisoformat(end_utc) - datetime.fromisoformat(start_utc)).total_seconds(),
    )
    return RunIdentity(
        run_id=run_id,
        framework=framework,
        config=config,
        task=task,
        seed=seed,
        start_time_utc=start_utc,
        end_time_utc=end_utc,
        duration_s=duration_s,
        status=status,
        num_envs=num_envs,
        max_iterations=max_iterations,
    )


def build_runtime(
    *,
    startup_time_s: StartupTime,
    iteration_times_s: Sequence[float],
    collection_fps: Sequence[float],
    total_fps: Sequence[float],
    steps_per_iteration: int,
) -> Runtime:
    """Assemble a :class:`~isaaclab.test.benchmark.schema.Runtime` from raw series.

    Args:
        startup_time_s: Per-phase startup wall-clock durations [s].
        iteration_times_s: Per-iteration wall-clock time [s].
        collection_fps: Per-iteration environment-stepping throughput
            [frames/s].
        total_fps: Per-iteration end-to-end throughput [frames/s].
        steps_per_iteration: Environment steps collected per iteration.

    Returns:
        Populated :class:`~isaaclab.test.benchmark.schema.Runtime` with
        aggregated :class:`~isaaclab.test.benchmark.schema.MeanStd` fields.
    """
    iter_times = list(iteration_times_s)
    iter_per_s = [1.0 / t for t in iter_times if t > 0]
    return Runtime(
        startup_time_s=startup_time_s,
        iterations_completed=len(iter_times),
        total_wall_time_s=float(sum(iter_times)),
        steps_per_iteration=steps_per_iteration,
        iteration_time_s=mean_std_peak(iter_times),
        collection_fps=mean_std_peak(collection_fps),
        total_fps=mean_std_peak(total_fps),
        iterations_per_s=mean_std_peak(iter_per_s),
    )


def build_learning(
    *,
    reward_series: Sequence[float],
    ep_length_series: Sequence[float],
    ema_alpha: float,
    keep_series: bool = True,
) -> Learning:
    """Assemble a :class:`~isaaclab.test.benchmark.schema.Learning` from raw curves.

    Args:
        reward_series: Per-iteration mean reward values.
        ep_length_series: Per-iteration mean episode-length values.
        ema_alpha: EMA smoothing factor in ``[0, 1]``; higher values weight
            recent observations more.
        keep_series: When ``True`` (default) the full per-iteration series is
            embedded in the bundle; set to ``False`` to reduce file size.

    Returns:
        Populated :class:`~isaaclab.test.benchmark.schema.Learning`.
    """
    rewards = list(reward_series)
    ep_lengths = list(ep_length_series)

    reward_curve = LearningCurve(
        final_raw=float(rewards[-1]) if rewards else 0.0,
        final_ema=ema(rewards, ema_alpha),
        series_per_iter=rewards if keep_series else None,
    )
    ep_length_curve = LearningCurve(
        final_raw=float(ep_lengths[-1]) if ep_lengths else 0.0,
        final_ema=ema(ep_lengths, ema_alpha),
        series_per_iter=ep_lengths if keep_series else None,
    )
    return Learning(ema_alpha=ema_alpha, reward=reward_curve, ep_length=ep_length_curve)


def build_runtime_bundle(
    *,
    run: RunIdentity,
    versions: Versions,
    hardware: Hardware,
    runtime: Runtime,
    resources: Resources,
    extra: dict | None = None,
) -> RuntimeBundle:
    """Assemble a :class:`~isaaclab.test.benchmark.schema.RuntimeBundle`.

    Args:
        run: Run identity metadata.
        versions: Software versions snapshot.
        hardware: Host hardware snapshot.
        runtime: Aggregated runtime metrics.
        resources: Aggregated resource-utilisation metrics.
        extra: Optional free-form scalar values not covered by the stable
            schema.

    Returns:
        Populated :class:`~isaaclab.test.benchmark.schema.RuntimeBundle`.
    """
    return RuntimeBundle(
        run=run,
        versions=versions,
        hardware=hardware,
        runtime=runtime,
        resources=resources,
        extra=extra,
    )


def build_training_bundle(
    *,
    run: RunIdentity,
    versions: Versions,
    hardware: Hardware,
    runtime: Runtime,
    resources: Resources,
    learning: Learning,
    success_rate: float | None = None,
    checkpoint_path: str | None = None,
    video_path: str | None = None,
    extra: dict | None = None,
) -> TrainingBundle:
    """Assemble a :class:`~isaaclab.test.benchmark.schema.TrainingBundle`.

    Args:
        run: Run identity metadata.
        versions: Software versions snapshot.
        hardware: Host hardware snapshot.
        runtime: Aggregated runtime metrics.
        resources: Aggregated resource-utilisation metrics.
        learning: Aggregated learning curves.
        success_rate: Final success rate ``[0..1]``, or ``None`` when the task
            does not track one.
        checkpoint_path: Path to the final saved policy checkpoint, if any.
        video_path: Path to a recorded rollout video/gif, if any.
        extra: Optional free-form scalar values not covered by the stable
            schema.

    Returns:
        Populated :class:`~isaaclab.test.benchmark.schema.TrainingBundle`.
    """
    return TrainingBundle(
        run=run,
        versions=versions,
        hardware=hardware,
        runtime=runtime,
        resources=resources,
        learning=learning,
        success_rate=success_rate,
        checkpoint_path=checkpoint_path,
        video_path=video_path,
        extra=extra,
    )


def build_startup_bundle(
    *,
    run: RunIdentity,
    versions: Versions,
    hardware: Hardware,
    phases: dict[str, StartupPhase],
    top_n: int,
    whitelist: str | None,
    extra: dict | None = None,
) -> StartupBundle:
    """Assemble a :class:`~isaaclab.test.benchmark.schema.StartupBundle`.

    Args:
        run: Run identity metadata (``framework``, ``num_envs``, and
            ``max_iterations`` are typically ``None`` for startup profiles).
        versions: Software versions snapshot.
        hardware: Host hardware snapshot.
        phases: Per-phase timing and cProfile data, keyed by phase name.
        top_n: Number of top cProfile functions retained per phase.
        whitelist: Optional cProfile name-filter pattern; ``None`` means no
            filtering.
        extra: Optional free-form scalar values not covered by the stable
            schema.

    Returns:
        Populated :class:`~isaaclab.test.benchmark.schema.StartupBundle`.
    """
    config = StartupConfig(top_n=top_n, whitelist=whitelist)
    return StartupBundle(
        run=run,
        versions=versions,
        hardware=hardware,
        phases=phases,
        config=config,
        extra=extra,
    )
