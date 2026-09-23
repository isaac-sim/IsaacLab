# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure assembly functions for benchmark bundle dataclasses.

Converts already-extracted per-iteration series and scalar measurements into
the frozen :mod:`~isaaclab.benchmark.schema` dataclasses that are then
serialised by :func:`~isaaclab.benchmark.serialize.write_bundle_file`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime

from isaaclab.benchmark.metrics import ema, mean_std_peak
from isaaclab.benchmark.schema import (
    EnvironmentStepTiming,
    Hardware,
    Learning,
    LearningCurve,
    MeanStd,
    PlayBundle,
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
    """Assemble a :class:`~isaaclab.benchmark.schema.RunConfig`.

    Args:
        physics_backend: Physics solver preset used by the run
            (e.g. ``"physx"``, ``"newton_mjwarp"``).
        rendering_backend: Rendering backend, or ``"none"`` for headless runs
            with no camera sensors.
        presets: Active Hydra preset tokens (e.g. ``["rgb"]``).  ``None``
            is treated as an empty list.

    Returns:
        Populated :class:`~isaaclab.benchmark.schema.RunConfig`.
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
    """Assemble a :class:`~isaaclab.benchmark.schema.RunIdentity`.

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
        Populated :class:`~isaaclab.benchmark.schema.RunIdentity` with
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


def _with_effective_mean(stats: MeanStd, effective_mean: float) -> MeanStd:
    """Return statistics with an effective aggregate mean."""
    return MeanStd(mean=effective_mean, std=stats.std, peak=stats.peak)


def build_runtime(
    *,
    startup_time_s: StartupTime,
    iteration_times_s: Sequence[float],
    collection_fps: Sequence[float],
    total_fps: Sequence[float],
    steps_per_iteration: int,
    aggregate_throughput: bool = False,
    frames_per_environment_step: int | None = None,
    environment_step_warmup_steps: int = 0,
    environment_step_times_s: Sequence[float] | None = None,
    simulation_step_times_s: Sequence[float] | None = None,
    simulation_step_calls: int | None = None,
) -> Runtime:
    """Assemble a :class:`~isaaclab.benchmark.schema.Runtime` from raw series.

    Args:
        startup_time_s: Per-phase startup wall-clock durations [s].
        iteration_times_s: Per-iteration wall-clock time [s].
        collection_fps: Per-iteration environment-stepping throughput
            [frames/s].
        total_fps: Per-iteration end-to-end throughput [frames/s].
        steps_per_iteration: Environment steps collected per iteration.
        aggregate_throughput: When ``True``, report throughput means as total
            completed steps divided by total wall time. Standard deviation
            remains the ordinary sample deviation of the per-iteration rates,
            and peak remains the maximum per-iteration throughput.
        frames_per_environment_step: Number of environment frames processed by each vectorized ``env.step()`` call.
        environment_step_warmup_steps: Number of initial environment-step calls excluded from timing.
        environment_step_times_s: Positive per-environment-step wall times [s].
        simulation_step_times_s: Synchronized simulation wall times per environment step [s].
        simulation_step_calls: Number of measured simulation-step calls.

    Returns:
        Populated :class:`~isaaclab.benchmark.schema.Runtime` with
        aggregated :class:`~isaaclab.benchmark.schema.MeanStd` fields.
    """
    iter_times = list(iteration_times_s)
    iter_per_s = [1.0 / t for t in iter_times if t > 0]
    collection_fps_agg = mean_std_peak(collection_fps)
    total_fps_agg = mean_std_peak(total_fps)
    iterations_per_s_agg = mean_std_peak(iter_per_s)
    if aggregate_throughput and iter_times:
        total_wall_time_s = float(sum(iter_times))
        effective_iterations_per_s = len(iter_times) / total_wall_time_s
        effective_fps = steps_per_iteration * effective_iterations_per_s
        collection_fps_agg = _with_effective_mean(collection_fps_agg, effective_fps)
        total_fps_agg = _with_effective_mean(total_fps_agg, effective_fps)
        iterations_per_s_agg = _with_effective_mean(iterations_per_s_agg, effective_iterations_per_s)
    if environment_step_times_s is not None and len(environment_step_times_s) == 0:
        raise ValueError(
            "No environment-step timing samples remained after warm-up. The workload may have stopped before "
            "warm-up completed; reduce the warm-up count or ensure more environment steps execute."
        )
    if environment_step_times_s is None and (simulation_step_times_s is not None or simulation_step_calls is not None):
        raise ValueError("environment_step_times_s is required with simulation timing")
    environment_step_timing = None
    if environment_step_times_s is not None:
        if frames_per_environment_step is None or frames_per_environment_step <= 0:
            raise ValueError("frames_per_environment_step must be greater than zero")
        environment_samples = list(environment_step_times_s)
        if any(value <= 0 for value in environment_samples):
            raise ValueError("environment_step_times_s must contain only positive samples")
        total_environment_time_s = sum(environment_samples)
        environment_fps_samples = [frames_per_environment_step / value for value in environment_samples]
        environment_step_fps = mean_std_peak(environment_fps_samples)
        if total_environment_time_s > 0:
            environment_step_fps = _with_effective_mean(
                environment_step_fps,
                frames_per_environment_step * len(environment_samples) / total_environment_time_s,
            )

        simulation_step_time_s = None
        outside_simulation_step_time_s = None
        outside_simulation_step_fraction = None
        if simulation_step_times_s is not None:
            if simulation_step_calls is None:
                raise ValueError("simulation_step_calls is required with simulation_step_times_s")
            if simulation_step_calls <= 0:
                raise ValueError("simulation_step_calls must be greater than zero")
            simulation_samples = list(simulation_step_times_s)
            if not simulation_samples or any(value <= 0 for value in simulation_samples):
                raise ValueError("simulation_step_times_s must contain only positive samples")
            if len(environment_samples) != len(simulation_samples):
                raise ValueError("Environment and simulation timing samples must have the same length")
            outside_simulation_samples = []
            for total, simulation in zip(environment_samples, simulation_samples):
                outside = total - simulation
                if outside < 0.0:
                    if not math.isclose(total, simulation, rel_tol=1e-9, abs_tol=1e-12):
                        raise ValueError("simulation time cannot exceed environment-step time")
                    # Preserve a non-negative partition for clock-resolution noise only.
                    outside = 0.0
                outside_simulation_samples.append(outside)
            simulation_step_time_s = mean_std_peak(simulation_samples)
            outside_simulation_step_time_s = mean_std_peak(outside_simulation_samples)
            outside_simulation_step_fraction = sum(outside_simulation_samples) / total_environment_time_s
        elif simulation_step_calls is not None:
            raise ValueError("simulation_step_times_s is required with simulation_step_calls")

        environment_step_timing = EnvironmentStepTiming(
            environment_step_time_s=mean_std_peak(environment_samples),
            environment_step_fps=environment_step_fps,
            simulation_step_time_s=simulation_step_time_s,
            outside_simulation_step_time_s=outside_simulation_step_time_s,
            outside_simulation_step_fraction=outside_simulation_step_fraction,
            environment_step_calls=len(environment_samples),
            simulation_step_calls=simulation_step_calls,
            measurement_mode=("serialized_synchronized" if simulation_step_times_s is not None else "host_return"),
            warmup_steps=environment_step_warmup_steps,
        )

    return Runtime(
        startup_time_s=startup_time_s,
        iterations_completed=len(iter_times),
        total_wall_time_s=float(sum(iter_times)),
        steps_per_iteration=steps_per_iteration,
        iteration_time_s=mean_std_peak(iter_times),
        collection_fps=collection_fps_agg,
        total_fps=total_fps_agg,
        iterations_per_s=iterations_per_s_agg,
        environment_step_timing=environment_step_timing,
    )


def _build_learning_curve(values: Sequence[float], ema_alpha: float, keep_series: bool) -> LearningCurve:
    samples = list(values)
    return LearningCurve(
        final_raw=float(samples[-1]) if samples else 0.0,
        final_ema=ema(samples, ema_alpha),
        series_per_iter=samples if keep_series else None,
    )


def build_learning(
    *,
    reward_series: Sequence[float],
    ep_length_series: Sequence[float],
    ema_alpha: float,
    success_rate_series: Sequence[float] | None = None,
    keep_series: bool = True,
) -> Learning:
    """Assemble a :class:`~isaaclab.benchmark.schema.Learning` from raw curves.

    Args:
        reward_series: Per-iteration mean reward values.
        ep_length_series: Per-iteration mean episode-length values.
        ema_alpha: EMA smoothing factor in ``[0, 1]``; higher values weight
            recent observations more.
        success_rate_series: Per-iteration success-rate values, or ``None``
            when unavailable.
        keep_series: When ``True`` (default) the full per-iteration series is
            embedded in the bundle; set to ``False`` to reduce file size.

    Returns:
        Populated :class:`~isaaclab.benchmark.schema.Learning`. Absent or empty
        success history produces a ``None`` success-rate curve.
    """
    return Learning(
        ema_alpha=ema_alpha,
        reward=_build_learning_curve(reward_series, ema_alpha, keep_series),
        ep_length=_build_learning_curve(ep_length_series, ema_alpha, keep_series),
        success_rate=(
            _build_learning_curve(success_rate_series, ema_alpha, keep_series) if success_rate_series else None
        ),
    )


def build_runtime_bundle(
    *,
    run: RunIdentity,
    versions: Versions,
    hardware: Hardware,
    runtime: Runtime,
    resources: Resources,
    extra: dict | None = None,
) -> RuntimeBundle:
    """Assemble a :class:`~isaaclab.benchmark.schema.RuntimeBundle`.

    Args:
        run: Run identity metadata.
        versions: Software versions snapshot.
        hardware: Host hardware snapshot.
        runtime: Aggregated runtime metrics.
        resources: Aggregated resource-utilisation metrics.
        extra: Optional free-form scalar values not covered by the stable
            schema.

    Returns:
        Populated :class:`~isaaclab.benchmark.schema.RuntimeBundle`.
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
    """Assemble a :class:`~isaaclab.benchmark.schema.TrainingBundle`.

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
        Populated :class:`~isaaclab.benchmark.schema.TrainingBundle`.
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


def build_play_bundle(
    *,
    run: RunIdentity,
    versions: Versions,
    hardware: Hardware,
    runtime: Runtime,
    resources: Resources,
    success_rate: float | None = None,
    reward: MeanStd | None = None,
    ep_length: MeanStd | None = None,
    checkpoint_path: str | None = None,
    video_path: str | None = None,
    extra: dict | None = None,
) -> PlayBundle:
    """Assemble a :class:`~isaaclab.benchmark.schema.PlayBundle`.

    Args:
        run: Run identity metadata.
        versions: Software versions snapshot.
        hardware: Host hardware snapshot.
        runtime: Aggregated runtime metrics.
        resources: Aggregated resource-utilisation metrics.
        success_rate: Mean success rate ``[0..1]`` over completed episodes, or
            ``None`` when the task does not report one.
        reward: Episode-return aggregate over completed episodes, or ``None``
            when no episode completed.
        ep_length: Episode-length aggregate over completed episodes, or ``None``
            when no episode completed.
        checkpoint_path: Path to the policy checkpoint that was rolled out.
        video_path: Path to a recorded rollout video/gif, if any.
        extra: Optional free-form scalar values not covered by the stable
            schema.

    Returns:
        Populated :class:`~isaaclab.benchmark.schema.PlayBundle`.
    """
    return PlayBundle(
        run=run,
        versions=versions,
        hardware=hardware,
        runtime=runtime,
        resources=resources,
        success_rate=success_rate,
        reward=reward,
        ep_length=ep_length,
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
    """Assemble a :class:`~isaaclab.benchmark.schema.StartupBundle`.

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
        Populated :class:`~isaaclab.benchmark.schema.StartupBundle`.
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
