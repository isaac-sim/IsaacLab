# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public schema for Isaac Lab benchmark bundles (v1.2).

Defines the on-disk JSON schema produced by the standalone benchmark scripts
under ``scripts/benchmarks/`` (``runtime.py``, ``training.py``, ``startup.py``).
Producers populate a :class:`TrainingBundle` or :class:`StartupBundle` and call
:func:`~isaaclab.test.benchmark.serialize.write_bundle_file` to emit
schema-compliant JSON. Consumers (dashboards, regression-comparison tools,
the in-tree Odin evaluation harness under ``tools/odin/``) read the same file
and reconstruct the dataclasses.

Each bundle is self-contained: every top-level bundle carries its own
:class:`Versions` and :class:`Hardware` metadata so a reader need not
cross-reference other files in the bundle directory.

Current version: 1.2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

SCHEMA_VERSION = "1.2"

Framework = Literal["rsl_rl", "rl_games", "skrl", "sb3"]
PhysicsBackend = Literal["physx", "newton_mjwarp", "newton_kamino", "ovphysx"]
# "newton" selects Newton's built-in Warp renderer.
RenderingBackend = Literal["none", "isaacsim_rtx", "ovrtx", "newton"]
RunStatus = Literal["completed", "interrupted", "crashed"]


@dataclass(frozen=True)
class MeanStd:
    """Scalar aggregate with mean, standard deviation, and optional peak.

    Args:
        mean: Central value of the aggregate. For most fields this is the
            arithmetic sample mean; for effective-throughput fields it is the
            aggregate rate (total completed work over total wall time).
        std: Ordinary sample standard deviation of the per-sample values.
            This remains centered on the sample mean even when ``mean`` is an
            effective aggregate rate.
        peak: Maximum observed value, or ``None`` where a peak is not
            meaningful (e.g. GPU utilisation, whose ceiling is always 100%).
    """

    mean: float
    std: float
    peak: float | None = None

    def __post_init__(self) -> None:
        # Peak is the maximum observed sample, so it remains greater than or
        # equal to either the sample mean or an effective aggregate rate. A
        # small tolerance absorbs independent rounding.
        if self.peak is not None and self.peak < self.mean - 1e-6:
            raise ValueError(f"peak ({self.peak}) must be >= mean ({self.mean})")


@dataclass(frozen=True)
class GpuDeviceInfo:
    """Information about a single GPU device.

    Args:
        name: Device model name.
        mem_gb: Total device memory [GB].
        compute_cap: CUDA compute capability (e.g. ``"9.0"``).
    """

    name: str
    mem_gb: float
    compute_cap: str


@dataclass(frozen=True)
class Hardware:
    """Host hardware snapshot captured at run time.

    Args:
        hostname: Host machine name.
        gpu_devices: Per-device GPU information.
        cpu_name: CPU model name.
        cpu_count: Physical CPU core count.
        ram_gb: Total host RAM [GB].
    """

    hostname: str
    gpu_devices: list[GpuDeviceInfo]
    cpu_name: str
    cpu_count: int
    ram_gb: float


@dataclass(frozen=True)
class Versions:
    """Software versions captured at run time.

    Version fields are ``None`` when the corresponding runtime or package is
    unavailable.
    """

    isaaclab: str
    isaacsim: str | None
    kit: str | None
    newton: str | None
    warp: str | None
    mjwarp: str | None
    torch: str
    rsl_rl: str | None
    rl_games: str | None
    skrl: str | None
    sb3: str | None
    git_commit: str | None
    git_branch: str | None
    git_dirty: bool
    numpy: str | None = None
    isaaclab_newton: str | None = None
    isaaclab_physx: str | None = None
    isaaclab_ov: str | None = None
    isaaclab_tasks: str | None = None
    isaaclab_rl: str | None = None
    ovrtx: str | None = None
    ovphysx: str | None = None
    mujoco: str | None = None
    cuda_bindings: str | None = None
    usd_core: str | None = None
    isaaclab_release: str | None = None


@dataclass(frozen=True)
class RunConfig:
    """Physics/rendering backend and active presets for a run.

    Args:
        physics_backend: Physics solver preset the run used.
        rendering_backend: Rendering backend, or ``"none"`` for headless runs
            with no camera sensors.
        presets: Active Hydra preset tokens applied to the run (e.g.
            ``["rgb", "ovrtx"]``). Open-ended so sensor data types,
            resolutions, and any other domain presets are captured without a
            closed enum; ``physics_backend`` / ``rendering_backend`` surface the
            two primary grouping dimensions as typed fields.
    """

    physics_backend: PhysicsBackend
    rendering_backend: RenderingBackend = "none"
    presets: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RunIdentity:
    """Identity of a benchmark run (training, runtime, or startup).

    Args:
        run_id: Stable identifier for the run.
        framework: RL library for training runs; ``None`` for non-learning
            (pure runtime, startup) runs.
        config: Physics/rendering/sensor configuration.
        task: Gym task id.
        seed: Environment/agent seed.
        start_time_utc: ISO-8601 UTC start timestamp.
        end_time_utc: ISO-8601 UTC end timestamp.
        duration_s: Wall-clock run duration [s].
        status: Terminal status of the run.
        num_envs: Number of parallel environments, or ``None`` (startup).
        max_iterations: Training iteration budget, or ``None`` (startup, runtime).
    """

    run_id: str
    framework: Framework | None
    config: RunConfig
    task: str
    seed: int
    start_time_utc: str
    end_time_utc: str
    duration_s: float
    status: RunStatus
    num_envs: int | None = None
    max_iterations: int | None = None

    def __post_init__(self) -> None:
        if self.duration_s < 0:
            raise ValueError(f"duration_s must be >= 0, got {self.duration_s}")


@dataclass(frozen=True)
class StartupTime:
    """Wall-clock duration of each startup phase [s]."""

    app_launch: float
    env_creation: float
    first_step: float
    python_imports: float | None = None
    task_config: float | None = None


@dataclass(frozen=True)
class EnvironmentStepTiming:
    """Environment-step wall time and optional synchronized simulation breakdown.

    ``host_return`` mode records how long the host spends inside ``env.step()``
    without forcing device completion. ``serialized_synchronized`` mode drains
    pending work at every environment and simulation boundary so the measured
    environment time can be partitioned into time inside and outside nested
    ``SimulationContext.step()`` calls. The latter mode serializes device work
    and is an observer-perturbed diagnostic, not production throughput.

    When the serialized mode is active, every timing and throughput aggregate
    in the enclosing :class:`Runtime` was collected under that instrumented
    schedule, not only the fields in this breakdown.

    Time outside simulation calls includes required action, actuator, state,
    manager, reset, wrapper, and synchronization work. It is not an estimate of
    removable Isaac Lab overhead.

    Args:
        environment_step_time_s: Per-environment-step wall time [s], interpreted
            according to :attr:`measurement_mode`.
        environment_step_fps: Reciprocal environment-step rate [frames/s],
            interpreted according to :attr:`measurement_mode`.
        simulation_step_time_s: Synchronized simulation wall time per environment step [s], when measured.
        outside_simulation_step_time_s: Time outside simulation calls per environment step [s], when measured.
        outside_simulation_step_fraction: Fraction of synchronized environment-step time outside simulation calls.
        environment_step_calls: Number of measured environment-step calls.
        simulation_step_calls: Number of measured simulation-step calls, when measured.
        measurement_mode: Timing boundary semantics. ``host_return`` does not
            force device completion. ``serialized_synchronized`` explicitly
            synchronizes every measured boundary.
    """

    environment_step_time_s: MeanStd
    environment_step_fps: MeanStd
    simulation_step_time_s: MeanStd | None
    outside_simulation_step_time_s: MeanStd | None
    outside_simulation_step_fraction: float | None
    environment_step_calls: int
    simulation_step_calls: int | None
    measurement_mode: Literal["host_return", "serialized_synchronized"]

    def __post_init__(self) -> None:
        """Validate that the timing fields form one consistent measurement mode."""
        if self.environment_step_calls <= 0:
            raise ValueError("environment_step_calls must be greater than zero")
        if not (
            math.isfinite(self.environment_step_time_s.mean)
            and self.environment_step_time_s.mean > 0.0
            and math.isfinite(self.environment_step_fps.mean)
            and self.environment_step_fps.mean > 0.0
        ):
            raise ValueError("environment step time and FPS must be greater than zero")

        breakdown = (
            self.simulation_step_time_s,
            self.outside_simulation_step_time_s,
            self.outside_simulation_step_fraction,
            self.simulation_step_calls,
        )
        if self.measurement_mode == "host_return":
            if any(value is not None for value in breakdown):
                raise ValueError("host_return timing cannot contain a synchronized simulation breakdown")
            return
        if self.measurement_mode != "serialized_synchronized":
            raise ValueError(f"Unsupported environment-step measurement mode: {self.measurement_mode}")
        if any(value is None for value in breakdown):
            raise ValueError("serialized_synchronized timing requires a complete simulation breakdown")

        assert self.simulation_step_time_s is not None
        assert self.outside_simulation_step_time_s is not None
        assert self.outside_simulation_step_fraction is not None
        assert self.simulation_step_calls is not None
        if self.simulation_step_calls <= 0:
            raise ValueError("simulation_step_calls must be greater than zero")
        if self.environment_step_time_s.mean <= 0.0 or self.simulation_step_time_s.mean <= 0.0:
            raise ValueError("synchronized environment and simulation step times must be greater than zero")
        if self.outside_simulation_step_time_s.mean < 0.0:
            raise ValueError("outside-simulation step time must not be negative")
        if not 0.0 <= self.outside_simulation_step_fraction <= 1.0:
            raise ValueError("outside_simulation_step_fraction must be in the range [0, 1]")
        if not math.isclose(
            self.environment_step_time_s.mean,
            self.simulation_step_time_s.mean + self.outside_simulation_step_time_s.mean,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("synchronized environment time must equal simulation plus outside-simulation time")
        if not math.isclose(
            self.outside_simulation_step_fraction,
            self.outside_simulation_step_time_s.mean / self.environment_step_time_s.mean,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("outside_simulation_step_fraction must match the aggregate timing ratio")


@dataclass(frozen=True)
class Runtime:
    """Aggregated runtime metrics for a run.

    Args:
        startup_time_s: Per-phase startup wall-clock durations [s].
        iterations_completed: Number of completed iterations.
        total_wall_time_s: Total run wall-clock time [s].
        steps_per_iteration: Environment steps collected per iteration.
        iteration_time_s: Per-iteration wall-clock time [s].
        collection_fps: Environment-stepping (rollout) throughput [frames/s] — environment
            steps per second across all environments during data collection (the scripts'
            "Collection FPS" / "Environment + Inference FPS").
        total_fps: End-to-end throughput [frames/s] including the policy update — the headline
            FPS (the scripts' "Total FPS" / "effective FPS"). For pure runtime runs with no
            learning, this equals :attr:`collection_fps`.
        iterations_per_s: Iteration rate [iter/s].
        environment_step_timing: Environment-step timing and optional synchronized
            simulation breakdown, when measured. Its measurement mode also
            describes the schedule used by the enclosing timing and rate fields.
    """

    startup_time_s: StartupTime
    iterations_completed: int
    total_wall_time_s: float
    steps_per_iteration: int
    iteration_time_s: MeanStd
    collection_fps: MeanStd
    total_fps: MeanStd
    iterations_per_s: MeanStd
    environment_step_timing: EnvironmentStepTiming | None = None


@dataclass(frozen=True)
class Resources:
    """Aggregated resource-utilisation metrics for a run.

    Utilisation fields leave :attr:`MeanStd.peak` as ``None`` (a peak of 100% is
    uninformative); memory fields populate ``peak``.

    Args:
        gpu_util_pct: GPU utilisation [%].
        gpu_mem_gb: GPU memory used [GB].
        cpu_util_pct: CPU utilisation [%].
        ram_gb: Host RAM used [GB].
    """

    gpu_util_pct: MeanStd
    gpu_mem_gb: MeanStd
    cpu_util_pct: MeanStd
    ram_gb: MeanStd


@dataclass(frozen=True)
class LearningCurve:
    """One learning curve (reward or episode length)."""

    final_raw: float
    final_ema: float
    series_per_iter: list[float] | None


@dataclass(frozen=True)
class Learning:
    """Learning curves for a training run, plus their EMA smoothing factor."""

    ema_alpha: float
    reward: LearningCurve
    ep_length: LearningCurve


@dataclass(frozen=True)
class RuntimeBundle:
    """Top-level shape of ``runtime.json`` (environment stepping, no learning).

    Mirrors :class:`TrainingBundle` without the learning metrics.

    Args:
        extra: Optional free-form scalar values (experimental or producer-specific)
            that are **not** part of the stable schema contract. Consumers must
            tolerate its absence and must not depend on specific keys; promote a key
            to a typed field once it is stable and broadly useful.
    """

    run: RunIdentity
    versions: Versions
    hardware: Hardware
    runtime: Runtime
    resources: Resources
    extra: dict[str, float | int | str | bool] | None = None
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class TrainingBundle:
    """Top-level shape of ``training.json`` — a runtime bundle plus learning metrics.

    Args:
        success_rate: Final success rate [0..1] when the task tracks one, else
            ``None``.
        checkpoint_path: Path to the final saved policy checkpoint, if any.
        video_path: Path to a recorded rollout video/gif, if any.
        extra: Optional free-form scalar values (experimental or producer-specific)
            that are **not** part of the stable schema contract. Consumers must
            tolerate its absence and must not depend on specific keys; promote a key
            to a typed field once it is stable and broadly useful.
    """

    run: RunIdentity
    versions: Versions
    hardware: Hardware
    runtime: Runtime
    resources: Resources
    learning: Learning
    success_rate: float | None = None
    checkpoint_path: str | None = None
    video_path: str | None = None
    extra: dict[str, float | int | str | bool] | None = None
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class PlayBundle:
    """Top-level shape of ``play.json`` — a checkpoint-driven inference rollout.

    Mirrors :class:`RuntimeBundle` (with :attr:`RunIdentity.framework` set to the
    RL library that produced the checkpoint) and adds the inference-evaluation
    aggregates: a success rate plus scalar reward and episode-length statistics.
    Unlike :class:`TrainingBundle`, :attr:`reward` and :attr:`ep_length` are
    scalar :class:`MeanStd` aggregates over completed episodes, **not**
    per-iteration learning curves.

    Args:
        success_rate: Mean success rate ``[0..1]`` over completed episodes, or
            ``None`` when the task does not report one.
        reward: Episode-return aggregate over completed episodes, or ``None``
            when no episode completed.
        ep_length: Episode-length aggregate over completed episodes, or ``None``
            when no episode completed.
        checkpoint_path: Path to the policy checkpoint that was rolled out.
        video_path: Path to a recorded rollout video/gif, if any.
        extra: Optional free-form scalar values (experimental or producer-specific)
            that are **not** part of the stable schema contract. Consumers must
            tolerate its absence and must not depend on specific keys; promote a key
            to a typed field once it is stable and broadly useful.
    """

    run: RunIdentity
    versions: Versions
    hardware: Hardware
    runtime: Runtime
    resources: Resources
    success_rate: float | None = None
    reward: MeanStd | None = None
    ep_length: MeanStd | None = None
    checkpoint_path: str | None = None
    video_path: str | None = None
    extra: dict[str, float | int | str | bool] | None = None
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class CProfileFunction:
    """One entry from a cProfile top-N table.

    Args:
        name: Function label.
        own_time_s: Own (exclusive) time [s].
        cum_time_s: Cumulative (inclusive) time [s].
        calls: Number of calls.
    """

    name: str
    own_time_s: float
    cum_time_s: float
    calls: int


@dataclass(frozen=True)
class StartupPhase:
    """Wall-clock total plus top cProfile functions for one startup phase."""

    total_time_s: float
    top_functions: list[CProfileFunction]


@dataclass(frozen=True)
class StartupConfig:
    """CLI configuration captured in a :class:`StartupBundle`."""

    top_n: int
    whitelist: str | None


@dataclass(frozen=True)
class StartupBundle:
    """Top-level shape of ``startup.json``.

    Reuses :class:`RunIdentity` with ``framework``/``num_envs``/``max_iterations``
    left unset, since they are not meaningful for a startup profile.

    Args:
        extra: Optional free-form scalar values (experimental or producer-specific)
            that are **not** part of the stable schema contract. Consumers must
            tolerate its absence and must not depend on specific keys; promote a key
            to a typed field once it is stable and broadly useful.
    """

    run: RunIdentity
    versions: Versions
    hardware: Hardware
    phases: dict[str, StartupPhase]
    config: StartupConfig
    extra: dict[str, float | int | str | bool] | None = None
    schema_version: str = SCHEMA_VERSION
