# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed programmatic API for Isaac Lab benchmark workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, TypeVar, overload

from .dispatch import run_benchmark_request
from .schema import PlayBundle, RuntimeBundle, StartupBundle, TrainingBundle

BenchmarkBundle = RuntimeBundle | StartupBundle | TrainingBundle | PlayBundle
_BenchmarkBundleT = TypeVar("BenchmarkBundleT", RuntimeBundle, StartupBundle, TrainingBundle, PlayBundle)
BenchmarkFormatter = Literal["schema", "omniperf", "osmo", "json", "summary"]
BenchmarkRlLibrary = Literal["rl_games", "rsl_rl", "sb3", "skrl"]
BenchmarkVisualizer = Literal["kit", "newton", "rerun", "viser"]


@dataclass(frozen=True)
class BenchmarkOutputConfig:
    """Output configuration shared by benchmark workflows.

    Args:
        path: Directory receiving benchmark output files.
        formatters: Output formatters to run.
    """

    path: Path = Path(".")
    formatters: tuple[BenchmarkFormatter, ...] = ("schema",)

    def __post_init__(self) -> None:
        if not self.formatters:
            raise ValueError("formatters must contain at least one formatter")


@dataclass(frozen=True)
class BenchmarkLauncherConfig:
    """Simulation launcher configuration shared by benchmark workflows.

    Args:
        device: Simulation device identifier, such as ``"cpu"`` or ``"cuda:0"``.
        enable_cameras: Whether to enable camera rendering.
        visualizers: Visualizers to enable. An empty tuple explicitly disables all visualizers;
            ``None`` preserves task and environment defaults.
        max_visible_envs: Maximum number of environments shown by visualizers.
        experience: Isaac Sim experience file.
        deterministic: Whether to request deterministic rendering and backend behavior.
        animation_recording: Whether to record time-sampled USD animations.
        animation_recording_start_time: Simulation time when animation recording starts [s].
        animation_recording_stop_time: Simulation time when animation recording stops [s].
        kit_args: Arguments forwarded directly to Omniverse Kit.
        livestream: Livestream mode, where ``0`` disables and ``1`` or ``2`` enables WebRTC.
        xr: Whether to enable XR mode.
        verbose: Whether to enable verbose simulator logging.
        info: Whether to enable informational simulator logging.
    """

    device: str | None = None
    enable_cameras: bool = False
    visualizers: tuple[BenchmarkVisualizer, ...] | None = None
    max_visible_envs: int | None = None
    experience: str | None = None
    deterministic: bool = False
    animation_recording: bool = False
    animation_recording_start_time: float | None = None
    animation_recording_stop_time: float | None = None
    kit_args: str | None = None
    livestream: Literal[0, 1, 2] | None = None
    xr: bool = False
    verbose: bool = False
    info: bool = False


@dataclass(frozen=True)
class BenchmarkRuntimeRequest:
    """Request an environment runtime benchmark.

    Args:
        task: Registered Gym task identifier.
        num_envs: Number of parallel environments.
        num_steps: Number of measured environment steps.
        warmup_steps: Number of warm-up steps excluded from throughput measurements.
        seed: Environment seed.
        measure_synchronized_step_breakdown: Whether to collect serialized synchronized
            environment/simulation step diagnostics.
        presets: Typed preset names applied to the task configuration.
        hydra_args: Additional Hydra overrides.
        output: Output configuration.
        launcher: Simulation launcher configuration.
    """

    task: str
    num_envs: int | None = None
    num_steps: int = 1000
    warmup_steps: int = 50
    seed: int | None = None
    measure_synchronized_step_breakdown: bool = False
    presets: tuple[str, ...] = field(default_factory=tuple)
    hydra_args: tuple[str, ...] = field(default_factory=tuple)
    output: BenchmarkOutputConfig = field(default_factory=BenchmarkOutputConfig)
    launcher: BenchmarkLauncherConfig = field(default_factory=BenchmarkLauncherConfig)

    @property
    def workflow(self) -> Literal["runtime"]:
        """Return the workflow dispatcher key."""
        return "runtime"


@dataclass(frozen=True)
class BenchmarkStartupRequest:
    """Request a startup profiling benchmark.

    Args:
        task: Registered Gym task identifier.
        num_envs: Number of parallel environments.
        seed: Environment seed.
        top_n: Number of top cProfile functions retained per phase.
        whitelist_config: Optional YAML whitelist for phase-specific functions.
        presets: Typed preset names applied to the task configuration.
        hydra_args: Additional Hydra overrides.
        output: Output configuration.
        launcher: Simulation launcher configuration.
    """

    task: str
    num_envs: int | None = None
    seed: int | None = None
    top_n: int | None = None
    whitelist_config: Path | None = None
    presets: tuple[str, ...] = field(default_factory=tuple)
    hydra_args: tuple[str, ...] = field(default_factory=tuple)
    output: BenchmarkOutputConfig = field(default_factory=BenchmarkOutputConfig)
    launcher: BenchmarkLauncherConfig = field(default_factory=BenchmarkLauncherConfig)

    @property
    def workflow(self) -> Literal["startup"]:
        """Return the workflow dispatcher key."""
        return "startup"


@dataclass(frozen=True)
class BenchmarkTrainingRequest:
    """Request an RL training benchmark.

    Args:
        backend: Reinforcement-learning backend to benchmark.
        task: Registered Gym task identifier.
        agent: Optional task agent configuration entry point.
        num_envs: Number of parallel environments.
        seed: Environment and agent seed.
        max_iterations: Maximum training iterations.
        warmup_steps: Number of initial environment steps excluded from environment-step timing.
        ray_proc_id: Ray worker process identifier.
        video: Whether to record training videos.
        video_length: Recorded video length [steps].
        video_interval: Interval between video recordings [steps].
        export_io_descriptors: Whether to export environment IO descriptors.
        capture_env_sensors: Number of environment views captured from each image-like sensor.
        capture_env_sensors_length: Length of each sensor capture window [steps].
        capture_env_sensors_interval: Interval between sensor capture windows [steps].
        capture_env_sensors_format: Storage format for captured sensor frames.
        ema_alpha: Learning-curve exponential moving-average coefficient.
        keep_series: Whether to retain full per-iteration learning series.
        check_success: Whether supported backends should stop after success convergence.
        success_threshold: Optional success threshold override.
        success_window: Optional success convergence window override [iterations].
        measure_synchronized_step_breakdown: Whether to collect serialized synchronized
            environment/simulation step diagnostics.
        presets: Typed preset names applied to the task configuration.
        backend_args: Backend-specific command-line arguments.
        hydra_args: Additional Hydra overrides.
        output: Output configuration.
        launcher: Simulation launcher configuration.
    """

    backend: BenchmarkRlLibrary
    task: str
    agent: str | None = None
    num_envs: int | None = None
    seed: int | None = None
    max_iterations: int | None = None
    warmup_steps: int = 1
    ray_proc_id: int | None = None
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    export_io_descriptors: bool = False
    capture_env_sensors: int = 0
    capture_env_sensors_length: int = 200
    capture_env_sensors_interval: int = 2000
    capture_env_sensors_format: Literal["tensorboard", "file"] = "tensorboard"
    ema_alpha: float = 0.1
    keep_series: bool = True
    check_success: bool = False
    success_threshold: float | None = None
    success_window: int | None = None
    measure_synchronized_step_breakdown: bool = False
    presets: tuple[str, ...] = field(default_factory=tuple)
    backend_args: tuple[str, ...] = field(default_factory=tuple)
    hydra_args: tuple[str, ...] = field(default_factory=tuple)
    output: BenchmarkOutputConfig = field(default_factory=BenchmarkOutputConfig)
    launcher: BenchmarkLauncherConfig = field(default_factory=BenchmarkLauncherConfig)

    @property
    def workflow(self) -> Literal["training"]:
        """Return the workflow dispatcher key."""
        return "training"


@dataclass(frozen=True)
class BenchmarkPlayRequest:
    """Request an RL checkpoint playback benchmark.

    Args:
        backend: Reinforcement-learning backend to benchmark.
        task: Registered Gym task identifier.
        checkpoint: Local or Nucleus checkpoint path. When omitted, the backend may use a
            published checkpoint.
        agent: Optional task agent configuration entry point.
        num_envs: Number of parallel environments.
        num_steps: Number of measured inference steps.
        warmup_steps: Number of initial environment steps excluded from environment-step timing.
        seed: Environment seed.
        measure_synchronized_step_breakdown: Whether to collect serialized synchronized
            environment/simulation step diagnostics.
        presets: Typed preset names applied to the task configuration.
        backend_args: Backend-specific command-line arguments.
        hydra_args: Additional Hydra overrides.
        output: Output configuration.
        launcher: Simulation launcher configuration.
    """

    backend: BenchmarkRlLibrary
    task: str
    checkpoint: str | None = None
    agent: str | None = None
    num_envs: int | None = None
    num_steps: int = 100
    warmup_steps: int = 1
    seed: int | None = None
    measure_synchronized_step_breakdown: bool = False
    presets: tuple[str, ...] = field(default_factory=tuple)
    backend_args: tuple[str, ...] = field(default_factory=tuple)
    hydra_args: tuple[str, ...] = field(default_factory=tuple)
    output: BenchmarkOutputConfig = field(default_factory=BenchmarkOutputConfig)
    launcher: BenchmarkLauncherConfig = field(default_factory=BenchmarkLauncherConfig)

    @property
    def workflow(self) -> Literal["play"]:
        """Return the workflow dispatcher key."""
        return "play"


BenchmarkRequest = BenchmarkRuntimeRequest | BenchmarkStartupRequest | BenchmarkTrainingRequest | BenchmarkPlayRequest


@dataclass(frozen=True)
class BenchmarkResult(Generic[_BenchmarkBundleT]):
    """Completed benchmark result.

    Args:
        bundle: Typed benchmark result bundle.
        output_paths: Files written by the selected formatters.
    """

    bundle: _BenchmarkBundleT
    output_paths: tuple[Path, ...]


@overload
def run_benchmark(request: BenchmarkRuntimeRequest) -> BenchmarkResult[RuntimeBundle]: ...


@overload
def run_benchmark(request: BenchmarkStartupRequest) -> BenchmarkResult[StartupBundle]: ...


@overload
def run_benchmark(request: BenchmarkTrainingRequest) -> BenchmarkResult[TrainingBundle]: ...


@overload
def run_benchmark(request: BenchmarkPlayRequest) -> BenchmarkResult[PlayBundle]: ...


def run_benchmark(request: BenchmarkRequest) -> BenchmarkResult:
    """Run a benchmark workflow from a typed request.

    Args:
        request: Runtime, startup, training, or play benchmark request.

    Returns:
        Completed benchmark bundle and output paths.
    """
    return run_benchmark_request(request)


def run_runtime_benchmark(request: BenchmarkRuntimeRequest) -> BenchmarkResult[RuntimeBundle]:
    """Run an environment runtime benchmark.

    Args:
        request: Runtime benchmark request.

    Returns:
        Completed benchmark bundle and output paths.
    """
    return run_benchmark(request)


def run_startup_benchmark(request: BenchmarkStartupRequest) -> BenchmarkResult[StartupBundle]:
    """Run a startup profiling benchmark.

    Args:
        request: Startup benchmark request.

    Returns:
        Completed benchmark bundle and output paths.
    """
    return run_benchmark(request)


def run_training_benchmark(request: BenchmarkTrainingRequest) -> BenchmarkResult[TrainingBundle]:
    """Run an RL training benchmark.

    Args:
        request: Training benchmark request.

    Returns:
        Completed benchmark bundle and output paths.
    """
    return run_benchmark(request)


def run_play_benchmark(request: BenchmarkPlayRequest) -> BenchmarkResult[PlayBundle]:
    """Run an RL checkpoint playback benchmark.

    Args:
        request: Playback benchmark request.

    Returns:
        Completed benchmark bundle and output paths.
    """
    return run_benchmark(request)
