# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared benchmark bundle fixtures."""

from dataclasses import replace

import pytest

from isaaclab.benchmark import formatters
from isaaclab.benchmark.schema import (
    CProfileFunction,
    EnvironmentStepTiming,
    GpuDeviceInfo,
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


@pytest.fixture(autouse=True)
def reset_formatters():
    """Formatter instances are process-global caches; keep tests independent."""
    formatters.MetricsFormatter.reset_instances()
    yield
    formatters.MetricsFormatter.reset_instances()


@pytest.fixture
def versions() -> Versions:
    return Versions(
        isaaclab="4.6.8",
        isaacsim="5.0.0",
        kit="107.1.0",
        newton="0.1.2",
        warp="1.7.3",
        mjwarp="0.0.4",
        torch="2.5.1",
        rsl_rl="2.3.0",
        rl_games=None,
        skrl=None,
        sb3=None,
        git_commit="3d42b11d513",
        git_branch="develop",
        git_dirty=False,
    )


@pytest.fixture
def hardware() -> Hardware:
    return Hardware(
        hostname="benchmark-host",
        gpu_devices=[GpuDeviceInfo(name="NVIDIA H100 80GB", mem_gb=80.0, compute_cap="9.0")],
        cpu_name="AMD EPYC 7763",
        cpu_count=64,
        ram_gb=512.0,
    )


@pytest.fixture
def resources() -> Resources:
    return Resources(
        gpu_util_pct=MeanStd(mean=80.0, std=5.0),
        gpu_mem_gb=MeanStd(mean=10.0, std=0.5, peak=12.0),
        cpu_util_pct=MeanStd(mean=30.0, std=4.0),
        ram_gb=MeanStd(mean=20.0, std=1.0, peak=24.0),
    )


@pytest.fixture
def run_identity() -> RunIdentity:
    return RunIdentity(
        run_id="runtime_newton_mjwarp_Isaac-Ant-Direct-v0_20260422-131500_seed42",
        framework=None,
        config=RunConfig(physics_backend="newton_mjwarp", rendering_backend="none"),
        task="Isaac-Ant-Direct-v0",
        seed=42,
        start_time_utc="2026-04-22T13:15:00Z",
        end_time_utc="2026-04-22T13:15:10Z",
        duration_s=10.0,
        status="completed",
        num_envs=16,
    )


@pytest.fixture
def serialized_step_timing() -> EnvironmentStepTiming:
    return EnvironmentStepTiming(
        environment_step_time_s=MeanStd(mean=0.08, std=0.01, peak=0.1),
        environment_step_fps=MeanStd(mean=200.0, std=2.0, peak=205.0),
        simulation_step_time_s=MeanStd(mean=0.05, std=0.01, peak=0.07),
        outside_simulation_step_time_s=MeanStd(mean=0.03, std=0.005, peak=0.04),
        outside_simulation_step_fraction=0.375,
        environment_step_calls=100,
        simulation_step_calls=400,
        measurement_mode="serialized_synchronized",
    )


@pytest.fixture
def runtime() -> Runtime:
    return Runtime(
        startup_time_s=StartupTime(app_launch=1.0, env_creation=2.0, first_step=0.5),
        iterations_completed=1,
        total_wall_time_s=4.0,
        steps_per_iteration=24,
        iteration_time_s=MeanStd(mean=1.0, std=0.0),
        collection_fps=MeanStd(mean=100.0, std=0.0),
        total_fps=MeanStd(mean=100.0, std=0.0),
        iterations_per_s=MeanStd(mean=1.0, std=0.0),
    )


@pytest.fixture
def runtime_bundle(run_identity, versions, hardware, runtime, resources) -> RuntimeBundle:
    return RuntimeBundle(run=run_identity, versions=versions, hardware=hardware, runtime=runtime, resources=resources)


@pytest.fixture
def training_bundle(runtime_bundle) -> TrainingBundle:
    return TrainingBundle(
        run=replace(runtime_bundle.run, framework="rsl_rl", max_iterations=1),
        versions=runtime_bundle.versions,
        hardware=runtime_bundle.hardware,
        runtime=runtime_bundle.runtime,
        resources=runtime_bundle.resources,
        learning=Learning(
            ema_alpha=0.95,
            reward=LearningCurve(final_raw=3.0, final_ema=2.5, series_per_iter=[1.0, 3.0]),
            ep_length=LearningCurve(final_raw=20.0, final_ema=18.0, series_per_iter=[10.0, 20.0]),
            success_rate=LearningCurve(final_raw=0.95, final_ema=0.91, series_per_iter=[0.1, 0.5, 0.95]),
        ),
        success_rate=0.75,
        checkpoint_path="logs/rsl_rl/ant/model_499.pt",
    )


@pytest.fixture
def play_bundle(runtime_bundle) -> PlayBundle:
    return PlayBundle(
        run=replace(runtime_bundle.run, framework="rsl_rl"),
        versions=runtime_bundle.versions,
        hardware=runtime_bundle.hardware,
        runtime=runtime_bundle.runtime,
        resources=runtime_bundle.resources,
        success_rate=0.75,
        reward=MeanStd(mean=4.0, std=1.0, peak=5.0),
        ep_length=MeanStd(mean=20.0, std=2.0, peak=25.0),
        checkpoint_path="model.pt",
    )


@pytest.fixture
def startup_bundle(runtime_bundle) -> StartupBundle:
    return StartupBundle(
        run=replace(runtime_bundle.run, num_envs=None),
        versions=runtime_bundle.versions,
        hardware=runtime_bundle.hardware,
        phases={
            "python_imports": StartupPhase(
                total_time_s=0.25,
                top_functions=[
                    CProfileFunction(name="isaaclab_tasks.utils:import", own_time_s=0.1, cum_time_s=0.2, calls=2)
                ],
            ),
            "first_step": StartupPhase(total_time_s=4.1, top_functions=[]),
        },
        config=StartupConfig(top_n=1, whitelist=None),
    )
