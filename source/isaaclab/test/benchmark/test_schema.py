# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the v1.0 Isaac Lab benchmark schema."""

import dataclasses
import json
import os

import pytest

from isaaclab.benchmark.schema import (
    SCHEMA_VERSION,
    CProfileFunction,
    GpuDeviceInfo,
    Hardware,
    Learning,
    LearningCurve,
    MeanStd,
    MeanStdPeak,
    Resources,
    RunIdentity,
    Runtime,
    StartupBundle,
    StartupConfig,
    StartupPhase,
    StartupPhaseTimes,
    StartupRunIdentity,
    TrainingBundle,
    Versions,
    write_bundle_file,
)


def _minimal_training_bundle() -> TrainingBundle:
    """Construct a valid TrainingBundle with placeholder numeric values."""
    return TrainingBundle(
        run=RunIdentity(
            run_id="rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42",
            framework="rsl_rl",
            backend="physx",
            task="Isaac-Ant-Direct-v0",
            seed=42,
            num_envs=4096,
            max_iterations=500,
            start_time_utc="2026-04-22T13:15:00Z",
            end_time_utc="2026-04-22T13:47:22Z",
            duration_s=1942.1,
            status="completed",
        ),
        versions=Versions(
            isaaclab="4.6.8",
            isaacsim="5.0.0",
            kit="107.1.0",
            newton="0.1.2",
            warp="1.7.3",
            mjwarp="0.0.4",
            torch="2.5.1",
            rsl_rl="2.3.0",
            skrl=None,
            git_commit="3d42b11d513",
            git_branch="develop",
            git_dirty=False,
        ),
        hardware=Hardware(
            hostname="benchmark-host",
            gpu_devices=[GpuDeviceInfo(name="NVIDIA H100 80GB", mem_gb=80.0, compute_cap="9.0")],
            cpu_name="AMD EPYC 7763",
            cpu_count=64,
            ram_gb=512.0,
        ),
        runtime=Runtime(
            startup_phase_times_s=StartupPhaseTimes(app_launch=18.4, env_creation=22.9, first_step=4.1),
            iterations_completed=500,
            total_wall_time_s=1946.0,
            steps_per_iteration=24,
            iteration_time_s=MeanStd(mean=3.82, std=0.04),
            env_steps_per_s=MeanStd(mean=1_071_780.0, std=11_200.0),
            iterations_per_s=MeanStd(mean=0.2618, std=0.0028),
        ),
        resources=Resources(
            gpu_util_pct=MeanStd(mean=87.2, std=6.1),
            gpu_mem_gb=MeanStdPeak(mean=18.4, std=0.3, peak=19.2),
            cpu_util_pct=MeanStd(mean=31.5, std=4.8),
            ram_gb=MeanStdPeak(mean=22.1, std=0.4, peak=24.8),
        ),
        learning=Learning(
            ema_alpha=0.05,
            reward=LearningCurve(final_raw=1823.4, final_ema=1796.1, series_per_iter=[12.3, 34.5, 58.1]),
            ep_length=LearningCurve(final_raw=987.0, final_ema=962.3, series_per_iter=[4.1, 5.0, 7.2]),
        ),
    )


def test_training_bundle_round_trip(tmp_path):
    """Writing a TrainingBundle and reloading via json gives back identical data."""
    bundle = _minimal_training_bundle()
    path = os.path.join(tmp_path, "training.json")
    write_bundle_file(bundle, path)

    with open(path) as f:
        data = json.load(f)

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["run"]["run_id"] == bundle.run.run_id
    assert data["runtime"]["env_steps_per_s"]["mean"] == pytest.approx(1_071_780.0)
    assert data["resources"]["ram_gb"]["peak"] == pytest.approx(24.8)
    assert data["learning"]["reward"]["series_per_iter"] == [12.3, 34.5, 58.1]
    assert data["versions"]["skrl"] is None


def test_training_bundle_without_series(tmp_path):
    """With series_per_iter=None, the JSON contains an explicit null."""
    bundle = _minimal_training_bundle()
    bundle_no_series = dataclasses.replace(
        bundle,
        learning=Learning(
            ema_alpha=0.05,
            reward=LearningCurve(final_raw=1.0, final_ema=1.0, series_per_iter=None),
            ep_length=LearningCurve(final_raw=1.0, final_ema=1.0, series_per_iter=None),
        ),
    )
    path = os.path.join(tmp_path, "training.json")
    write_bundle_file(bundle_no_series, path)
    with open(path) as f:
        data = json.load(f)
    assert data["learning"]["reward"]["series_per_iter"] is None
    assert data["learning"]["ep_length"]["series_per_iter"] is None


def test_startup_bundle_round_trip(tmp_path):
    """StartupBundle round-trips with phase dict and top-function lists."""
    bundle = StartupBundle(
        run=StartupRunIdentity(
            run_id="rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42",
            framework="rsl_rl",
            backend="physx",
            task="Isaac-Ant-Direct-v0",
            seed=42,
            start_time_utc="2026-04-22T13:15:00Z",
            end_time_utc="2026-04-22T13:15:48Z",
            duration_s=48.7,
            status="completed",
        ),
        versions=_minimal_training_bundle().versions,
        hardware=_minimal_training_bundle().hardware,
        phases={
            "app_launch": StartupPhase(
                total_time_s=18.4,
                top_functions=[CProfileFunction(name="isaaclab.x:y", own_time_s=1.8, cum_time_s=2.4, calls=4312)],
            ),
            "env_creation": StartupPhase(total_time_s=22.9, top_functions=[]),
            "first_step": StartupPhase(total_time_s=4.1, top_functions=[]),
        },
        config=StartupConfig(top_n=30, whitelist="startup_whitelist.yaml"),
    )
    path = os.path.join(tmp_path, "startup.json")
    write_bundle_file(bundle, path)
    with open(path) as f:
        data = json.load(f)
    assert data["phases"]["app_launch"]["total_time_s"] == pytest.approx(18.4)
    assert data["phases"]["app_launch"]["top_functions"][0]["calls"] == 4312


def test_package_reexports_match_schema_module():
    """`from isaaclab.benchmark import ...` resolves to the same objects as
    `from isaaclab.benchmark.schema import ...`. Keeps the convenience
    namespace honest if someone forgets to update __all__."""
    import isaaclab.benchmark as pkg
    from isaaclab.benchmark import schema

    for name in pkg.__all__:
        assert getattr(pkg, name) is getattr(schema, name), name
