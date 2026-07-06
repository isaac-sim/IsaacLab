# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for benchmark bundle builders (Isaac-Sim-free)."""

import json
import os

import pytest

from isaaclab.test.benchmark import builders
from isaaclab.test.benchmark.schema import (
    GpuDeviceInfo,
    Hardware,
    MeanStd,
    Resources,
    RuntimeBundle,
    StartupTime,
    TrainingBundle,
    Versions,
)
from isaaclab.test.benchmark.serialize import write_bundle_file


def _versions():
    return Versions(
        isaaclab="4.6.8",
        isaacsim=None,
        kit=None,
        newton=None,
        warp=None,
        mjwarp=None,
        torch="2.5.1",
        rsl_rl=None,
        rl_games=None,
        skrl=None,
        sb3=None,
        git_commit=None,
        git_branch=None,
        git_dirty=False,
    )


def _hardware():
    return Hardware(
        hostname="h",
        gpu_devices=[GpuDeviceInfo(name="g", mem_gb=80.0, compute_cap="9.0")],
        cpu_name="c",
        cpu_count=64,
        ram_gb=512.0,
    )


def _resources():
    return Resources(
        gpu_util_pct=MeanStd(80.0, 5.0),
        gpu_mem_gb=MeanStd(10.0, 0.5, 12.0),
        cpu_util_pct=MeanStd(30.0, 4.0),
        ram_gb=MeanStd(20.0, 1.0, 24.0),
    )


def test_run_config_presets_default_empty():
    assert builders.build_run_config("physx").presets == []
    assert builders.build_run_config("newton_mjwarp", presets=["rgb"]).presets == ["rgb"]


def test_run_identity_computes_duration():
    run = builders.build_run_identity(
        run_id="x",
        framework="rsl_rl",
        config=builders.build_run_config("newton_mjwarp"),
        task="t",
        seed=0,
        start_utc="2026-04-22T13:15:00+00:00",
        end_utc="2026-04-22T13:15:30+00:00",
        num_envs=4096,
        max_iterations=500,
    )
    assert run.duration_s == pytest.approx(30.0)


def test_build_runtime_aggregates():
    rt = builders.build_runtime(
        startup_time_s=StartupTime(app_launch=1.0, env_creation=2.0, first_step=0.5),
        iteration_times_s=[1.0, 1.0, 2.0],
        collection_fps=[100.0, 110.0],
        total_fps=[90.0, 95.0],
        steps_per_iteration=24,
    )
    assert rt.iterations_completed == 3
    assert rt.total_wall_time_s == pytest.approx(4.0)
    assert rt.total_fps.peak == pytest.approx(95.0)
    assert rt.iterations_per_s.mean > 0


def test_build_training_bundle_round_trips(tmp_path):
    run = builders.build_run_identity(
        run_id="x",
        framework="rsl_rl",
        config=builders.build_run_config("physx"),
        task="t",
        seed=0,
        start_utc="2026-04-22T13:15:00+00:00",
        end_utc="2026-04-22T13:16:00+00:00",
        num_envs=16,
        max_iterations=5,
    )
    rt = builders.build_runtime(
        startup_time_s=StartupTime(1.0, 2.0, 0.5),
        iteration_times_s=[1.0, 1.0],
        collection_fps=[100.0],
        total_fps=[90.0],
        steps_per_iteration=24,
    )
    learning = builders.build_learning(reward_series=[1.0, 2.0, 3.0], ep_length_series=[10.0, 12.0], ema_alpha=0.1)
    b = builders.build_training_bundle(
        run=run,
        versions=_versions(),
        hardware=_hardware(),
        runtime=rt,
        resources=_resources(),
        learning=learning,
        success_rate=0.9,
        checkpoint_path="m.pt",
    )
    assert isinstance(b, TrainingBundle)
    assert b.learning.reward.final_raw == pytest.approx(3.0)
    p = os.path.join(tmp_path, "training.json")
    write_bundle_file(b, p)
    with open(p) as fh:
        data = json.load(fh)
    assert data["success_rate"] == pytest.approx(0.9)
    assert data["runtime"]["total_fps"]["mean"] == pytest.approx(90.0)


def test_build_learning_empty_series():
    learning = builders.build_learning(reward_series=[], ep_length_series=[], ema_alpha=0.1)
    assert learning.reward.final_raw == pytest.approx(0.0)
    assert learning.reward.final_ema == pytest.approx(0.0)
    assert learning.reward.series_per_iter == []
    assert learning.ep_length.final_raw == pytest.approx(0.0)
    assert learning.ep_length.final_ema == pytest.approx(0.0)
    assert learning.ep_length.series_per_iter == []


def test_build_learning_keep_series_false():
    learning = builders.build_learning(
        reward_series=[1.0, 2.0], ep_length_series=[10.0], ema_alpha=0.1, keep_series=False
    )
    assert learning.reward.series_per_iter is None
    assert learning.ep_length.series_per_iter is None


def test_build_runtime_bundle_no_learning(tmp_path):
    run = builders.build_run_identity(
        run_id="x",
        framework=None,
        config=builders.build_run_config("newton_mjwarp"),
        task="t",
        seed=0,
        start_utc="2026-04-22T13:15:00+00:00",
        end_utc="2026-04-22T13:15:10+00:00",
        num_envs=16,
    )
    rt = builders.build_runtime(
        startup_time_s=StartupTime(1.0, 2.0, 0.5),
        iteration_times_s=[1.0],
        collection_fps=[100.0],
        total_fps=[100.0],
        steps_per_iteration=24,
    )
    b = builders.build_runtime_bundle(
        run=run, versions=_versions(), hardware=_hardware(), runtime=rt, resources=_resources()
    )
    assert isinstance(b, RuntimeBundle)
    p = os.path.join(tmp_path, "runtime.json")
    write_bundle_file(b, p)
    with open(p) as fh:
        assert json.load(fh)["run"]["framework"] is None
