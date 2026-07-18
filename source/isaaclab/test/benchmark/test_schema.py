# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the v1.0 Isaac Lab benchmark schema."""

import dataclasses
import json
import os

import pytest

from isaaclab.test.benchmark.schema import (
    SCHEMA_VERSION,
    CProfileFunction,
    GpuDeviceInfo,
    Hardware,
    Learning,
    LearningCurve,
    MeanStd,
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
from isaaclab.test.benchmark.serialize import write_bundle_file


def _versions() -> Versions:
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


def _hardware() -> Hardware:
    return Hardware(
        hostname="benchmark-host",
        gpu_devices=[GpuDeviceInfo(name="NVIDIA H100 80GB", mem_gb=80.0, compute_cap="9.0")],
        cpu_name="AMD EPYC 7763",
        cpu_count=64,
        ram_gb=512.0,
    )


def _run_identity(framework="rsl_rl", num_envs=4096, max_iterations=500) -> RunIdentity:
    return RunIdentity(
        run_id="rsl-rl_newton_mjwarp_Isaac-Ant-Direct-v0_20260422-131500_seed42",
        framework=framework,
        config=RunConfig(physics_backend="newton_mjwarp", rendering_backend="none"),
        task="Isaac-Ant-Direct-v0",
        seed=42,
        start_time_utc="2026-04-22T13:15:00Z",
        end_time_utc="2026-04-22T13:47:22Z",
        duration_s=1942.1,
        status="completed",
        num_envs=num_envs,
        max_iterations=max_iterations,
    )


def _runtime() -> Runtime:
    return Runtime(
        startup_time_s=StartupTime(app_launch=18.4, env_creation=22.9, first_step=4.1),
        iterations_completed=500,
        total_wall_time_s=1946.0,
        steps_per_iteration=24,
        iteration_time_s=MeanStd(mean=3.82, std=0.04),
        collection_fps=MeanStd(mean=1_142_000.0, std=9_500.0),
        total_fps=MeanStd(mean=1_071_780.0, std=11_200.0),
        iterations_per_s=MeanStd(mean=0.2618, std=0.0028),
    )


def _resources() -> Resources:
    return Resources(
        gpu_util_pct=MeanStd(mean=87.2, std=6.1),  # peak omitted
        gpu_mem_gb=MeanStd(mean=18.4, std=0.3, peak=19.2),
        cpu_util_pct=MeanStd(mean=31.5, std=4.8),
        ram_gb=MeanStd(mean=22.1, std=0.4, peak=24.8),
    )


def _minimal_training_bundle() -> TrainingBundle:
    return TrainingBundle(
        run=_run_identity(),
        versions=_versions(),
        hardware=_hardware(),
        runtime=_runtime(),
        resources=_resources(),
        learning=Learning(
            ema_alpha=0.05,
            reward=LearningCurve(final_raw=1823.4, final_ema=1796.1, series_per_iter=[12.3, 34.5, 58.1]),
            ep_length=LearningCurve(final_raw=987.0, final_ema=962.3, series_per_iter=[4.1, 5.0, 7.2]),
        ),
        success_rate=0.91,
        checkpoint_path="logs/rsl_rl/ant/2026-04-22_13-15-00/model_499.pt",
        video_path=None,
    )


def test_training_bundle_round_trip(tmp_path):
    """TrainingBundle round-trips through JSON with the reshaped fields."""
    bundle = _minimal_training_bundle()
    path = os.path.join(tmp_path, "training.json")
    write_bundle_file(bundle, path)

    with open(path) as f:
        data = json.load(f)

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["run"]["framework"] == "rsl_rl"
    assert data["run"]["config"]["physics_backend"] == "newton_mjwarp"
    assert data["run"]["config"]["rendering_backend"] == "none"
    assert data["run"]["config"]["presets"] == []
    assert data["runtime"]["collection_fps"]["mean"] == pytest.approx(1_142_000.0)
    assert data["runtime"]["total_fps"]["mean"] == pytest.approx(1_071_780.0)
    # merged MeanStd: util has no peak, memory does
    assert data["resources"]["gpu_util_pct"]["peak"] is None
    assert data["resources"]["ram_gb"]["peak"] == pytest.approx(24.8)
    assert data["success_rate"] == pytest.approx(0.91)
    assert data["checkpoint_path"].endswith("model_499.pt")
    assert data["video_path"] is None
    assert data["versions"]["sb3"] is None


def test_runtime_bundle_round_trip(tmp_path):
    """RuntimeBundle (no learning) round-trips; framework/max_iterations may be None."""
    bundle = RuntimeBundle(
        run=_run_identity(framework=None, max_iterations=None),
        versions=_versions(),
        hardware=_hardware(),
        runtime=_runtime(),
        resources=_resources(),
    )
    path = os.path.join(tmp_path, "runtime.json")
    write_bundle_file(bundle, path)
    with open(path) as f:
        data = json.load(f)
    assert data["run"]["framework"] is None
    assert data["run"]["max_iterations"] is None
    assert "learning" not in data
    assert data["resources"]["gpu_mem_gb"]["peak"] == pytest.approx(19.2)


def test_training_bundle_without_series(tmp_path):
    bundle = dataclasses.replace(
        _minimal_training_bundle(),
        learning=Learning(
            ema_alpha=0.05,
            reward=LearningCurve(final_raw=1.0, final_ema=1.0, series_per_iter=None),
            ep_length=LearningCurve(final_raw=1.0, final_ema=1.0, series_per_iter=None),
        ),
    )
    path = os.path.join(tmp_path, "training.json")
    write_bundle_file(bundle, path)
    with open(path) as f:
        data = json.load(f)
    assert data["learning"]["reward"]["series_per_iter"] is None
    assert data["learning"]["ep_length"]["series_per_iter"] is None


def test_startup_bundle_reuses_run_identity(tmp_path):
    """StartupBundle reuses RunIdentity with num_envs/max_iterations/framework unset."""
    bundle = StartupBundle(
        run=RunIdentity(
            run_id="startup_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42",
            framework=None,
            config=RunConfig(physics_backend="physx"),
            task="Isaac-Ant-Direct-v0",
            seed=42,
            start_time_utc="2026-04-22T13:15:00Z",
            end_time_utc="2026-04-22T13:15:48Z",
            duration_s=48.7,
            status="completed",
        ),
        versions=_versions(),
        hardware=_hardware(),
        phases={
            "app_launch": StartupPhase(
                total_time_s=18.4,
                top_functions=[CProfileFunction(name="isaaclab.x:y", own_time_s=1.8, cum_time_s=2.4, calls=4312)],
            ),
            "first_step": StartupPhase(total_time_s=4.1, top_functions=[]),
        },
        config=StartupConfig(top_n=30, whitelist="startup_whitelist.yaml"),
    )
    path = os.path.join(tmp_path, "startup.json")
    write_bundle_file(bundle, path)
    with open(path) as f:
        data = json.load(f)
    assert data["run"]["num_envs"] is None
    assert data["phases"]["app_launch"]["top_functions"][0]["calls"] == 4312


def test_mean_std_rejects_peak_below_mean():
    with pytest.raises(ValueError):
        MeanStd(mean=10.0, std=1.0, peak=5.0)


def test_run_identity_rejects_negative_duration():
    with pytest.raises(ValueError):
        RunIdentity(
            run_id="x",
            framework=None,
            config=RunConfig(physics_backend="physx"),
            task="t",
            seed=0,
            start_time_utc="a",
            end_time_utc="b",
            duration_s=-1.0,
            status="crashed",
        )


def test_package_reexports_match_schema_module():
    """Every schema symbol exported from the package is the same object as in schema.py."""
    import isaaclab.test.benchmark as pkg
    from isaaclab.test.benchmark import schema

    schema_names = {n for n in dir(schema) if not n.startswith("_")}
    checked = [n for n in getattr(pkg, "__all__", []) if n in schema_names]
    assert checked, "no schema symbols found in package __all__"
    for name in checked:
        assert getattr(pkg, name) is getattr(schema, name), name


def test_write_bundle_file_is_atomic(tmp_path, monkeypatch):
    """A failure mid-serialise must not clobber an existing good file."""
    import isaaclab.test.benchmark.serialize as serialize

    path = os.path.join(tmp_path, "training.json")
    write_bundle_file(_minimal_training_bundle(), path)
    with open(path) as fh:
        good = fh.read()

    # Force json.dump to blow up after the temp file is opened.
    def _boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(serialize.json, "dump", _boom)
    with pytest.raises(RuntimeError):
        write_bundle_file(_minimal_training_bundle(), path)

    # Original file is intact; no leftover .tmp.
    with open(path) as fh:
        assert fh.read() == good
    assert not os.path.exists(path + ".tmp")


def test_extra_field_round_trips(tmp_path):
    """The free-form `extra` mapping round-trips with scalar values; defaults to None."""
    bundle = dataclasses.replace(
        _minimal_training_bundle(),
        extra={"grad_norm": 0.42, "note": "warmup", "stable": True, "restarts": 2},
    )
    path = os.path.join(tmp_path, "training.json")
    write_bundle_file(bundle, path)
    with open(path) as f:
        data = json.load(f)
    assert data["extra"] == {"grad_norm": 0.42, "note": "warmup", "stable": True, "restarts": 2}

    # default is None and serialises to JSON null
    rt = RuntimeBundle(
        run=_run_identity(framework=None, max_iterations=None),
        versions=_versions(),
        hardware=_hardware(),
        runtime=_runtime(),
        resources=_resources(),
    )
    path2 = os.path.join(tmp_path, "runtime.json")
    write_bundle_file(rt, path2)
    with open(path2) as f:
        data2 = json.load(f)
    assert data2["extra"] is None


def test_run_config_presets_round_trip(tmp_path):
    """RunConfig.presets is an open-ended token list; defaults to [] and round-trips."""
    assert RunConfig(physics_backend="physx").presets == []
    cfg = RunConfig(physics_backend="newton_mjwarp", rendering_backend="ovrtx", presets=["rgb", "ovrtx_renderer"])
    base = _minimal_training_bundle()
    bundle = dataclasses.replace(base, run=dataclasses.replace(base.run, config=cfg))
    path = os.path.join(tmp_path, "training.json")
    write_bundle_file(bundle, path)
    with open(path) as f:
        data = json.load(f)
    assert data["run"]["config"]["presets"] == ["rgb", "ovrtx_renderer"]
    assert "sensor_dtype" not in data["run"]["config"]
    assert "sensor_resolution" not in data["run"]["config"]
