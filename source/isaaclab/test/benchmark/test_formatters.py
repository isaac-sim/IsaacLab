# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for metrics formatters."""

import json
import logging
import os

import pytest

from isaaclab.test.benchmark import formatters
from isaaclab.test.benchmark.measurements import SingleMeasurement, StringMetadata, TestPhase
from isaaclab.test.benchmark.schema import (
    GpuDeviceInfo,
    Hardware,
    MeanStd,
    Resources,
    RunConfig,
    RunIdentity,
    Runtime,
    RuntimeBundle,
    StartupTime,
    Versions,
)


def _minimal_runtime_bundle() -> RuntimeBundle:
    return RuntimeBundle(
        run=RunIdentity(
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
        ),
        versions=Versions(
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
        ),
        hardware=Hardware(
            hostname="benchmark-host",
            gpu_devices=[GpuDeviceInfo(name="NVIDIA H100 80GB", mem_gb=80.0, compute_cap="9.0")],
            cpu_name="AMD EPYC 7763",
            cpu_count=64,
            ram_gb=512.0,
        ),
        runtime=Runtime(
            startup_time_s=StartupTime(app_launch=1.0, env_creation=2.0, first_step=0.5),
            iterations_completed=1,
            total_wall_time_s=4.0,
            steps_per_iteration=24,
            iteration_time_s=MeanStd(mean=1.0, std=0.0),
            collection_fps=MeanStd(mean=100.0, std=0.0),
            total_fps=MeanStd(mean=100.0, std=0.0),
            iterations_per_s=MeanStd(mean=1.0, std=0.0),
        ),
        resources=Resources(
            gpu_util_pct=MeanStd(mean=80.0, std=5.0),
            gpu_mem_gb=MeanStd(mean=10.0, std=0.5, peak=12.0),
            cpu_util_pct=MeanStd(mean=30.0, std=4.0),
            ram_gb=MeanStd(mean=20.0, std=1.0, peak=24.0),
        ),
    )


@pytest.fixture(autouse=True)
def reset_formatters():
    formatters.MetricsFormatter.reset_instances()
    yield
    formatters.MetricsFormatter.reset_instances()


def test_schema_bundle_file_serializes_bundle_and_handles_missing_bundle(tmp_path, caplog):
    formatter = formatters.MetricsFormatter.get_instance("schema")
    phase = TestPhase(phase_name="runtime")
    phase.measurements.append(SingleMeasurement(name="Test FPS", value=60.0, unit="FPS"))
    formatter.add_metrics(phase)
    formatter.finalize(str(tmp_path), "runtime", bundle=_minimal_runtime_bundle())

    with open(os.path.join(str(tmp_path), "runtime.json")) as f:
        data = json.load(f)
    assert isinstance(formatter, formatters.SchemaBundleFile)
    assert data["run"]["task"] == "Isaac-Ant-Direct-v0"
    assert data["run"]["framework"] is None
    assert data["runtime"]["total_fps"]["mean"] == pytest.approx(100.0)
    assert data["resources"]["gpu_mem_gb"]["peak"] == pytest.approx(12.0)
    assert data["schema_version"]
    assert "Test FPS" not in json.dumps(data)

    with caplog.at_level(logging.WARNING, logger="isaaclab.test.benchmark.formatters"):
        formatter.finalize(str(tmp_path), "missing", bundle=None)
    assert not os.path.exists(os.path.join(str(tmp_path), "missing.json"))
    assert any("no bundle" in record.message.lower() for record in caplog.records)


@pytest.mark.parametrize("formatter_cls", [formatters.OsmoKPIFile, formatters.OmniPerfKPIFile])
def test_kpi_formatters_clear_phases_after_finalize(tmp_path, formatter_cls):
    formatter = formatter_cls()
    phase = TestPhase(phase_name="runtime")
    phase.metadata.append(StringMetadata(name="phase", data="runtime"))
    phase.measurements.append(SingleMeasurement(name="FPS", value=60.0, unit="FPS"))

    formatter.add_metrics(phase)
    formatter.finalize(str(tmp_path), "first")
    formatter.finalize(str(tmp_path), "second")

    assert os.path.exists(os.path.join(str(tmp_path), "first.json"))
    assert not os.path.exists(os.path.join(str(tmp_path), "second.json"))


def test_osmo_writes_one_file_per_phase(tmp_path):
    formatter = formatters.OsmoKPIFile()
    for phase_name, value in (("startup", 1.0), ("runtime", 60.0)):
        phase = TestPhase(phase_name=phase_name)
        phase.metadata.append(StringMetadata(name="phase", data=phase_name))
        phase.measurements.append(SingleMeasurement(name="FPS", value=value, unit="FPS"))
        formatter.add_metrics(phase)

    formatter.finalize(str(tmp_path), "metrics")

    assert sorted(path.name for path in tmp_path.glob("*.json")) == ["metrics_runtime.json", "metrics_startup.json"]
    with open(tmp_path / "metrics_runtime.json") as f:
        assert json.load(f)["FPS"] == 60.0
