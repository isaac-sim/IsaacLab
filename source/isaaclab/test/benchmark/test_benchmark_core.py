# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for BaseIsaacLabBenchmark."""

import json
import os
from dataclasses import replace

import pytest

from isaaclab.test.benchmark import formatters
from isaaclab.test.benchmark.benchmark_core import BaseIsaacLabBenchmark
from isaaclab.test.benchmark.measurements import SingleMeasurement, StringMetadata

pytestmark = pytest.mark.benchmark

# ==============================================================================
# BaseIsaacLabBenchmark Tests
# ==============================================================================


@pytest.fixture(autouse=True)
def reset_formatters():
    formatters.MetricsFormatter.reset_instances()
    yield
    formatters.MetricsFormatter.reset_instances()


def _minimal_runtime_bundle():
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


def _minimal_training_bundle():
    from isaaclab.test.benchmark.schema import Learning, LearningCurve, TrainingBundle

    bundle = _minimal_runtime_bundle()
    return TrainingBundle(
        run=replace(bundle.run, framework="rsl_rl", max_iterations=1),
        versions=bundle.versions,
        hardware=bundle.hardware,
        runtime=bundle.runtime,
        resources=bundle.resources,
        learning=Learning(
            ema_alpha=0.95,
            reward=LearningCurve(final_raw=3.0, final_ema=2.5, series_per_iter=[1.0, 3.0]),
            ep_length=LearningCurve(final_raw=20.0, final_ema=18.0, series_per_iter=[10.0, 20.0]),
        ),
        success_rate=0.75,
    )


def _minimal_startup_bundle():
    from isaaclab.test.benchmark.schema import CProfileFunction, StartupBundle, StartupConfig, StartupPhase

    bundle = _minimal_runtime_bundle()
    return StartupBundle(
        run=replace(bundle.run, num_envs=None),
        versions=bundle.versions,
        hardware=bundle.hardware,
        phases={
            "python_imports": StartupPhase(
                total_time_s=0.25,
                top_functions=[
                    CProfileFunction(
                        name="isaaclab_tasks.utils:import_packages", own_time_s=0.1, cum_time_s=0.2, calls=2
                    )
                ],
            )
        },
        config=StartupConfig(top_n=1, whitelist=None),
    )


def _formatter_keys(benchmark: BaseIsaacLabBenchmark) -> list[str]:
    return [key for key, _ in benchmark._metrics]


def test_benchmark_collects_metadata_measurements_and_writes_json(tmp_path):
    output_path = tmp_path / "nested" / "output"
    benchmark = BaseIsaacLabBenchmark(
        benchmark_name="my_workflow",
        formatter_type="omniperf",
        output_path=str(output_path),
        use_recorders=False,
        output_prefix="test",
    )

    benchmark.add_measurement(
        "runtime",
        measurement=[
            SingleMeasurement(name="metric1", value=10.0, unit="ms"),
            SingleMeasurement(name="metric2", value=20.0, unit="ms"),
        ],
    )
    benchmark.add_measurement("runtime", metadata=StringMetadata(name="custom", data="value"))
    benchmark._finalize_impl()

    with open(benchmark.output_file_path) as f:
        data = json.load(f)

    assert output_path.exists()
    assert benchmark.benchmark_name == "my_workflow"
    assert benchmark._use_recorders is False
    assert not hasattr(benchmark, "_manual_recorders") or benchmark._manual_recorders is None
    assert data["benchmark_info"]["workflow_name"] == "my_workflow"
    assert "timestamp" in data["benchmark_info"]
    assert data["runtime"]["metric1"] == 10.0
    assert data["runtime"]["metric2"] == 20.0
    assert data["runtime"]["custom"] == "value"


def test_benchmark_updates_recorders_and_cleans_up(tmp_path):
    benchmark = BaseIsaacLabBenchmark(
        benchmark_name="test_benchmark",
        formatter_type="omniperf",
        output_path=str(tmp_path),
        use_recorders=True,
        output_prefix="test",
    )

    benchmark.add_measurement("runtime", measurement=SingleMeasurement(name="execution_time", value=100.5, unit="ms"))
    benchmark.update_manual_recorders()
    assert benchmark._manual_recorders["CPUInfo"]._n >= 1
    assert benchmark._manual_recorders["MemoryInfo"]._rss_n >= 1

    benchmark._finalize_impl()
    assert os.path.exists(benchmark.output_file_path)
    assert benchmark._manual_recorders is None
    assert benchmark._frametime_recorders is None


def test_formatter_selection_and_output_filenames(tmp_path):
    default_benchmark = BaseIsaacLabBenchmark(
        "default", formatter_type="   ", output_path=str(tmp_path), use_recorders=False
    )
    assert _formatter_keys(default_benchmark) == ["omniperf"]

    single = BaseIsaacLabBenchmark("single", formatter_type="json", output_path=str(tmp_path), use_recorders=False)
    single.add_measurement("runtime", measurement=SingleMeasurement(name="execution_time", value=100.5, unit="ms"))
    single._finalize_impl()
    assert os.path.exists(os.path.join(str(tmp_path), f"{single.output_prefix}.json"))
    assert not os.path.exists(os.path.join(str(tmp_path), f"{single.output_prefix}_json.json"))

    multi = BaseIsaacLabBenchmark(
        "multi",
        formatter_type="schema,json,json",
        output_path=str(tmp_path),
        use_recorders=False,
        output_prefix="test",
    )
    assert _formatter_keys(multi) == ["schema", "json"]
    multi.attach_bundle(_minimal_runtime_bundle())
    multi.add_measurement("runtime", measurement=SingleMeasurement(name="execution_time", value=100.5, unit="ms"))
    multi._finalize_impl()

    schema_path = os.path.join(str(tmp_path), f"{multi.output_prefix}_schema.json")
    json_path = os.path.join(str(tmp_path), f"{multi.output_prefix}_json.json")
    assert os.path.exists(schema_path)
    assert os.path.exists(json_path)

    with open(schema_path) as f:
        schema_data = json.load(f)
    with open(json_path) as f:
        json_data = json.load(f)
    assert schema_data != json_data
    assert schema_data["run"]["task"] == "Isaac-Ant-Direct-v0"


def test_attached_bundles_are_projected_to_flat_formatters(tmp_path):
    cases = [
        (_minimal_runtime_bundle(), "runtime", "Mean Total FPS", 100.0),
        (_minimal_training_bundle(), "train", "Last Reward", 3.0),
        (_minimal_startup_bundle(), "python_imports", "Wall Clock Time", 0.25),
    ]

    for index, (bundle, phase, metric, expected) in enumerate(cases):
        benchmark = BaseIsaacLabBenchmark(
            f"bundle_{index}",
            formatter_type="omniperf",
            output_path=str(tmp_path),
            use_recorders=False,
            output_prefix=f"bundle_{index}",
        )
        benchmark.attach_bundle(bundle)
        benchmark._finalize_impl()

        with open(benchmark.output_file_path) as f:
            data = json.load(f)
        assert data[phase][metric] == expected


def test_metrics_formatter_factory_registration_cache_and_errors():
    expected = {
        "json": formatters.JSONFileMetrics,
        "osmo": formatters.OsmoKPIFile,
        "omniperf": formatters.OmniPerfKPIFile,
        "summary": formatters.SummaryMetrics,
        "schema": formatters.SchemaBundleFile,
    }
    for key, cls in expected.items():
        assert isinstance(formatters.MetricsFormatter.get_instance(key), cls)

    formatter = formatters.MetricsFormatter.get_instance("omniperf")
    assert formatter is formatters.MetricsFormatter.get_instance("omniperf")
    formatters.MetricsFormatter.reset_instances()
    assert formatter is not formatters.MetricsFormatter.get_instance("omniperf")

    with pytest.raises(ValueError, match="Unknown formatter type"):
        formatters.MetricsFormatter.get_instance("invalid_type")
