# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for BaseIsaacLabBenchmark."""

import json
import os
from dataclasses import replace

import pytest

from isaaclab.benchmark import benchmark_core as benchmark_core_module
from isaaclab.benchmark import formatters
from isaaclab.benchmark.benchmark_core import BaseIsaacLabBenchmark, _runtime_measurements
from isaaclab.benchmark.measurements import SingleMeasurement, StringMetadata
from isaaclab.benchmark.schema import EnvironmentStepTiming, MeanStd
from isaaclab.sim import utils as sim_utils

pytestmark = pytest.mark.benchmark


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
    benchmark.finalize()

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
    assert benchmark._manual_recorders["CPUInfo"].get_runtime_data()["cpu_utilization"]["n"] == 1
    assert benchmark._manual_recorders["MemoryInfo"].get_runtime_data()["memory_utilization"]["rss_n"] == 1

    benchmark.finalize()
    with open(benchmark.output_file_path) as f:
        data = json.load(f)
    assert data["runtime"]["execution_time"] == 100.5
    assert data["runtime"]["System Memory RSS n"] == 1
    assert data["hardware_info"]["physical_cores"] > 0
    assert "torch_version" in data["version_info"]
    assert benchmark._manual_recorders is None
    assert benchmark._frametime_recorders is None


def test_benchmark_skips_frametime_recorders_without_kit(monkeypatch, tmp_path, caplog):
    """Frametime recorders are optional when the simulation runs without Kit."""

    def fail_if_called(_extension_name: str) -> None:
        raise AssertionError("enable_extension should not be called without Kit")

    monkeypatch.setattr(benchmark_core_module, "has_kit", lambda: False)
    monkeypatch.setattr(sim_utils, "enable_extension", fail_if_called)

    with caplog.at_level("WARNING"):
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="kitless",
            formatter_type="omniperf",
            output_path=str(tmp_path),
            use_recorders=True,
            frametime_recorders=True,
        )

    assert benchmark._frametime_recorders == {}
    assert "Kit is not running" in caplog.text


def test_benchmark_skips_frametime_recorders_if_kit_app_stops(monkeypatch, tmp_path, caplog):
    """A disappearing Kit app does not fail optional frametime recorder setup."""

    def raise_missing_kit_app(_extension_name: str) -> None:
        raise RuntimeError("Failed to acquire interface: omni::kit::IApp")

    monkeypatch.setattr(benchmark_core_module, "has_kit", lambda: True)
    monkeypatch.setattr(sim_utils, "enable_extension", raise_missing_kit_app)

    with caplog.at_level("WARNING"):
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="kitless",
            formatter_type="omniperf",
            output_path=str(tmp_path),
            use_recorders=True,
            frametime_recorders=True,
        )

    assert benchmark._frametime_recorders == {}
    assert "Could not initialize Kit frametime recorders" in caplog.text


def test_formatter_selection_and_output_filenames(tmp_path, runtime_bundle):
    default_benchmark = BaseIsaacLabBenchmark(
        "default", formatter_type="   ", output_path=str(tmp_path), use_recorders=False
    )
    assert _formatter_keys(default_benchmark) == ["omniperf"]

    single = BaseIsaacLabBenchmark("single", formatter_type="json", output_path=str(tmp_path), use_recorders=False)
    single.add_measurement("runtime", measurement=SingleMeasurement(name="execution_time", value=100.5, unit="ms"))
    single_paths = single.finalize()
    assert os.path.exists(os.path.join(str(tmp_path), f"{single.output_prefix}.json"))
    assert not os.path.exists(os.path.join(str(tmp_path), f"{single.output_prefix}_json.json"))
    assert single_paths == (tmp_path / f"{single.output_prefix}.json",)

    multi = BaseIsaacLabBenchmark(
        "multi",
        formatter_type="schema,json,json",
        output_path=str(tmp_path),
        use_recorders=False,
        output_prefix="test",
    )
    assert _formatter_keys(multi) == ["schema", "json"]
    multi.attach_bundle(runtime_bundle)
    multi.add_measurement("runtime", measurement=SingleMeasurement(name="execution_time", value=100.5, unit="ms"))
    multi_paths = multi.finalize()

    schema_path = os.path.join(str(tmp_path), f"{multi.output_prefix}_schema.json")
    json_path = os.path.join(str(tmp_path), f"{multi.output_prefix}_json.json")
    assert os.path.exists(schema_path)
    assert os.path.exists(json_path)
    assert set(multi_paths) == {
        tmp_path / f"{multi.output_prefix}_schema.json",
        tmp_path / f"{multi.output_prefix}_json.json",
    }

    with open(schema_path) as f:
        schema_data = json.load(f)
    with open(json_path) as f:
        json_data = json.load(f)
    assert schema_data != json_data
    assert schema_data["run"]["task"] == "Isaac-Ant-Direct-v0"


def test_attached_bundles_are_projected_to_flat_formatters(
    tmp_path, runtime_bundle, training_bundle, play_bundle, startup_bundle
):
    cases = [
        (runtime_bundle, "runtime", "Mean Total FPS", 100.0),
        (training_bundle, "train", "Last Reward", 3.0),
        (play_bundle, "play", "Mean Reward", 4.0),
        (startup_bundle, "python_imports", "Wall Clock Time", 0.25),
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
        benchmark.finalize()

        with open(benchmark.output_file_path) as f:
            data = json.load(f)
        assert data[phase][metric] == expected


def test_environment_step_timing_flat_labels_describe_measurement_mode(runtime, serialized_step_timing):
    host_return = EnvironmentStepTiming(
        environment_step_time_s=MeanStd(mean=0.08, std=0.01, peak=0.1),
        environment_step_fps=MeanStd(mean=200.0, std=2.0, peak=205.0),
        simulation_step_time_s=None,
        outside_simulation_step_time_s=None,
        outside_simulation_step_fraction=None,
        environment_step_calls=100,
        simulation_step_calls=None,
        measurement_mode="host_return",
    )

    host_names = {
        measurement.name
        for measurement in _runtime_measurements(replace(runtime, environment_step_timing=host_return))["runtime"]
    }
    synchronized_names = {
        measurement.name
        for measurement in _runtime_measurements(replace(runtime, environment_step_timing=serialized_step_timing))[
            "runtime"
        ]
    }

    assert "Mean Environment Step Host-Return FPS" in host_names
    assert "Mean Serialized Synchronized Environment Step FPS" in synchronized_names
    assert "Mean Total FPS" in host_names
    assert "Mean Total FPS" not in synchronized_names
    assert {
        "Serialized Diagnostic Total Wall Time",
        "Mean Serialized Diagnostic Iteration Time",
        "Mean Serialized Diagnostic Collection FPS",
        "Mean Serialized Diagnostic Total FPS",
        "Mean Serialized Diagnostic Iterations per Second",
    } <= synchronized_names
    synchronized_breakdown_names = {
        "Mean Synchronized Simulation Time per Environment Step",
        "Mean Outside Simulation Time per Environment Step",
        "Outside Simulation Step Fraction",
    }
    assert synchronized_breakdown_names <= synchronized_names
    assert host_names.isdisjoint(synchronized_breakdown_names)


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


def test_schema_benchmark_without_bundle_fails_before_writing(tmp_path):
    benchmark = BaseIsaacLabBenchmark(
        "missing_bundle",
        formatter_type="schema",
        output_path=str(tmp_path),
        use_recorders=False,
        output_prefix="missing_bundle",
    )

    with pytest.raises(RuntimeError, match="requires an attached benchmark bundle"):
        benchmark.finalize()

    assert not (tmp_path / f"{benchmark.output_prefix}.json").exists()
