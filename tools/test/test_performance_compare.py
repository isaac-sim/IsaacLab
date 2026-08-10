# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for performance comparison against an authoritative baseline table."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tools import performance_compare


@pytest.fixture
def runtime_bundle() -> dict:
    return {
        "schema_version": "1.3",
        "run": {
            "task": "Isaac-Cartpole-Direct",
            "config": {"physics_backend": "physx", "rendering_backend": "none", "presets": []},
            "num_envs": 4096,
        },
        "versions": {"isaacsim": "6.1.0-test"},
        "hardware": {
            "gpu_devices": [{"name": "NVIDIA L40S", "mem_gb": 44.0, "compute_cap": "8.9"}],
        },
        "runtime": {"total_fps": {"mean": 95.0, "std": 1.0, "peak": 98.0}},
        "resources": {
            "gpu_mem_gb": {"mean": 9.0, "std": 0.5, "peak": 11.0},
            "ram_gb": {"mean": 7.0, "std": 0.2, "peak": 8.0},
        },
    }


def _baseline_table() -> dict:
    metric = {"warn_regression_pct": 5.0, "fail_regression_pct": 10.0}
    return {
        "schema_version": 1,
        "baselines": [
            {
                "task": "Isaac-Cartpole-Direct",
                "physics_backend": "physx",
                "rendering_backend": "none",
                "presets": [],
                "gpu_model": "l40s",
                "isaacsim_version": "6.1.0-test",
                "num_envs": 4096,
                "metrics": {
                    "total_fps": {"reference": 100.0, **metric},
                    "gpu_mem_peak_gb": {"reference": 10.0, **metric},
                    "ram_peak_gb": {"reference": 8.0, **metric},
                },
            }
        ],
    }


def test_compare_uses_opposite_regression_directions_for_throughput_and_memory(runtime_bundle: dict) -> None:
    report = performance_compare.compare(runtime_bundle, _baseline_table())

    assert report.verdict == "FAIL"
    assert [(metric.name, metric.regression_pct, metric.verdict) for metric in report.metrics] == [
        ("total_fps", 5.0, "WARN"),
        ("gpu_mem_peak_gb", 10.0, "FAIL"),
        ("ram_peak_gb", 0.0, "PASS"),
    ]


def test_compare_applies_warning_and_failure_boundaries(runtime_bundle: dict) -> None:
    runtime_bundle["runtime"]["total_fps"]["mean"] = 90.0
    runtime_bundle["resources"]["gpu_mem_gb"]["peak"] = 10.5
    runtime_bundle["resources"]["ram_gb"]["peak"] = 7.0

    report = performance_compare.compare(runtime_bundle, _baseline_table())

    assert [metric.verdict for metric in report.metrics] == ["FAIL", "WARN", "PASS"]


def test_compare_rejects_incomplete_or_ambiguous_baselines(runtime_bundle: dict) -> None:
    table = _baseline_table()
    table["baselines"][0]["task"] = "Another-Task"
    with pytest.raises(performance_compare.ComparisonError, match="No baseline row"):
        performance_compare.compare(runtime_bundle, table)

    table = _baseline_table()
    table["baselines"].append(dict(table["baselines"][0]))
    with pytest.raises(performance_compare.ComparisonError, match="Multiple baseline rows"):
        performance_compare.compare(runtime_bundle, table)

    table = _baseline_table()
    del table["baselines"][0]["metrics"]["ram_peak_gb"]
    with pytest.raises(performance_compare.ComparisonError, match="missing required metrics: ram_peak_gb"):
        performance_compare.compare(runtime_bundle, table)

    table = _baseline_table()
    table["baselines"][0]["metrics"]["total_fps"]["reference"] = float("inf")
    with pytest.raises(performance_compare.ComparisonError, match="must be finite"):
        performance_compare.compare(runtime_bundle, table)

    table = _baseline_table()
    table["baselines"][0]["metrics"]["total_fps"]["reference"] = 10**400
    with pytest.raises(performance_compare.ComparisonError, match="must be finite"):
        performance_compare.compare(runtime_bundle, table)


def test_write_outputs_reports_all_metrics(runtime_bundle: dict, tmp_path: Path) -> None:
    report = performance_compare.compare(runtime_bundle, _baseline_table())
    markdown = tmp_path / "comparison.md"
    output_json = tmp_path / "comparison.json"
    junit = tmp_path / "comparison.xml"

    performance_compare.write_outputs(report, markdown, output_json, junit)

    markdown_text = markdown.read_text()
    assert "| Total FPS | 95" in markdown_text
    assert "| Warn | Fail |" in markdown_text
    output = json.loads(output_json.read_text())
    assert output["verdict"] == "FAIL"
    assert output["metrics"][0]["warn_regression_pct"] == 5.0
    assert output["metrics"][0]["fail_regression_pct"] == 10.0
    suite = ET.parse(junit).getroot()
    assert suite.attrib == {"name": "performance-comparison", "tests": "3", "failures": "1", "skipped": "0"}
    assert [case.attrib["name"] for case in suite.findall("testcase")] == [
        "total_fps",
        "gpu_mem_peak_gb",
        "ram_peak_gb",
    ]


def test_main_skips_only_when_baseline_is_not_configured(runtime_bundle: dict, tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(runtime_bundle))
    common_args = [
        "--benchmark_result",
        str(benchmark_path),
        "--markdown",
        str(tmp_path / "comparison.md"),
        "--output_json",
        str(tmp_path / "comparison.json"),
        "--junit",
        str(tmp_path / "comparison.xml"),
    ]

    assert performance_compare.main(common_args) == 0
    assert json.loads((tmp_path / "comparison.json").read_text())["verdict"] == "SKIP"

    malformed_baseline = tmp_path / "malformed.json"
    malformed_baseline.write_text("{}")
    assert performance_compare.main([*common_args, "--baseline", str(malformed_baseline)]) == 1
    assert json.loads((tmp_path / "comparison.json").read_text())["verdict"] == "ERROR"
