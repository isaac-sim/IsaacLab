# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for metrics formatters."""

import json
import os
import re
from datetime import datetime

import pytest

from isaaclab.benchmark import formatters
from isaaclab.benchmark.measurements import SingleMeasurement, StringMetadata
from isaaclab.benchmark.measurements import TestPhase as Phase

pytestmark = pytest.mark.benchmark


def test_default_output_filenames_are_unique_with_identical_timestamps(monkeypatch) -> None:
    class FixedDatetime:
        @classmethod
        def now(cls) -> datetime:
            return datetime(2026, 7, 27, 14, 41, 22, 123456)

    monkeypatch.setattr(formatters, "datetime", FixedDatetime)

    first = formatters.get_default_output_filename("benchmark")
    second = formatters.get_default_output_filename("benchmark")

    assert first != second
    assert re.fullmatch(r"benchmark_2026-07-27_14-41-22-123456_[0-9a-f]{8}", first)


def test_schema_bundle_file_serializes_bundle_and_rejects_missing_bundle(tmp_path, runtime_bundle):
    formatter = formatters.MetricsFormatter.get_instance("schema")
    phase = Phase(phase_name="runtime")
    phase.measurements.append(SingleMeasurement(name="Test FPS", value=60.0, unit="FPS"))
    formatter.add_metrics(phase)
    formatter.finalize(str(tmp_path), "runtime", bundle=runtime_bundle)

    with open(os.path.join(str(tmp_path), "runtime.json")) as f:
        data = json.load(f)
    assert isinstance(formatter, formatters.SchemaBundleFile)
    assert data["run"]["task"] == "Isaac-Ant-Direct-v0"
    assert data["run"]["framework"] is None
    assert data["runtime"]["total_fps"]["mean"] == pytest.approx(100.0)
    assert data["resources"]["gpu_mem_gb"]["peak"] == pytest.approx(12.0)
    assert data["schema_version"]
    assert "Test FPS" not in json.dumps(data)

    with pytest.raises(RuntimeError, match="requires a benchmark bundle"):
        formatter.finalize(str(tmp_path), "missing", bundle=None)
    assert not os.path.exists(os.path.join(str(tmp_path), "missing.json"))


@pytest.mark.parametrize("formatter_cls", [formatters.OsmoKPIFile, formatters.OmniPerfKPIFile])
def test_kpi_formatters_clear_phases_after_finalize(tmp_path, formatter_cls):
    formatter = formatter_cls()
    phase = Phase(phase_name="runtime")
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
        phase = Phase(phase_name=phase_name)
        phase.metadata.append(StringMetadata(name="phase", data=phase_name))
        phase.measurements.append(SingleMeasurement(name="FPS", value=value, unit="FPS"))
        formatter.add_metrics(phase)

    formatter.finalize(str(tmp_path), "metrics")

    assert sorted(path.name for path in tmp_path.glob("*.json")) == ["metrics_runtime.json", "metrics_startup.json"]
    with open(tmp_path / "metrics_runtime.json") as f:
        assert json.load(f)["FPS"] == 60.0
