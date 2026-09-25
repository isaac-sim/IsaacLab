# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the removal of periodic USS sampling from the memory recorder.

``MemoryInfoRecorder`` used to call ``psutil.Process.memory_full_info()`` once per
:class:`~isaaclab.benchmark.BenchmarkMonitor` tick. That call walks the process page
tables, so it perturbed the workload the benchmark was timing. These tests pin the
behaviour that replaced it: the recorder reads only cheap counters, and no USS value
reaches the emitted output.
"""

import pathlib
import threading
import time

import pytest

from isaaclab.benchmark.benchmark_core import BaseIsaacLabBenchmark
from isaaclab.benchmark.benchmark_monitor import BenchmarkMonitor

pytestmark = pytest.mark.benchmark

ENTRYPOINT_DIR = pathlib.Path(__file__).resolve().parents[2] / "isaaclab" / "benchmark" / "entrypoints"

# The nine call sites that drive a BenchmarkMonitor over a timed region.
ENTRYPOINTS = (
    "runtime.py",
    "backends/rl_games/benchmark_train_rl_games.py",
    "backends/rl_games/benchmark_play_rl_games.py",
    "backends/rsl_rl/benchmark_train_rsl_rl.py",
    "backends/rsl_rl/benchmark_play_rsl_rl.py",
    "backends/sb3/benchmark_train_sb3.py",
    "backends/sb3/benchmark_play_sb3.py",
    "backends/skrl/benchmark_train_skrl.py",
    "backends/skrl/benchmark_play_skrl.py",
)


USS_FIELD_NAMES = (
    "System Memory USS",
    "uss_mean",
    "uss_std",
    "uss_peak",
    "uss_n",
)


def _uss_fields(text: str) -> list[str]:
    """Return the USS field names present in serialized output, if any."""
    return [name for name in USS_FIELD_NAMES if name in text]


@pytest.fixture
def no_uss_query(monkeypatch):
    """Make any USS query a hard failure for the duration of a test."""
    import psutil

    def _explode(self):  # noqa: ARG001 — bound method, self is the process
        raise AssertionError("benchmark recorders must not call memory_full_info()")

    monkeypatch.setattr(psutil.Process, "memory_full_info", _explode)


def _benchmark(tmp_path, formatter_type="omniperf"):
    return BaseIsaacLabBenchmark(
        benchmark_name="memory_recorder_regression",
        formatter_type=formatter_type,
        output_path=str(tmp_path),
        use_recorders=True,
        output_prefix="test",
    )


def test_benchmark_monitor_never_queries_uss(tmp_path, no_uss_query):
    """Test that a live BenchmarkMonitor drives the recorders without any USS query."""
    benchmark = _benchmark(tmp_path)
    before = threading.active_count()

    with BenchmarkMonitor(benchmark, interval=0.01):
        time.sleep(0.2)

    # The monitor swallows recorder exceptions, so assert on the monitor's own record
    # rather than trusting that no traceback surfaced.
    assert benchmark._manual_recorders["MemoryInfo"]._rss_n >= 1
    assert threading.active_count() == before, "monitor thread outlived its context"
    benchmark.finalize()


def test_monitor_reports_no_recorder_exception(tmp_path, no_uss_query):
    """Test that the monitor loop completes without capturing a recorder exception."""
    benchmark = _benchmark(tmp_path)
    monitor = BenchmarkMonitor(benchmark, interval=0.01)
    with monitor:
        time.sleep(0.1)
    assert monitor._exception is None
    benchmark.finalize()


def test_finalized_output_contains_no_uss(tmp_path, no_uss_query):
    """Test that the written benchmark output carries RSS but no USS field."""
    benchmark = _benchmark(tmp_path)
    benchmark.update_manual_recorders()
    benchmark.finalize()

    written = pathlib.Path(benchmark.output_file_path).read_text()
    assert "System Memory RSS" in written
    assert not _uss_fields(written)


# "schema" is excluded: it requires an attached benchmark bundle, which is out of scope
# here. The typed schema carries no USS field, so this change cannot affect it.
@pytest.mark.parametrize("formatter_type", ["omniperf", "json", "osmo", "summary"])
def test_supported_formatters_still_serialize(tmp_path, formatter_type, no_uss_query):
    """Test that the measurement-driven formatters still serialize with USS gone."""
    benchmark = _benchmark(tmp_path / formatter_type, formatter_type=formatter_type)
    benchmark.update_manual_recorders()

    paths = benchmark.finalize()

    assert paths, f"{formatter_type} wrote no output"
    for path in paths:
        written = pathlib.Path(path)
        assert written.exists(), f"{formatter_type} reported {written} but it was not written"
        assert not _uss_fields(written.read_text())


@pytest.mark.parametrize("relative_path", ENTRYPOINTS)
def test_entrypoints_keep_monitor_and_recorder_configuration(relative_path):
    """Test that each benchmark entry point still runs a monitor with recorders enabled.

    Exercising these end to end needs Isaac Sim, so this is a configuration guard: it
    fails if a call site loses its monitor or stops enabling recorders.
    """
    source = (ENTRYPOINT_DIR / relative_path).read_text()
    assert "BenchmarkMonitor(" in source
    assert "use_recorders=True" in source
    assert "memory_full_info" not in source
