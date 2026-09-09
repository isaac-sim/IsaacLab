# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the pure helpers in ``scripts/benchmarks/benchmark_renderer.py``.

The script drives ``nsys`` and a rendering backend end-to-end, so those paths are not covered
here. These tests exercise the record-building, NVTX-parsing, and CLI-parsing logic that does not
require a GPU or profiling tools.
"""

import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "scripts" / "benchmarks" / "benchmark_renderer.py"


def _load_module():
    """Import ``benchmark_renderer.py`` as a module, bypassing its CLI entry point."""
    spec = importlib.util.spec_from_file_location("benchmark_renderer", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def benchmark_renderer():
    return _load_module()


def test_render_scope_matches_render_context(benchmark_renderer):
    """The script's NVTX scope name must match the one the renderer emits, or nothing is parsed."""
    from isaaclab.renderers.render_context import RENDER_PROFILE_SCOPE

    assert benchmark_renderer.RENDER_SCOPE == RENDER_PROFILE_SCOPE


def test_pixels_per_second(benchmark_renderer):
    """Throughput is env count times tile area, scaled from milliseconds to seconds."""
    assert benchmark_renderer.pixels_per_second(median_ms=1000.0, num_envs=4, resolution=256) == pytest.approx(
        4 * 256 * 256
    )


def test_build_record_ok(benchmark_renderer):
    profile = {"name": "p", "preset": "newton_renderer,rgb", "settings": {"tlas": "sah"}}
    results = {"size": 20, "median": 2.0, "mean": 2.1, "min": 1.5, "max": 3.0, "stdev": 0.2}

    record = benchmark_renderer.build_record(profile, results, num_envs=4, resolution=256)

    assert record["status"] == "ok"
    assert record["size"] == 20
    assert record["median_ms"] == 2.0
    assert record["pixels_per_second"] == pytest.approx(benchmark_renderer.pixels_per_second(2.0, 4, 256))


def test_build_record_failed(benchmark_renderer):
    profile = {"name": "p", "preset": "newton_renderer,rgb", "settings": {"tlas": "sah"}}

    record = benchmark_renderer.build_record(profile, None, num_envs=4, resolution=256)

    assert record == {
        "name": "p",
        "preset": "newton_renderer,rgb",
        "settings": {"tlas": "sah"},
        "status": "failed",
        "log": str(Path(benchmark_renderer.OUTPUT_PATH) / "p.log"),
    }


def _write_nvtx_report(path: Path, ranges: list[tuple[int, int]], use_string_ids: bool) -> None:
    """Write a minimal nsys-export-shaped SQLite database with the given NVTX ranges."""
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        if use_string_ids:
            connection.execute("INSERT INTO StringIds VALUES (1, ?)", (benchmark_renderer_scope,))
            connection.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER)")
            for start, end in ranges:
                connection.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, NULL, 1)", (start, end))
        else:
            connection.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER)")
            for start, end in ranges:
                connection.execute(
                    "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL)", (start, end, benchmark_renderer_scope)
                )
        connection.commit()


benchmark_renderer_scope = "IsaacLab::Renderer::render"


@pytest.mark.parametrize("use_string_ids", [True, False])
def test_nvtx_ranges_matches_inline_and_interned_labels(benchmark_renderer, tmp_path, use_string_ids):
    """A range's label may live inline in ``text`` or be interned via ``StringIds``; both resolve."""
    db_path = tmp_path / "report.sqlite"
    _write_nvtx_report(db_path, [(0, 10), (20, 25)], use_string_ids)

    with sqlite3.connect(db_path) as connection:
        ranges = benchmark_renderer.nvtx_ranges(connection, benchmark_renderer_scope)

    assert ranges == [(0, 10), (20, 25)]


def test_parse_profile_skips_padding_and_keeps_num_frames(benchmark_renderer, tmp_path):
    """Only the ``num_frames`` ranges after :data:`FRAME_PADDING` warm-up frames are summarized."""
    db_path = tmp_path / "report.sqlite"
    padding = benchmark_renderer.FRAME_PADDING
    # padding warm-up frames of 1ms each, then 3 measured frames of 2ms each, then 1 extra frame that must be dropped.
    ranges = [(i * 1_000_000, i * 1_000_000 + 1_000_000) for i in range(padding)]
    measured_start = padding * 1_000_000
    ranges += [(measured_start + i * 2_000_000, measured_start + i * 2_000_000 + 2_000_000) for i in range(4)]
    _write_nvtx_report(db_path, ranges, use_string_ids=False)

    results = benchmark_renderer.parse_profile(str(db_path), num_frames=3)

    assert results["size"] == 3
    assert results["median"] == pytest.approx(2.0)
    assert results["min"] == pytest.approx(2.0)
    assert results["max"] == pytest.approx(2.0)


def test_parse_profile_returns_none_without_matching_ranges(benchmark_renderer, tmp_path):
    """A report with no ``RENDER_SCOPE`` ranges means profiling was never enabled for that run."""
    db_path = tmp_path / "report.sqlite"
    _write_nvtx_report(db_path, [], use_string_ids=False)

    assert benchmark_renderer.parse_profile(str(db_path), num_frames=3) is None


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args], capture_output=True, text=True, cwd=ROOT, timeout=30
    )


def test_cli_lists_available_profiles_as_json(benchmark_renderer):
    """With no profile selected, the CLI reports every declared profile name and exits cleanly."""
    result = _run_cli(["--json"])

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["available_profiles"] == [profile["name"] for profile in benchmark_renderer.PROFILES]


def test_cli_rejects_unmatched_profile_glob():
    """An unmatched profile pattern fails fast, before any profiling subprocess is launched."""
    result = _run_cli(["does-not-exist-*"])

    assert result.returncode == 1
    assert "No profile found matching: does-not-exist-*" in result.stderr
