# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the pure helpers in ``scripts/benchmarks/benchmark_renderer.py``.

The script drives a rendering backend end-to-end, so that path is not covered here. These tests
exercise the record-building, log-parsing, and CLI-parsing logic that does not require a GPU.
"""

import importlib.util
import json
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
    """The script's timer name must match the one the renderer prints, or nothing is parsed."""
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


_RENDER_SCOPE = "IsaacLab::Renderer::render"


def _write_log(path: Path, timings_ms: list[float]) -> None:
    """Write a synthetic run log with one ``wp.ScopedTimer`` print line per timing.

    Interleaves unrelated lines to mimic real subprocess output (warp init banner, other timers),
    so the parser is exercised against noise rather than a file with only matching lines.
    """
    lines = ["Warp 1.17.0 initialized:", "SomeOtherScope took 0.10 ms"]
    for value in timings_ms:
        lines.append(f"{_RENDER_SCOPE} took {value:.2f} ms")
    path.write_text("\n".join(lines) + "\n")


def test_parse_log_skips_padding_and_keeps_num_frames(benchmark_renderer, tmp_path):
    """Only the ``num_frames`` timings after :data:`FRAME_PADDING` warm-up frames are summarized."""
    log_path = tmp_path / "profile.log"
    padding = benchmark_renderer.FRAME_PADDING
    # padding warm-up frames of 1ms each, then 4 measured frames of 2ms each (only 3 are kept).
    timings = [1.0] * padding + [2.0] * 4
    _write_log(log_path, timings)

    results = benchmark_renderer.parse_log(str(log_path), num_frames=3)

    assert results["size"] == 3
    assert results["median"] == pytest.approx(2.0)
    assert results["min"] == pytest.approx(2.0)
    assert results["max"] == pytest.approx(2.0)


def test_parse_log_returns_none_without_matching_lines(benchmark_renderer, tmp_path):
    """A log with no ``RENDER_SCOPE`` timings means profiling was never enabled for that run."""
    log_path = tmp_path / "profile.log"
    _write_log(log_path, [])

    assert benchmark_renderer.parse_log(str(log_path), num_frames=3) is None


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
