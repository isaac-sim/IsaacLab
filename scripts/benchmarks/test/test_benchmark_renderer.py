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
import re
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


def test_render_scope_matches_benchmark_wrapper(benchmark_renderer):
    """The script's timer name must match the one the renderer prints, or nothing is parsed."""
    from isaaclab.benchmark.stepping import RENDER_PROFILE_SCOPE

    assert benchmark_renderer.RENDER_SCOPE == RENDER_PROFILE_SCOPE


def test_physics_scope_matches_benchmark_wrapper(benchmark_renderer):
    """The script's physics timer name must match the one the benchmark wrapper prints."""
    from isaaclab.benchmark.stepping import PHYSICS_PROFILE_SCOPE

    assert benchmark_renderer.PHYSICS_SCOPE == PHYSICS_PROFILE_SCOPE


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
_PHYSICS_SCOPE = "IsaacLab::Physics::step"


def _write_log(path: Path, timings_ms: list[float], physics_ms: float | None = None, steps: int = 1) -> None:
    """Write a synthetic run log with one ``wp.ScopedTimer`` print line per timing.

    Interleaves unrelated lines to mimic real subprocess output (warp init banner, other timers),
    so the parser is exercised against noise rather than a file with only matching lines.

    Args:
        path: File to write.
        timings_ms: One render time per frame [ms].
        physics_ms: Time per physics step [ms], or ``None`` to omit physics lines entirely, as a
            run captured without ``ISAACLAB_PHYSICS_PROFILE`` would.
        steps: Physics steps logged before each render, mimicking the task's decimation.
    """
    lines = ["Warp 1.17.0 initialized:", "SomeOtherScope took 0.10 ms"]
    for value in timings_ms:
        if physics_ms is not None:
            lines.extend([f"{_PHYSICS_SCOPE} took {physics_ms:.2f} ms"] * steps)
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


def test_parse_log_sums_physics_steps_within_one_frame(benchmark_renderer, tmp_path):
    """Decimation logs several physics steps per render, so they sum into the frame they precede.

    Counting them as frames instead would report a physics median of one step and shift the render
    samples the padding skips past.
    """
    log_path = tmp_path / "profile.log"
    timings = [2.0] * (benchmark_renderer.FRAME_PADDING + 3)
    _write_log(log_path, timings, physics_ms=1.0, steps=3)

    results = benchmark_renderer.parse_log(str(log_path), num_frames=3)

    assert results["size"] == 3
    assert results["median"] == pytest.approx(2.0)
    assert results["physics"]["median"] == pytest.approx(3.0)
    assert results["total"]["median"] == pytest.approx(2.0 + 3.0)


def test_parse_log_reads_an_absent_scope_as_zero(benchmark_renderer, tmp_path):
    """A run captured without ``ISAACLAB_PHYSICS_PROFILE`` still parses, with physics reading zero."""
    log_path = tmp_path / "profile.log"
    _write_log(log_path, [2.0] * (benchmark_renderer.FRAME_PADDING + 3))

    results = benchmark_renderer.parse_log(str(log_path), num_frames=3)

    assert results["physics"]["median"] == pytest.approx(0.0)
    assert results["total"]["median"] == pytest.approx(2.0)


def test_parse_log_takes_an_arbitrary_scope_mapping(benchmark_renderer, tmp_path):
    """Scopes are data, so a caller can summarize a timer the script does not know about."""
    log_path = tmp_path / "profile.log"
    lines = ["Custom::scope took 4.00 ms", f"{_RENDER_SCOPE} took 2.00 ms"] * (benchmark_renderer.FRAME_PADDING + 3)
    log_path.write_text("\n".join(lines) + "\n")

    results = benchmark_renderer.parse_log(
        str(log_path),
        num_frames=3,
        scopes={
            benchmark_renderer.FRAME_SCOPE: benchmark_renderer.RENDER_SCOPE_PATTERN,
            "custom": re.compile(r"Custom::scope took ([\d.]+) ms"),
        },
    )

    assert results["custom"]["median"] == pytest.approx(4.0)
    assert "physics" not in results


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
