# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the pure helpers in ``scripts/benchmarks/benchmark_renderer.py``.

The script drives a rendering backend end-to-end, so that path is not covered here. These tests
exercise the record-building, structured timing, and CLI-parsing logic that does not require a GPU.
"""

import importlib.util
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

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


def test_profile_scopes_match_benchmark_wrappers(benchmark_renderer):
    """The script's timer names must match the ones collected by the profiling shims."""
    from isaaclab.benchmark.stepping import PHYSICS_PROFILE_SCOPE, RENDER_PROFILE_SCOPE

    assert benchmark_renderer.RENDER_SCOPE == RENDER_PROFILE_SCOPE
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


def _write_profile(path: Path, timings_ms: list[tuple[str, float]]) -> None:
    """Write ordered scope timings [ms] in the runtime benchmark's structured format."""
    path.write_text(json.dumps({"timings_ms": timings_ms}))


def test_parse_profile_skips_padding_and_keeps_num_frames(benchmark_renderer, tmp_path):
    """Summarize only the requested frames after warm-up, preserving unrounded timings."""
    profile_path = tmp_path / "profile.json"
    padding = benchmark_renderer.FRAME_PADDING
    measured_ms = 2.123456789
    timings = [1.0] * padding + [measured_ms] * 3 + [99.0]
    _write_profile(profile_path, [(_RENDER_SCOPE, value) for value in timings])

    results = benchmark_renderer.parse_profile(str(profile_path), num_frames=3)

    assert results["size"] == 3
    assert results["median"] == measured_ms
    assert results["min"] == measured_ms
    assert results["max"] == measured_ms
    assert results["physics"]["median"] == pytest.approx(0.0)
    assert results["total"]["median"] == measured_ms


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        (None, FileNotFoundError),
        ("{", json.JSONDecodeError),
        ("{}", ValueError),
        (json.dumps({"timings_ms": [(_PHYSICS_SCOPE, 1.0)]}), None),
        (json.dumps({"timings_ms": [(_RENDER_SCOPE, 1.0)]}), None),
    ],
)
def test_parse_profile_handles_unusable_timings(benchmark_renderer, tmp_path, contents, error):
    """Invalid files raise; valid files without measured frames return no result."""
    profile_path = tmp_path / "profile.json"
    if contents is not None:
        profile_path.write_text(contents)

    with pytest.raises(error) if error else nullcontext():
        assert benchmark_renderer.parse_profile(str(profile_path), num_frames=3) is None


def test_parse_profile_sums_physics_steps_within_one_frame(benchmark_renderer, tmp_path):
    """Frame boundaries group a variable number of steps and ignore an unfinished frame."""
    profile_path = tmp_path / "profile.json"
    render_ms = [2.1, 3.2, 4.3]
    physics_ms = [[0.123456789, 0.234567891, 0.345678912], [], [0.456789123]]
    timings = [(_RENDER_SCOPE, 99.0)] * benchmark_renderer.FRAME_PADDING
    for render, steps in zip(render_ms, physics_ms):
        timings.extend((_PHYSICS_SCOPE, value) for value in steps)
        timings.append(("Unreported::scope", 100.0))
        timings.append((_RENDER_SCOPE, render))
    timings.append((_PHYSICS_SCOPE, 999.0))
    _write_profile(profile_path, timings)

    frames = benchmark_renderer.parse_frames(str(profile_path))[benchmark_renderer.FRAME_PADDING :]
    results = benchmark_renderer.parse_profile(str(profile_path), num_frames=3)

    assert results["size"] == 3
    assert [frame["render"] for frame in frames] == render_ms
    assert [frame["physics"] for frame in frames] == [sum(steps) for steps in physics_ms]
    assert results["median"] == render_ms[1]
    assert results["physics"]["median"] == sum(physics_ms[2])
    assert results["total"]["mean"] == pytest.approx(
        sum(render + sum(steps) for render, steps in zip(render_ms, physics_ms)) / len(render_ms)
    )


def test_parse_profile_takes_an_arbitrary_scope_mapping(benchmark_renderer, tmp_path):
    """Scopes are data, so a caller can summarize a timer the script does not know about."""
    profile_path = tmp_path / "profile.json"
    timings = [("Custom::scope", 4.0), (_RENDER_SCOPE, 2.0)] * (benchmark_renderer.FRAME_PADDING + 3)
    _write_profile(profile_path, timings)

    results = benchmark_renderer.parse_profile(
        str(profile_path),
        num_frames=3,
        scopes={
            benchmark_renderer.FRAME_SCOPE: _RENDER_SCOPE,
            "custom": "Custom::scope",
        },
    )

    assert results["custom"]["median"] == pytest.approx(4.0)
    assert "physics" not in results


@pytest.mark.parametrize("write_timings", [False, True])
def test_run_profile_reads_fresh_structured_output(benchmark_renderer, tmp_path, monkeypatch, write_timings):
    """The run consumes its own profiling artifact; stale artifacts and timing logs cannot satisfy it."""
    profile = {"name": "p", "preset": "newton_renderer,rgb", "settings": {"tlas": "sah", "blas": "lbvh"}}
    profile_path = tmp_path / "p" / "profile_timings.json"
    profile_path.parent.mkdir()
    frames = benchmark_renderer.FRAME_PADDING + 2
    _write_profile(profile_path, [(_RENDER_SCOPE, 99.0)] * frames)
    monkeypatch.setattr(benchmark_renderer, "OUTPUT_PATH", str(tmp_path))

    def launch(cmd, **kwargs):
        assert cmd[cmd.index("--output_path") + 1] == str(profile_path.parent)
        assert "--profile_output_path" not in cmd
        assert kwargs["env"]["ISAACLAB_RENDER_PROFILE"] == "1"
        assert kwargs["env"]["ISAACLAB_PHYSICS_PROFILE"] == "1"
        assert not profile_path.exists()
        if write_timings:
            _write_profile(profile_path, [(_PHYSICS_SCOPE, 1.234567), (_RENDER_SCOPE, 2.345678)] * frames)
        return SimpleNamespace(stdout=iter([f"{_RENDER_SCOPE} took 50.00 ms\n"] * frames), returncode=0, wait=lambda: 0)

    monkeypatch.setattr(benchmark_renderer.subprocess, "Popen", launch)
    args = benchmark_renderer._build_arg_parser().parse_args(["--num_frames", "2"])

    with nullcontext() if write_timings else pytest.raises(FileNotFoundError):
        results = benchmark_renderer.run_profile(profile, args)

    assert (tmp_path / "p.log").read_text() == f"{_RENDER_SCOPE} took 50.00 ms\n" * frames
    if write_timings:
        record = benchmark_renderer.build_record(profile, results, args.num_envs, args.resolution)
        assert record["median_ms"] == 2.345678
        assert record["physics_median_ms"] == 1.234567
        assert record["total_median_ms"] == 2.345678 + 1.234567
        assert "p" in "\n".join(benchmark_renderer.format_table([record]))


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


def test_cli_reports_grouped_timings_as_json(benchmark_renderer, monkeypatch, capsys):
    """Grouped physics and total statistics are reported without contaminating JSON stdout."""
    profile = benchmark_renderer.PROFILES[0]
    stats = {key: 2.0 for key in benchmark_renderer.STAT_KEYS}
    results = {"size": 1, **stats, "physics": stats, "total": stats}
    monkeypatch.setattr(benchmark_renderer, "run_profile", lambda profile, args: results)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--json", profile["name"]])
    monkeypatch.setattr(benchmark_renderer, "log_stream", sys.stdout)

    with pytest.raises(SystemExit) as error:
        benchmark_renderer.main()

    assert error.value.code == 0
    captured = capsys.readouterr()
    record = json.loads(captured.out)["profiles"][0]
    assert record["physics_median_ms"] == results["physics"]["median"]
    assert record["total_median_ms"] == results["total"]["median"]
    assert "physics_median:" in captured.err


def test_cli_rejects_unmatched_profile_glob():
    """An unmatched profile pattern fails fast, before any profiling subprocess is launched."""
    result = _run_cli(["does-not-exist-*"])

    assert result.returncode == 1
    assert "No profile found matching: does-not-exist-*" in result.stderr
