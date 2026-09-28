# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the runtime benchmark entry point."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_TASK = "Isaac-Cartpole-Direct"


@pytest.mark.parametrize("measure_sync_step", [False, True], ids=["default", "synchronized_breakdown"])
def test_runtime_writes_all_requested_formats(tmp_path, monkeypatch, measure_sync_step: bool):
    """Tasks without a benchmark mode write reports without collecting profiling scopes."""
    monkeypatch.setenv("ISAACLAB_RENDER_PROFILE", "1")
    monkeypatch.setenv("ISAACLAB_PHYSICS_PROFILE", "1")
    cmd = [
        sys.executable,
        "scripts/benchmarks/runtime.py",
        "--task",
        _TASK,
        "--num_envs",
        "16",
        "--num_steps",
        "20",
        "--warmup_steps",
        "0",
        "--seed",
        "0",
        "--device",
        "cpu",
        "--output_path",
        str(tmp_path),
        "--benchmark_formatter",
        "schema,omniperf",
        *(["--measure_sync_step"] if measure_sync_step else []),
        "presets=newton_mjwarp",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        pytest.fail(
            f"runtime benchmark rc={res.returncode}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}"
        )

    device_lines = [line for line in res.stdout.splitlines() if "Environment device" in line]
    assert device_lines and device_lines[-1].endswith(": cpu"), f"unexpected device output: {device_lines}"

    files = sorted(tmp_path.glob("*.json"))
    assert not (tmp_path / "profile_timings.json").exists()
    schema_files = [path for path in files if path.name.endswith("_schema.json")]
    omniperf_files = [path for path in files if path.name.endswith("_omniperf.json")]
    assert len(schema_files) == len(omniperf_files) == 1

    schema_data = json.loads(schema_files[0].read_text())
    assert schema_data["run"]["config"]["physics_backend"] == "newton_mjwarp"
    assert schema_data["runtime"]["iterations_completed"] == 20
    assert "scope_timings" not in schema_data["runtime"]
    assert schema_data["extra"] is None
    assert schema_data["runtime"]["startup_time_s"]["first_step"] > 0.0
    timing = schema_data["runtime"]["environment_step_timing"]
    assert timing["warmup_steps"] == 0
    assert timing["environment_step_calls"] == 20
    assert timing["environment_step_fps"]["mean"] > 0
    if measure_sync_step:
        assert timing["simulation_step_calls"] > 0
        assert timing["simulation_step_time_s"]["mean"] > 0.0
        assert timing["outside_simulation_step_time_s"]["mean"] >= 0.0
        assert timing["outside_simulation_step_fraction"] >= 0.0
        assert timing["measurement_mode"] == "serialized_synchronized"
    else:
        assert timing["simulation_step_calls"] is None
        assert timing["simulation_step_time_s"] is None
        assert timing["outside_simulation_step_time_s"] is None
        assert timing["outside_simulation_step_fraction"] is None
        assert timing["measurement_mode"] == "host_return"
    assert timing["environment_step_time_s"]["std"] >= 0.0
    omniperf_data = json.loads(omniperf_files[0].read_text())
    assert omniperf_data["benchmark_info"]["environment_step_measurement_mode"] == timing["measurement_mode"]
    assert omniperf_data["benchmark_info"]["environment_step_warmup_steps"] == 0
    if measure_sync_step:
        assert "Mean Serialized Diagnostic Total FPS" in omniperf_data["runtime"]
        assert "Mean Total FPS" not in omniperf_data["runtime"]
    else:
        assert "Mean Total FPS" in omniperf_data["runtime"]
        assert "Mean Serialized Diagnostic Total FPS" not in omniperf_data["runtime"]


def test_runtime_api_returns_profile_summary(tmp_path, monkeypatch):
    """The API and saved reports carry scalar summaries while raw samples stay local."""
    monkeypatch.setenv("ISAACLAB_RENDER_PROFILE", "1")
    monkeypatch.setenv("ISAACLAB_PHYSICS_PROFILE", "1")
    monkeypatch.setenv("BENCHMARK_RENDER_RESOLUTION", "64")
    script = tmp_path / "run_profile.py"
    script.write_text("""
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

from isaaclab.benchmark import BenchmarkOutputConfig, BenchmarkRuntimeRequest, run_runtime_benchmark
from isaaclab.benchmark.stepping import PHYSICS_PROFILE_SCOPE, RENDER_PROFILE_SCOPE

result = run_runtime_benchmark(BenchmarkRuntimeRequest(
    task="Isaac-RenderBenchmark-Franka-Cabinet",
    num_envs=1,
    num_steps=2,
    warmup_steps=1,
    presets=("newton_renderer", "rgb"),
    hydra_args=("env.benchmark_mode=physics_render",),
    output=BenchmarkOutputConfig(path=Path(sys.argv[1]), formatters=("schema", "omniperf")),
))
assert result.bundle.extra is not None
timings = json.loads((Path(sys.argv[1]) / "profile_timings.json").read_text())["timings_ms"]
assert len(result.output_paths) == 2
schema_path = next(path for path in result.output_paths if path.name.endswith("_schema.json"))
schema = json.loads(schema_path.read_text())
assert schema == json.loads(json.dumps(asdict(result.bundle)))
assert schema["schema_version"] == "1.4"
assert "scope_timings" not in schema["runtime"]
omniperf_path = next(path for path in result.output_paths if path.name.endswith("_omniperf.json"))
metrics = json.loads(omniperf_path.read_text())["runtime"]
expected = {}
for prefix, scope in (("physics", PHYSICS_PROFILE_SCOPE), ("render", RENDER_PROFILE_SCOPE)):
    samples = [elapsed_ms for name, elapsed_ms in timings if name == scope]
    expected.update({
        f"{prefix}_mean_ms": statistics.mean(samples),
        f"{prefix}_std_ms": statistics.stdev(samples),
        f"{prefix}_max_ms": max(samples),
        f"{prefix}_calls": len(samples),
    })
    assert metrics[f"{scope} Calls"] == expected[f"{prefix}_calls"]
    assert metrics[f"Mean {scope} Time per Call"] == expected[f"{prefix}_mean_ms"]
assert result.bundle.extra == expected
""")
    res = subprocess.run(
        [sys.executable, str(script), str(tmp_path)], cwd=ROOT, capture_output=True, text=True, timeout=900
    )
    assert res.returncode == 0, f"STDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}"
