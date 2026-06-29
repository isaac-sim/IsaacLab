# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for scripts/benchmarks/runtime.py (Newton/MJWarp = Isaac-Sim-free)."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_TASK = "Isaac-Cartpole-Direct"

# Top-level keys that identify a schema RuntimeBundle (no ``learning`` block).
_RUNTIME_BUNDLE_KEYS = {"run", "versions", "hardware", "runtime", "resources"}


def _find_bundle(out_dir: Path, expected_keys: set[str]) -> dict:
    """Return the parsed JSON whose top-level keys cover ``expected_keys``.

    The schema formatter names its file from a timestamped prefix, so the smoke
    tests glob the output directory rather than hardcode the filename.
    """
    candidates = sorted(out_dir.glob("*.json"))
    assert candidates, f"no *.json written to {out_dir}"
    for path in candidates:
        data = json.loads(path.read_text())
        if expected_keys <= set(data):
            return data
    pytest.fail(f"no bundle in {out_dir} contained keys {expected_keys}; found {[p.name for p in candidates]}")


def test_runtime_writes_runtime_bundle(tmp_path, require_isaacsim):
    sh = ROOT / "isaaclab.sh"
    cmd = [
        str(sh),
        "-p",
        "scripts/benchmarks/runtime.py",
        "--task",
        _TASK,
        "--num_envs",
        "16",
        "--num_frames",
        "20",
        "--seed",
        "0",
        "--output_path",
        str(tmp_path),
        "presets=newton_mjwarp",
        "--headless",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        pytest.fail(f"runtime.py rc={res.returncode}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}")
    data = _find_bundle(tmp_path, _RUNTIME_BUNDLE_KEYS)
    assert data["schema_version"] == "1.0"
    assert data["run"]["framework"] is None
    assert "learning" not in data
    assert data["run"]["config"]["physics_backend"] == "newton_mjwarp"
    assert data["runtime"]["iterations_completed"] == 20
    assert data["runtime"]["total_fps"]["mean"] > 0


def test_runtime_multi_formatter_writes_schema_and_omniperf(tmp_path, require_isaacsim):
    """Two formatters -> two files, suffixed ``_schema.json`` and ``_omniperf.json``."""
    sh = ROOT / "isaaclab.sh"
    cmd = [
        str(sh),
        "-p",
        "scripts/benchmarks/runtime.py",
        "--task",
        _TASK,
        "--num_envs",
        "16",
        "--num_frames",
        "20",
        "--seed",
        "0",
        "--output_path",
        str(tmp_path),
        "--benchmark_formatter",
        "schema,omniperf",
        "presets=newton_mjwarp",
        "--headless",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        pytest.fail(f"runtime.py rc={res.returncode}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}")

    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 2, f"expected exactly 2 json files, found {[p.name for p in files]}"

    schema_files = [p for p in files if p.name.endswith("_schema.json")]
    omniperf_files = [p for p in files if p.name.endswith("_omniperf.json")]
    assert len(schema_files) == 1, f"expected one *_schema.json, found {[p.name for p in files]}"
    assert len(omniperf_files) == 1, f"expected one *_omniperf.json, found {[p.name for p in files]}"

    # Schema file parses as a RuntimeBundle.
    schema_data = json.loads(schema_files[0].read_text())
    assert set(schema_data) >= _RUNTIME_BUNDLE_KEYS
    assert schema_data["run"]["framework"] is None
    assert schema_data["runtime"]["total_fps"]["mean"] > 0

    # Omniperf file is the flat KPI shape: a dict keyed by phase name.
    omniperf_data = json.loads(omniperf_files[0].read_text())
    assert "benchmark_info" in omniperf_data
    assert "runtime" in omniperf_data
    assert "run" not in omniperf_data, "omniperf output should not carry the schema-bundle shape"
