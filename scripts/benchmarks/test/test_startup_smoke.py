# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for scripts/benchmarks/startup.py (Newton/MJWarp = Isaac-Sim-free)."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_TASK = "Isaac-Cartpole-Direct"

_EXPECTED_PHASES = {"app_launch", "python_imports", "task_config", "env_creation", "first_step"}

# Top-level keys that identify a schema StartupBundle.
_STARTUP_BUNDLE_KEYS = {"run", "phases"}


def _find_bundle(out_dir: Path, expected_keys: set[str]) -> dict:
    """Return the parsed JSON whose top-level keys cover ``expected_keys``.

    The schema backend names its file from a timestamped prefix, so the smoke
    tests glob the output directory rather than hardcode the filename.
    """
    candidates = sorted(out_dir.glob("*.json"))
    assert candidates, f"no *.json written to {out_dir}"
    for path in candidates:
        data = json.loads(path.read_text())
        if expected_keys <= set(data):
            return data
    pytest.fail(f"no bundle in {out_dir} contained keys {expected_keys}; found {[p.name for p in candidates]}")


def test_startup_writes_startup_bundle(tmp_path, require_isaacsim):
    sh = ROOT / "isaaclab.sh"
    cmd = [
        str(sh),
        "-p",
        "scripts/benchmarks/startup.py",
        "--task",
        _TASK,
        "--num_envs",
        "16",
        "--seed",
        "0",
        "--output_path",
        str(tmp_path),
        "presets=newton_mjwarp",
        "--headless",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        pytest.fail(f"startup.py rc={res.returncode}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}")

    data = _find_bundle(tmp_path, _STARTUP_BUNDLE_KEYS)

    # Top-level schema
    assert data["schema_version"] == "1.0", f"unexpected schema_version: {data['schema_version']}"
    assert data["run"]["framework"] is None, "startup bundle should have framework=null"

    # All five phases must be present
    assert set(data["phases"].keys()) == _EXPECTED_PHASES, f"unexpected phases: {set(data['phases'].keys())}"

    # Each phase must have a positive total_time_s
    for phase_name, phase in data["phases"].items():
        assert phase["total_time_s"] > 0, f"phase '{phase_name}' has total_time_s <= 0"

    # At least one phase must have top_functions with a valid 'calls' int
    has_calls = False
    for phase in data["phases"].values():
        for fn in phase.get("top_functions", []):
            if isinstance(fn.get("calls"), int):
                has_calls = True
                break
        if has_calls:
            break
    assert has_calls, "No top_functions entry with an integer 'calls' field found"

    # Config block must be present with top_n
    assert "config" in data, "StartupBundle missing 'config' field"
    assert isinstance(data["config"]["top_n"], int), "config.top_n should be an integer"
