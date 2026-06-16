# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for scripts/benchmarks/training.py with --rl_library rsl_rl."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_TASK = "Isaac-Cartpole-Direct"

# Top-level keys that identify a schema TrainingBundle (runtime bundle plus ``learning``).
_TRAINING_BUNDLE_KEYS = {"run", "versions", "hardware", "runtime", "resources", "learning"}


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


def test_training_rsl_rl_writes_training_bundle(tmp_path, require_isaacsim):
    sh = ROOT / "isaaclab.sh"
    cmd = [
        str(sh),
        "-p",
        "scripts/benchmarks/training.py",
        "--rl_library",
        "rsl_rl",
        "--task",
        _TASK,
        "--num_envs",
        "16",
        "--max_iterations",
        "5",
        "--seed",
        "0",
        "--output_path",
        str(tmp_path),
        "presets=newton_mjwarp",
        "--headless",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        pytest.fail(f"training.py rc={res.returncode}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}")
    data = _find_bundle(tmp_path, _TRAINING_BUNDLE_KEYS)
    assert data["schema_version"] == "1.0"
    assert data["run"]["framework"] == "rsl_rl"
    assert data["run"]["config"]["physics_backend"] == "newton_mjwarp"
    assert 1 <= data["runtime"]["iterations_completed"] <= 5
    assert data["runtime"]["total_fps"]["mean"] > 0
    assert data["learning"]["reward"]["series_per_iter"] is not None
    assert len(data["learning"]["reward"]["series_per_iter"]) >= 1
    assert data["learning"]["reward"]["final_ema"] is not None
