# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for scripts/benchmarks/play.py with --rl_library skrl.

Trains a short cartpole run to produce a checkpoint, then plays that checkpoint
through the benchmark and asserts the emitted ``PlayBundle``.
"""

import json
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_TASK = "Isaac-Cartpole-Direct"

# Top-level keys that identify a schema PlayBundle (runtime bundle, no ``learning``).
_PLAY_BUNDLE_KEYS = {"run", "versions", "hardware", "runtime", "resources"}


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


def _newest_checkpoint(pattern: str, since: float) -> Path:
    """Return the newest file matching ``pattern`` under the repo created at or after ``since``.

    Args:
        pattern: ``Path.rglob`` glob, relative to the repo root, for the backend's
            checkpoint files.
        since: ``time.time()`` recorded before the training run; only files whose
            mtime is at or after this are considered, so a stale checkpoint from a
            prior run is never selected.
    """
    matches = [p for p in ROOT.rglob(pattern) if p.stat().st_mtime >= since]
    if not matches:
        found = [str(p) for p in ROOT.rglob(pattern)]
        pytest.fail(f"no checkpoint matching {pattern!r} created since {since}; found existing: {found}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def test_play_skrl_emits_play_bundle(tmp_path, require_isaacsim):
    sh = ROOT / "isaaclab.sh"
    train_out = tmp_path / "train"
    play_out = tmp_path / "play"

    start = time.time()
    train_cmd = [
        str(sh),
        "-p",
        "scripts/benchmarks/training.py",
        "--rl_library",
        "skrl",
        "--task",
        _TASK,
        "--num_envs",
        "16",
        "--max_iterations",
        "20",
        "presets=newton_mjwarp",
        "--headless",
        "--output_path",
        str(train_out),
    ]
    res = subprocess.run(train_cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        pytest.fail(f"training.py rc={res.returncode}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}")

    ckpt = _newest_checkpoint("logs/skrl/**/checkpoints/*.pt", start)

    play_cmd = [
        str(sh),
        "-p",
        "scripts/benchmarks/play.py",
        "--rl_library",
        "skrl",
        "--task",
        _TASK,
        "--num_envs",
        "16",
        "--num_frames",
        "250",
        "--checkpoint",
        str(ckpt),
        "presets=newton_mjwarp",
        "--headless",
        "--output_path",
        str(play_out),
    ]
    res = subprocess.run(play_cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        pytest.fail(f"play.py rc={res.returncode}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}")

    data = _find_bundle(play_out, _PLAY_BUNDLE_KEYS)
    assert data["run"]["framework"] == "skrl"
    assert data["runtime"]["total_fps"]["mean"] > 0
    assert data["checkpoint_path"]
    assert data["reward"] is not None
    assert "mean" in data["reward"]
