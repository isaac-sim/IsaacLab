# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the recovery-matrix runner's status machinery (dry-run, no GPU)."""

import json
import os
import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "run_recovery_matrix.sh"


def run_dry(tmp_path, extra_env=None):
    env = dict(os.environ, SYSID_PYTHON=sys.executable)
    env.pop("PROTOCOL_DATASET", None)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        ["bash", str(RUNNER), str(tmp_path), "--dry-run"], env=env, capture_output=True, text=True, timeout=60
    )
    return proc, tmp_path / "matrix_result.json"


def test_dry_run_writes_status(tmp_path):
    proc, status_path = run_dry(tmp_path)
    assert proc.returncode == 0, proc.stderr
    status = json.loads(status_path.read_text())
    assert status["state"] == "dry_run"
    assert "nominal_s0" in status["cells"] and "hot_s1" in status["cells"]
    assert "protocol_s0" not in status["cells"]


def test_dry_run_includes_protocol_cell(tmp_path):
    proc, status_path = run_dry(tmp_path, extra_env={"PROTOCOL_DATASET": "/some/dataset.pt"})
    assert proc.returncode == 0, proc.stderr
    status = json.loads(status_path.read_text())
    assert "protocol_s0" in status["cells"]


def test_status_updates_are_atomic_files(tmp_path):
    proc, status_path = run_dry(tmp_path)
    assert proc.returncode == 0
    assert status_path.exists()
    assert not (tmp_path / "matrix_result.json.tmp").exists()  # tmp+rename cleanup


def make_failing_stub(tmp_path):
    """Real python for status updates; hard-fails fit.py; no-ops dataset gen."""
    stub = tmp_path / "pystub"
    stub.write_text(
        "#!/bin/bash\n"
        'case "$*" in\n'
        "  *fit.py*) exit 7;;\n"
        "  *make_synthetic_dataset.py*) exit 0;;\n"
        f'  *) exec {sys.executable} "$@";;\n'
        "esac\n"
    )
    stub.chmod(0o755)
    return str(stub)


def test_failure_injection_stops_and_marks_fail(tmp_path):
    out = tmp_path / "out"
    env = dict(os.environ, SYSID_PYTHON=make_failing_stub(tmp_path))
    env.pop("PROTOCOL_DATASET", None)
    proc = subprocess.run(["bash", str(RUNNER), str(out)], env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 1
    status = json.loads((out / "matrix_result.json").read_text())
    assert status["state"] == "fail" and status["pass"] is False
    assert status["runs"]["nominal_s0"]["state"] == "fit_failed"
    assert "nominal_s1" not in status["runs"]  # stop-on-failure: later cells never ran
    assert not (out / "matrix_result.json.tmp").exists()


def test_rerun_resets_stale_status(tmp_path):
    stale = {"state": "pass", "pass": True, "runs": {"ghost": {"state": "pass"}}}
    (tmp_path / "matrix_result.json").write_text(json.dumps(stale))
    proc, status_path = run_dry(tmp_path)
    assert proc.returncode == 0
    status = json.loads(status_path.read_text())
    assert status["state"] == "dry_run" and status["pass"] is False
    assert status["runs"] == {}  # ghost entries purged
