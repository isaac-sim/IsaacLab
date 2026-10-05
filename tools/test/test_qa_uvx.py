# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check that the QA runner reports process failures and bounds hung commands."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def qa():
    spec = importlib.util.spec_from_file_location("qa_uvx", Path(__file__).resolve().parents[1] / "qa_uvx.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_command_preserves_output_and_exit_code(qa, tmp_path):
    log = tmp_path / "failed.log"
    result = qa._run_check(
        "failure",
        [sys.executable, "-c", "print('diagnostic'); raise SystemExit(7)"],
        tmp_path,
        os.environ.copy(),
        log,
        10,
    )
    assert result["status"] == "failed"
    assert result["returncode"] == 7
    assert "diagnostic" in log.read_text()


def test_timeout_is_a_failure(qa, tmp_path):
    result = qa._run_check(
        "hang",
        [sys.executable, "-c", "import time; time.sleep(30)"],
        tmp_path,
        os.environ.copy(),
        tmp_path / "hang.log",
        0.2,
    )
    assert result["status"] == "failed"
    assert "Timed out" in result["reason"]
    assert result["seconds"] < 10


def test_cli_failure_writes_failed_report_and_returns_nonzero(qa, tmp_path, monkeypatch):
    # Python rejects uvx's options, providing a real nonzero subprocess without network access.
    monkeypatch.setattr(qa.shutil, "which", lambda name: sys.executable if name == "uvx" else None)
    output = tmp_path / "results"
    assert qa.main(["--output", str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["passed"] is False
    assert report["counts"]["failed"] >= 1
    assert any(check["name"] == "demo-list" and check["status"] == "failed" for check in report["checks"])
