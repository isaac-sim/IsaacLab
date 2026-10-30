# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the GitHub Actions JUnit summary renderer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_junit_summary_separates_xfail_from_skip(tmp_path: Path) -> None:
    """Pytest expected failures should render as unreliable, not skipped."""
    report_path = tmp_path / "report.xml"
    report_path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="3" failures="0" errors="0" skipped="2" time="1.5">
  <testcase classname="test_rendering" name="test_passes" time="0.5"/>
  <testcase classname="test_rendering" name="test_skips" time="0.5">
    <skipped type="pytest.skip" message="Unsupported backend."/>
  </testcase>
  <testcase classname="test_rendering" name="test_known_failure" time="0.5">
    <skipped type="pytest.xfail" message="Known rendering regression (NVBUG#1234567)."/>
  </testcase>
</testsuite>
""",
        encoding="utf-8",
    )
    script_path = Path(__file__).resolve().parents[4] / ".github" / "actions" / "run-tests" / "junit_summary.py"

    result = subprocess.run(
        [sys.executable, str(script_path), str(report_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "🟢 1 PASSED" in result.stdout
    assert "🟠 1 SKIPPED" in result.stdout
    assert "🟡 1 UNRELIABLE (XFAIL)" in result.stdout
    assert "Unsupported backend." in result.stdout
    assert "Known rendering regression (NVBUG#1234567)." in result.stdout
