# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify package import boundaries in a fresh interpreter."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.kitless


def test_geometry_imports_without_research_or_rtx_runtime() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("_check_imports.py"))],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Standalone import boundary passed" in result.stdout
