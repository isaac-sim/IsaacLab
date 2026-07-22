# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for kitless construction of shared environment configurations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FRESH_PROCESS_SCRIPT = Path(__file__).with_name("_env_cfgs_fresh_process.py").resolve()


def test_factories_are_kitless_in_fresh_process():
    """Every shared factory should validate without loading integration or runtime modules."""
    result = subprocess.run(
        [sys.executable, str(_FRESH_PROCESS_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    assert result.returncode == 0, output
