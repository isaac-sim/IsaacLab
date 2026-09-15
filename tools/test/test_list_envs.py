# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the environment-listing script."""

import subprocess
import sys
from pathlib import Path


def test_list_envs_runs_without_launching_isaac_sim() -> None:
    """Environment discovery must work in the default Kit-less environment."""
    repository_root = Path(__file__).parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "environments" / "list_envs.py"),
            "--keyword",
            "Nonexistent-Task",
            "--show_presets",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Available Environments in Isaac Lab" in result.stdout
