# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - none (system Python, no uv/conda env active)
Tests:
    - ./isaaclab.sh -i -> verify isaaclab importable in system Python
"""

from __future__ import annotations

import os
import sys

import pytest
from utils import run_cmd


class Test_Cli_Install_In_Globalenv_Smoke:
    """./isaaclab.sh -i with no uv/conda env active (system Python)."""

    @classmethod
    def setup_class(cls):
        if os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX"):
            pytest.skip("test requires no active uv/conda environment")

    @pytest.mark.install_path_cli
    @pytest.mark.docker
    @pytest.mark.slow
    @pytest.mark.timeout(1800)
    def test_install_makes_isaaclab_importable(self, isaaclab_root):
        """``./isaaclab.sh -i`` succeeds and installs into the system Python."""

        cli_script = isaaclab_root / "isaaclab.sh"

        # PEP 668 requires opt-in to install into the system Python on
        # Ubuntu 24.04 (the Docker base image).
        install_env = {"PIP_BREAK_SYSTEM_PACKAGES": "1"}
        result = run_cmd([str(cli_script), "-i"], cwd=isaaclab_root, env=install_env)
        assert result.returncode == 0, f"./isaaclab.sh -i failed:\n{result.stdout}\n{result.stderr}"

        # isaaclab must be importable in the same (system) Python that ran pytest.
        result = run_cmd(
            [sys.executable, "-c", "import isaaclab; print('isaaclab ok')"],
            cwd=isaaclab_root,
        )
        assert result.returncode == 0, f"import isaaclab failed:\n{result.stdout}\n{result.stderr}"
