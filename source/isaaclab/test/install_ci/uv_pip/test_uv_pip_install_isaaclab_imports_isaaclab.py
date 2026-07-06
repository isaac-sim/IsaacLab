# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - (wheel supplied by runner: tools/run_install_ci.py --build-wheel or --wheel <path>)
    - ./isaaclab.sh -u
    - uv pip install <wheel>
Tests:
    - python -c "import isaaclab" -> verify isaaclab importable
    - python -c "from isaaclab import __version__; print(__version__)" -> verify version matches wheel
"""

from __future__ import annotations

import shutil

import pytest
from utils import UV_Mixin


@pytest.mark.install_path_uv_pip
class Test_Uv_Pip_Install_Isaaclab_Imports_Isaaclab(UV_Mixin):
    """``uv pip install <wheel>`` with no extras, verify the base package is usable."""

    _wheel: str = ""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.fixture(autouse=True, scope="class")
    def _install_wheel(self, isaaclab_root, wheel):
        cls = self.__class__
        cls._wheel = str(wheel)

        self.create_uv_env(isaaclab_root)
        cls.env_path = self.env_path
        cls.python = self.python
        cls.cli_script = self.cli_script

        result = self.run_in_uv_env(["uv", "pip", "install", cls._wheel], cwd=isaaclab_root, timeout=900)
        assert result.returncode == 0, f"uv pip install {cls._wheel} failed:\n{result.stdout}\n{result.stderr}"

        yield

        self.destroy_uv_env()

    @pytest.mark.docker
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(900)
    def test_install_makes_isaaclab_importable(self):
        """``import isaaclab`` succeeds after ``uv pip install <wheel>``."""
        result = self.run_in_uv_env(["python", "-c", "import isaaclab"])
        assert result.returncode == 0, f"import isaaclab failed:\n{result.stdout}\n{result.stderr}"

    @pytest.mark.docker
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(900)
    def test_install_makes_isaaclab_version_match_wheel(self):
        """``isaaclab.__version__`` equals the version segment in the wheel filename."""
        result = self.run_in_uv_env(["python", "-c", "from isaaclab import __version__; print(__version__)"])
        assert result.returncode == 0, f"import __version__ failed:\n{result.stdout}\n{result.stderr}"
        imported_version = result.stdout.strip()
        expected_version = self._wheel.split("/")[-1].split("-")[1]
        assert imported_version == expected_version, (
            f"isaaclab.__version__ mismatch: expected {expected_version}, got {imported_version}"
        )
