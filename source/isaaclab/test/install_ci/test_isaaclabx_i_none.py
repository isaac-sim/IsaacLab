# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test the core install (./isaaclab.sh -i none).

``./isaaclab.sh -i none`` installs the always-on core set of submodules without
any optional submodules (mimic, teleop) or optional extra dependencies
(newton physics library, RL frameworks, visualizer backends, OV wheels).  All core
packages must be importable after this install, and training with the default physics
preset must succeed.
"""

from __future__ import annotations

import shutil

import pytest
from utils import UV_Mixin, find_isaaclab_root

# Core packages that must be importable after ``-i none``.
_CORE_PACKAGES = [
    "isaaclab",
    "isaaclab_ppisp",
    "isaaclab_assets",
    "isaaclab_contrib",
    "isaaclab_experimental",
    "isaaclab_newton",
    "isaaclab_ov",
    "isaaclab_ovphysx",
    "isaaclab_physx",
    "isaaclab_rl",
    "isaaclab_tasks",
    "isaaclab_tasks_experimental",
    "isaaclab_visualizers",
]


class Test_Install_None(UV_Mixin):
    """./isaaclab.sh -i none: core set installed, no optional extras."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

        try:
            import isaacsim  # noqa: F401
        except ImportError:
            if not (find_isaaclab_root() / "_isaac_sim").exists():
                pytest.skip("isaacsim is not importable and _isaac_sim link not found, skipping")

    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.native
    @pytest.mark.timeout(1800)
    def test_core_install_all_packages_importable(self, isaaclab_root):
        """All core packages are importable after ./isaaclab.sh -i none."""

        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", "none"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i none failed:\n{result.stdout}\n{result.stderr}"

            for pkg in _CORE_PACKAGES:
                result = self.run_in_uv_env(["python", "-c", f"import {pkg}; print('{pkg} ok')"])
                assert result.returncode == 0, f"import {pkg} failed:\n{result.stdout}\n{result.stderr}"

        finally:
            self.destroy_uv_env()

    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.native
    @pytest.mark.timeout(1800)
    def test_core_install_optional_submodules_not_installed(self, isaaclab_root):
        """Optional submodules (mimic, teleop) are absent after -i none."""

        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", "none"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i none failed:\n{result.stdout}\n{result.stderr}"

            for pkg in ("isaaclab_mimic", "isaaclab_teleop"):
                result = self.run_in_uv_env(["python", "-c", f"import {pkg}"])
                assert result.returncode != 0, f"{pkg} should not be installed after -i none"

            for pkg in ("ovrtx", "ovphysx"):
                result = self.run_in_uv_env(["python", "-c", f"import {pkg}"])
                assert result.returncode != 0, f"{pkg} should not be installed after -i none"

        finally:
            self.destroy_uv_env()

    @pytest.mark.uv
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.native
    @pytest.mark.timeout(3600)
    def test_core_install_physx_tests_pass(self, isaaclab_root):
        """isaaclab_physx tests pass after core install (physx is always in the core set)."""

        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", "none"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i none failed:\n{result.stdout}\n{result.stderr}"

            test_dir = str(isaaclab_root / "source" / "isaaclab_physx" / "test")
            result = self.run_in_uv_env(
                ["python", "-m", "pytest", test_dir, "-sv", "--tb=short"],
                cwd=isaaclab_root,
            )
            output = result.stdout + result.stderr
            assert result.returncode == 0, f"isaaclab_physx tests failed (rc={result.returncode}):\n{output}"

        finally:
            self.destroy_uv_env()
