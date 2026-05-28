# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test uv-based installation scenarios for isaaclab."""

from __future__ import annotations

import shutil

import pytest
from utils import UV_Mixin, find_isaaclab_root


def _skip_if_isaacsim_unavailable() -> None:
    """Skip the current test when isaacsim is neither importable nor symlinked at ``_isaac_sim``."""
    try:
        import isaacsim  # noqa: F401
    except ImportError:
        if not (find_isaaclab_root() / "_isaac_sim").exists():
            pytest.skip("isaacsim is not importable and _isaac_sim link not found, skipping")


class Test_UV_Env_Smoke(UV_Mixin):
    """Test ./isaaclab.x -u, then validate with some quick checks."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.timeout(10)
    def test_isaaclab_sh_uv_creates_env_with_python_312(self, isaaclab_root):
        """Run ./isaaclab.x -u and verify the created env has Python 3.12."""

        try:
            self.create_uv_env(isaaclab_root)
            # python --version
            version_output = self.run_in_uv_env(["python", "--version"]).stdout.strip()
            assert "3.12" in version_output, f"Expected Python 3.12, got: {version_output}"
        finally:
            self.destroy_uv_env()

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.timeout(200)
    def test_isaaclab_core_installs_core_including_assets(self, isaaclab_root):
        """Run ./isaaclab.x -i core and verify the core set (incl. assets) is importable.

        Under the new install model, ``isaaclab_assets`` is always installed as
        part of the core set.  Passing ``core`` installs the full core set without
        any optional submodules or extra feature dependencies.
        """

        try:
            self.create_uv_env(isaaclab_root)

            # ./isaaclab.x -i core — core set only, no optional extras
            result = self.run_in_uv_env([str(self.cli_script), "-i", "core"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i core failed:\n{result.stdout}\n{result.stderr}"

            # All core packages should be importable.
            for pkg in ("isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_physx"):
                result = self.run_in_uv_env(["python", "-c", f"import {pkg}; print('{pkg} ok')"])
                assert result.returncode == 0, f"import {pkg} failed:\n{result.stdout}\n{result.stderr}"

        finally:
            self.destroy_uv_env()

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.timeout(300)
    def test_isaaclab_newton_extra_installs_newton_sim(self, isaaclab_root):
        """Run ./isaaclab.x -i newton and verify the newton[sim] extra is installed.

        ``newton`` is an extra feature selector: it reinstalls the already-present
        core packages (``isaaclab_newton``, ``isaaclab_physx``, ``isaaclab_visualizers``)
        with their newton extras, pulling in the ``newton[sim]`` git dependency.
        """

        try:
            self.create_uv_env(isaaclab_root)

            # ./isaaclab.x -i newton — installs core + newton extras
            result = self.run_in_uv_env([str(self.cli_script), "-i", "newton"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i newton failed:\n{result.stdout}\n{result.stderr}"

            # The newton[sim] extra should make the newton package importable.
            result = self.run_in_uv_env(["python", "-c", "import newton; print('newton ok')"])
            assert result.returncode == 0, f"import newton failed:\n{result.stdout}\n{result.stderr}"

        finally:
            self.destroy_uv_env()

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(1800)
    def test_mimic_importable_after_install(self, isaaclab_root):
        """isaaclab_mimic is importable after ./isaaclab.sh -i mimic."""
        _skip_if_isaacsim_unavailable()

        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", "mimic"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i mimic failed:\n{result.stdout}\n{result.stderr}"

            result = self.run_in_uv_env(["python", "-c", "import isaaclab_mimic; print('isaaclab_mimic ok')"])
            assert result.returncode == 0, f"import isaaclab_mimic failed:\n{result.stdout}\n{result.stderr}"

        finally:
            self.destroy_uv_env()

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(1800)
    def test_mimic_not_installed_by_core(self, isaaclab_root):
        """isaaclab_mimic is absent after ./isaaclab.sh -i core (core only)."""
        _skip_if_isaacsim_unavailable()

        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", "core"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i core failed:\n{result.stdout}\n{result.stderr}"

            result = self.run_in_uv_env(["python", "-c", "import isaaclab_mimic"])
            assert result.returncode != 0, "isaaclab_mimic should not be installed after -i core"

        finally:
            self.destroy_uv_env()

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(1800)
    def test_core_still_present_after_mimic_install(self, isaaclab_root):
        """Core packages remain importable after ./isaaclab.sh -i mimic."""
        _skip_if_isaacsim_unavailable()

        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", "mimic"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i mimic failed:\n{result.stdout}\n{result.stderr}"

            for pkg in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl"):
                result = self.run_in_uv_env(["python", "-c", f"import {pkg}; print('{pkg} ok')"])
                assert result.returncode == 0, (
                    f"import {pkg} failed after mimic install:\n{result.stdout}\n{result.stderr}"
                )

        finally:
            self.destroy_uv_env()

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.timeout(3600)
    def test_install_newton_and_run_tests(self, isaaclab_root):
        """Install newton extension and run the isaaclab_newton test suite."""
        _skip_if_isaacsim_unavailable()

        try:
            self.create_uv_env(isaaclab_root)

            # ./isaaclab.sh -i newton
            result = self.run_in_uv_env([str(self.cli_script), "-i", "newton"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i newton failed:\n{result.stdout}\n{result.stderr}"

            # Run isaaclab_newton test suite
            test_dir = str(isaaclab_root / "source" / "isaaclab_newton" / "test")
            result = self.run_in_uv_env(
                ["python", "-m", "pytest", test_dir, "-sv", "--tb=short"],
                cwd=isaaclab_root,
            )
            output = result.stdout + result.stderr
            assert result.returncode == 0, f"isaaclab_newton tests failed (rc={result.returncode}):\n{output}"

        finally:
            self.destroy_uv_env()
