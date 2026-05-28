# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test uv-based installation scenarios for isaaclab."""

from __future__ import annotations

import shutil

import pytest
from utils import UV_Mixin


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
