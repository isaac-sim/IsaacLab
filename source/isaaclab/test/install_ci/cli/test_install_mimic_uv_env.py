# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test the optional mimic submodule install (./isaaclab.sh -i mimic).

``mimic`` is an optional submodule — it is not part of the always-installed
core set because its base dependencies (ipywidgets, h5py) are heavier than
the rest.  Users who need imitation-learning workflows explicitly opt in with
``./isaaclab.sh -i mimic``.
"""

from __future__ import annotations

import shutil

import pytest
from utils import UV_Mixin, find_isaaclab_root


class Test_Install_Mimic(UV_Mixin):
    """./isaaclab.sh -i mimic: installs core + isaaclab_mimic."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

        try:
            import isaacsim  # noqa: F401
        except ImportError:
            if not (find_isaaclab_root() / "_isaac_sim").exists():
                pytest.skip("isaacsim is not importable and _isaac_sim link not found, skipping")

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(1800)
    def test_mimic_importable_after_install(self, isaaclab_root):
        """isaaclab_mimic is importable after ./isaaclab.sh -i mimic."""

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
