# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for visualizer backend extra-feature installs.

``visualizer`` is an extra feature selector that reinstalls the already-present
``isaaclab_visualizers`` core package with specific backend extras:

  - ``./isaaclab.sh -i 'visualizer[rerun]'``  → rerun-sdk + newton[sim]
  - ``./isaaclab.sh -i 'visualizer[viser]'``  → viser + newton[sim]
  - ``./isaaclab.sh -i 'visualizer[newton]'`` → imgui-bundle + newton[sim]
  - ``./isaaclab.sh -i visualizer``            → all backends (default)

All backends also pull in the ``newton[sim]`` git dependency because the
Newton renderer underpins every visualizer implementation.
"""

from __future__ import annotations

import shutil

import pytest
from utils import UV_Mixin, find_isaaclab_root


class Test_Install_Visualizer(UV_Mixin):
    """./isaaclab.sh -i 'visualizer[<backend>]' installs the chosen visualizer extras."""

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
    def test_visualizer_rerun_backend_importable(self, isaaclab_root):
        """rerun-sdk is importable after ./isaaclab.sh -i 'visualizer[rerun]'."""

        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", "visualizer[rerun]"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i visualizer[rerun] failed:\n{result.stdout}\n{result.stderr}"

            result = self.run_in_uv_env(["python", "-c", "import rerun; print('rerun ok')"])
            assert result.returncode == 0, f"import rerun failed:\n{result.stdout}\n{result.stderr}"

        finally:
            self.destroy_uv_env()

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(1800)
    def test_visualizer_viser_backend_importable(self, isaaclab_root):
        """viser is importable after ./isaaclab.sh -i 'visualizer[viser]'."""

        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", "visualizer[viser]"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i visualizer[viser] failed:\n{result.stdout}\n{result.stderr}"

            result = self.run_in_uv_env(["python", "-c", "import viser; print('viser ok')"])
            assert result.returncode == 0, f"import viser failed:\n{result.stdout}\n{result.stderr}"

        finally:
            self.destroy_uv_env()

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(1800)
    def test_visualizer_default_installs_all_backends(self, isaaclab_root):
        """./isaaclab.sh -i visualizer (no selector) installs all visualizer backends."""

        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", "visualizer"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i visualizer failed:\n{result.stdout}\n{result.stderr}"

            for pkg in ("rerun", "viser"):
                result = self.run_in_uv_env(["python", "-c", f"import {pkg}; print('{pkg} ok')"])
                assert result.returncode == 0, (
                    f"import {pkg} failed after visualizer[all]:\n{result.stdout}\n{result.stderr}"
                )

        finally:
            self.destroy_uv_env()

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(1800)
    def test_visualizer_all_backends_pull_newton_sim(self, isaaclab_root):
        """Every visualizer backend install also provides the newton package."""

        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", "visualizer"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i visualizer failed:\n{result.stdout}\n{result.stderr}"

            result = self.run_in_uv_env(["python", "-c", "import newton; print('newton ok')"])
            assert result.returncode == 0, (
                f"import newton failed after visualizer install:\n{result.stdout}\n{result.stderr}"
            )

        finally:
            self.destroy_uv_env()

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.timeout(3600)
    def test_train_with_rerun_visualizer(self, isaaclab_root):
        """Training with --visualizer rerun works after ./isaaclab.sh -i 'newton,rl[rsl-rl],visualizer[rerun]'."""

        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env(
                [str(self.cli_script), "-i", "newton,rl[rsl-rl],visualizer[rerun]"],
                cwd=isaaclab_root,
            )
            assert result.returncode == 0, f"install failed:\n{result.stdout}\n{result.stderr}"

            result = self.run_in_uv_env(
                [
                    str(self.cli_script),
                    "-p",
                    "scripts/reinforcement_learning/rsl_rl/train.py",
                    "--task",
                    "Isaac-Cartpole-Direct-v0",
                    "--num_envs",
                    "64",
                    "presets=newton_mjwarp",
                    "--max_iterations",
                    "5",
                    "--headless",
                ],
                cwd=isaaclab_root,
            )
            output = result.stdout + result.stderr
            assert result.returncode == 0, f"training failed (rc={result.returncode}):\n{output}"
            assert "Traceback (most recent call last):" not in output, f"training raised an exception:\n{output}"

        finally:
            self.destroy_uv_env()
