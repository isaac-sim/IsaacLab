# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - ./isaaclab.sh -u
Tests:
    - ./isaaclab.sh -i rl[rsl-rl] -> verify rsl_rl importable
    - ./isaaclab.sh -i rl[skrl] -> verify skrl importable
    - ./isaaclab.sh -i rl[sb3] -> verify stable_baselines3 importable
    - ./isaaclab.sh -i newton,rl[rsl-rl] -> verify cartpole training with rsl_rl works
    - ./isaaclab.sh -i newton,rl[skrl] -> verify cartpole training with skrl works
    - ./isaaclab.sh -i newton,rl[sb3] -> verify cartpole training with sb3 works
    - ./isaaclab.sh -i rl -> verify all RL frameworks installed
"""

from __future__ import annotations

import shutil

import pytest
from utils import UV_Mixin, find_isaaclab_root

_TRAIN_SCRIPT = "scripts/reinforcement_learning/{framework}/train.py"

# (selector, importable_package, train_script_dir, train_extra_args)
_RL_CONFIGS = [
    ("rsl-rl", "rsl_rl", "rsl_rl", ["presets=newton_mjwarp"]),
    ("skrl", "skrl", "skrl", []),
    ("sb3", "stable_baselines3", "sb3", []),
]


class Test_Cli_Install_Rl_In_Uvenv_Runs_Training(UV_Mixin):
    """./isaaclab.sh -i 'rl[<framework>]' installs the RL framework extras."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

        try:
            import isaacsim  # noqa: F401
        except ImportError:
            if not (find_isaaclab_root() / "_isaac_sim").exists():
                pytest.skip("isaacsim is not importable and _isaac_sim link not found, skipping")

    @pytest.mark.install_path_cli
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(1800)
    @pytest.mark.parametrize("selector,import_pkg,_train_dir,_train_args", _RL_CONFIGS)
    def test_install_rl_makes_framework_importable(self, isaaclab_root, selector, import_pkg, _train_dir, _train_args):
        """./isaaclab.sh -i 'rl[<selector>]' makes the framework importable."""

        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", f"rl[{selector}]"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i rl[{selector}] failed:\n{result.stdout}\n{result.stderr}"

            result = self.run_in_uv_env(["python", "-c", f"import {import_pkg}; print('{import_pkg} ok')"])
            assert result.returncode == 0, (
                f"import {import_pkg} failed after rl[{selector}]:\n{result.stdout}\n{result.stderr}"
            )

        finally:
            self.destroy_uv_env()

    @pytest.mark.install_path_cli
    @pytest.mark.uv
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.timeout(3600)
    def test_install_newton_rl_rsl_rl_trains_cartpole(self, isaaclab_root):
        """./isaaclab.sh -i 'newton,rl[rsl-rl]' then train Isaac-Cartpole-Direct with rsl_rl."""

        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", "newton,rl[rsl-rl]"], cwd=isaaclab_root)
            assert result.returncode == 0, f"install failed:\n{result.stdout}\n{result.stderr}"

            result = self.run_in_uv_env(
                [
                    str(self.cli_script),
                    "-p",
                    "scripts/reinforcement_learning/rsl_rl/train.py",
                    "--task",
                    "Isaac-Cartpole-Direct",
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
            assert result.returncode == 0, f"rsl_rl training failed (rc={result.returncode}):\n{output}"
            assert "Traceback (most recent call last):" not in output, f"rsl_rl training raised an exception:\n{output}"

        finally:
            self.destroy_uv_env()

    @pytest.mark.install_path_cli
    @pytest.mark.uv
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.timeout(3600)
    def test_install_newton_rl_skrl_trains_cartpole(self, isaaclab_root):
        """./isaaclab.sh -i 'newton,rl[skrl]' then train Isaac-Cartpole-Direct with skrl."""

        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", "newton,rl[skrl]"], cwd=isaaclab_root)
            assert result.returncode == 0, f"install failed:\n{result.stdout}\n{result.stderr}"

            result = self.run_in_uv_env(
                [
                    str(self.cli_script),
                    "-p",
                    "scripts/reinforcement_learning/skrl/train.py",
                    "--task",
                    "Isaac-Cartpole-Direct",
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
            assert result.returncode == 0, f"skrl training failed (rc={result.returncode}):\n{output}"
            assert "Traceback (most recent call last):" not in output, f"skrl training raised an exception:\n{output}"

        finally:
            self.destroy_uv_env()

    @pytest.mark.install_path_cli
    @pytest.mark.uv
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.timeout(3600)
    def test_install_newton_rl_sb3_trains_cartpole(self, isaaclab_root):
        """./isaaclab.sh -i 'newton,rl[sb3]' then train Isaac-Cartpole-Direct with sb3."""

        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", "newton,rl[sb3]"], cwd=isaaclab_root)
            assert result.returncode == 0, f"install failed:\n{result.stdout}\n{result.stderr}"

            result = self.run_in_uv_env(
                [
                    str(self.cli_script),
                    "-p",
                    "scripts/reinforcement_learning/sb3/train.py",
                    "--task",
                    "Isaac-Cartpole-Direct",
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
            assert result.returncode == 0, f"sb3 training failed (rc={result.returncode}):\n{output}"
            assert "Traceback (most recent call last):" not in output, f"sb3 training raised an exception:\n{output}"

        finally:
            self.destroy_uv_env()

    @pytest.mark.install_path_cli
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(1800)
    def test_install_rl_pulls_all_frameworks(self, isaaclab_root):
        """./isaaclab.sh -i 'rl' (no selector) installs all RL frameworks."""

        try:
            self.create_uv_env(isaaclab_root)

            result = self.run_in_uv_env([str(self.cli_script), "-i", "rl"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i rl failed:\n{result.stdout}\n{result.stderr}"

            for pkg in ("rsl_rl", "skrl", "stable_baselines3"):
                result = self.run_in_uv_env(["python", "-c", f"import {pkg}; print('{pkg} ok')"])
                assert result.returncode == 0, f"import {pkg} failed after rl[all]:\n{result.stdout}\n{result.stderr}"

        finally:
            self.destroy_uv_env()
