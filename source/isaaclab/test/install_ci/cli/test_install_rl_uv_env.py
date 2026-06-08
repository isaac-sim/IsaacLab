# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for RL framework extra-feature installs.

Each test installs the core set + a specific RL framework via
``./isaaclab.sh -i 'rl[<framework>]'`` and then verifies that
(a) the framework is importable and (b) a short training run succeeds.

Valid selectors for the ``rl`` feature:
  - ``rsl-rl``  → rsl-rl-lib
  - ``skrl``    → skrl
  - ``sb3``     → stable-baselines3
  - ``rl-games`` → rl-games (git dep)
  - (no selector / ``all``) → all frameworks
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


class Test_Install_RL_Frameworks(UV_Mixin):
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

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(1800)
    @pytest.mark.parametrize("selector,import_pkg,_train_dir,_train_args", _RL_CONFIGS)
    def test_rl_framework_importable_after_install(self, isaaclab_root, selector, import_pkg, _train_dir, _train_args):
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

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.timeout(3600)
    def test_train_cartpole_rsl_rl(self, isaaclab_root):
        """./isaaclab.sh -i 'newton,rl[rsl-rl]' then train Isaac-Cartpole-Direct-v0 with rsl_rl."""

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
            assert result.returncode == 0, f"rsl_rl training failed (rc={result.returncode}):\n{output}"
            assert "Traceback (most recent call last):" not in output, f"rsl_rl training raised an exception:\n{output}"

        finally:
            self.destroy_uv_env()

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.timeout(3600)
    def test_train_cartpole_skrl(self, isaaclab_root):
        """./isaaclab.sh -i 'newton,rl[skrl]' then train Isaac-Cartpole-Direct-v0 with skrl."""

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
            assert result.returncode == 0, f"skrl training failed (rc={result.returncode}):\n{output}"
            assert "Traceback (most recent call last):" not in output, f"skrl training raised an exception:\n{output}"

        finally:
            self.destroy_uv_env()

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.timeout(3600)
    def test_train_cartpole_sb3(self, isaaclab_root):
        """./isaaclab.sh -i 'newton,rl[sb3]' then train Isaac-Cartpole-Direct-v0 with sb3."""

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
            assert result.returncode == 0, f"sb3 training failed (rc={result.returncode}):\n{output}"
            assert "Traceback (most recent call last):" not in output, f"sb3 training raised an exception:\n{output}"

        finally:
            self.destroy_uv_env()

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(1800)
    def test_rl_all_installs_all_frameworks(self, isaaclab_root):
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
