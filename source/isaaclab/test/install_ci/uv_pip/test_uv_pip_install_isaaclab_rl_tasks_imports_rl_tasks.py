# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - (wheel supplied by runner: tools/run_install_ci.py --build-wheel or --wheel <path>)
    - ./isaaclab.sh -u
    - uv pip install <wheel>[sb3,skrl,rsl-rl]
Tests:
    - python -c "import isaaclab_rl" -> verify isaaclab_rl importable
    - python -c "import isaaclab_tasks" -> verify isaaclab_tasks importable
    - python -c "import isaacsim" -> verify isaacsim NOT installed (extra not requested)
"""

from __future__ import annotations

import shutil

import pytest
from utils import UV_Mixin


@pytest.mark.install_path_uv_pip
class Test_Uv_Pip_Install_Isaaclab_Rl_Tasks_Imports_Rl_Tasks(UV_Mixin):
    """``uv pip install <wheel>[sb3,skrl,rsl-rl]``: verify RL imports without Isaac Sim.

    The extras are named individually on purpose. ``test_install_rl_tasks_omits_isaacsim``
    asserts Isaac Sim is absent, and the aggregate ``all`` extra carries it -- switching to
    ``[all]`` would make that assertion fail.
    """

    _wheel: str = ""
    _extras: str = "[sb3,skrl,rsl-rl]"

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.fixture(autouse=True, scope="class")
    def _install_wheel(self, isaaclab_root, wheel):
        cls = self.__class__
        cls._wheel = str(wheel)

        # Create the uv env and install the RL extras (no isaacsim, no NVIDIA flags).
        self.create_uv_env(isaaclab_root)
        cls.env_path = self.env_path
        cls.python = self.python
        cls.cli_script = self.cli_script

        target = f"{cls._wheel}{cls._extras}"
        result = self.run_in_uv_env(
            ["uv", "pip", "install", target],
            cwd=isaaclab_root,
            timeout=1200,
        )
        assert result.returncode == 0, f"uv pip install {target} failed:\n{result.stdout}\n{result.stderr}"

        yield

        self.destroy_uv_env()

    @pytest.mark.docker
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(1200)
    def test_install_rl_tasks_makes_isaaclab_rl_importable(self):
        """``import isaaclab_rl`` succeeds after ``uv pip install <wheel>[sb3,skrl,rsl-rl]``."""
        result = self.run_in_uv_env(["python", "-c", "import isaaclab_rl"])
        assert result.returncode == 0, f"import isaaclab_rl failed:\n{result.stdout}\n{result.stderr}"

    @pytest.mark.docker
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(1200)
    def test_install_rl_tasks_makes_isaaclab_tasks_importable(self):
        """``import isaaclab_tasks`` succeeds after ``uv pip install <wheel>[sb3,skrl,rsl-rl]``."""
        result = self.run_in_uv_env(["python", "-c", "import isaaclab_tasks"])
        assert result.returncode == 0, f"import isaaclab_tasks failed:\n{result.stdout}\n{result.stderr}"

    @pytest.mark.docker
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(1200)
    def test_install_rl_tasks_omits_isaacsim(self):
        """The Isaac Sim runtime is absent after installing the RL extras (isaacsim extra not requested).

        ``import isaacsim`` is not the check: the standalone importers ship in the base
        dependencies and contribute an ``isaacsim.asset`` portion, so the namespace package
        resolves without the runtime. Ask the distribution instead.
        """
        result = self.run_in_uv_env(
            ["python", "-c", "import importlib.metadata as m; m.version('isaacsim')"],
        )
        assert result.returncode != 0, f"isaacsim should not be installed by the {self._extras} extras"
