# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - (none: uv run resolves and creates the environment on first invocation)
Tests:
    - uv run --extra teleop python -c "import isaacsim, isaaclab_teleop, isaacteleop,
        isaaclab_mimic.envs"
        -> verify the teleop extra co-resolves and imports every module the teleop
           workflow scripts need
    - uv run --extra teleop isaaclab teleop run --help
        -> verify the isaaclab teleop entry point runs from that environment
"""

from __future__ import annotations

import platform
import shutil

import pytest
from utils import aarch64_isaacsim_env, run_cmd

# The documented XR teleoperation extra: an aggregate of ``isaacsim`` (Kit XR runtime) and
# ``teleop`` (Isaac Teleop plus CloudXR). The two only co-resolve because the
# ``websockets>=14.0`` override in the root pyproject relaxes isaacsim-kernel's ==12.0 pin.
_TELEOP_EXTRA = ["--extra", "teleop"]


@pytest.mark.install_path_uv_run
class Test_Uv_Run_Teleop_Imports_Isaac_Teleop:
    """``uv run --extra teleop`` resolves and imports the XR teleop stack.

    This is the positive counterpart to
    ``cli/test_cli_install_core_in_uvenv_correctness.py``, which only asserts that
    ``isaaclab_teleop`` is absent after a core install. The commands deliberately omit
    ``--frozen`` so the extras resolve from ``pyproject.toml`` rather than requiring the
    committed ``uv.lock`` to already carry them. The environment goes to a temporary
    directory via ``UV_PROJECT_ENVIRONMENT`` so the repository checkout stays clean.
    """

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")
        # The teleop extra gates isaacteleop and dex-retargeting on Linux x86_64.
        if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
            pytest.skip("Isaac Teleop is only supported on Linux x86_64")

    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(3600)
    def test_uv_run_teleop_extra_imports_the_teleop_stack(self, isaaclab_root, tmp_path):
        """Verify the teleop extra installs Isaac Sim and Isaac Teleop together."""
        result = run_cmd(
            [
                "uv",
                "run",
                *_TELEOP_EXTRA,
                "python",
                "-c",
                # isaaclab_mimic.envs and the subtask instruction UI are imported at module
                # level by record_demos.py, so ``isaaclab teleop record`` needs them present.
                "import isaacsim, isaaclab_teleop, isaacteleop, isaaclab_mimic.envs;"
                " from isaaclab_mimic.ui.instruction_display import InstructionDisplay",
            ],
            cwd=isaaclab_root,
            env={
                "UV_PROJECT_ENVIRONMENT": str(tmp_path / "venv"),
                "OMNI_KIT_ACCEPT_EULA": "yes",
                **aarch64_isaacsim_env(),
            },
            timeout=3300,
        )
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.timeout(3600)
    def test_uv_run_teleop_exposes_the_teleop_entry_point(self, isaaclab_root, tmp_path):
        """Verify ``isaaclab teleop`` runs from the teleop environment."""
        result = run_cmd(
            ["uv", "run", *_TELEOP_EXTRA, "isaaclab", "teleop", "run", "--help"],
            cwd=isaaclab_root,
            env={
                "UV_PROJECT_ENVIRONMENT": str(tmp_path / "venv"),
                "OMNI_KIT_ACCEPT_EULA": "yes",
                **aarch64_isaacsim_env(),
            },
            timeout=3300,
        )
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        assert "--cloudxr_env" in f"{result.stdout}\n{result.stderr}"
