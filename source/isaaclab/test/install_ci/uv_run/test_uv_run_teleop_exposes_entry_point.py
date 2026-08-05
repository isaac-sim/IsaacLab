# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - (none: uv run creates the environment from the committed uv.lock on first invocation)
Tests:
    - uv run --frozen --extra teleop isaaclab teleop run --help
        -> verify the teleop extra resolves and the isaaclab teleop entry point runs
    - uv run --frozen --extra teleop python -c "import isaacteleop, isaaclab_teleop"
        -> verify the Isaac Teleop stack is importable from the teleop environment
"""

from __future__ import annotations

import platform
import shutil

import pytest
from utils import aarch64_isaacsim_env, run_cmd

# The documented XR teleoperation extra: Isaac Sim (Kit XR runtime) plus Isaac Teleop.
_TELEOP_EXTRA = ["--frozen", "--extra", "teleop"]


@pytest.mark.install_path_uv_run
class Test_Uv_Run_Teleop_Exposes_Entry_Point:
    """``uv run --extra teleop`` resolves and exposes the teleop entry point.

    This is the positive counterpart to
    ``cli/test_cli_install_core_in_uvenv_correctness.py``, which only asserts that
    ``isaaclab_teleop`` is absent after a core install. ``--frozen`` uses the committed
    ``uv.lock`` as-is, matching the sibling ``uv_run`` tests. The environment goes to a
    temporary directory via ``UV_PROJECT_ENVIRONMENT`` so the repository checkout stays
    clean.
    """

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")
        # The teleop extra gates isaacteleop and dex-retargeting on Linux x86_64.
        if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
            pytest.skip("Isaac Teleop is only supported on Linux x86_64")

    @pytest.mark.docker
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.gpu
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

    @pytest.mark.docker
    @pytest.mark.uv
    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.timeout(3600)
    def test_uv_run_teleop_installs_the_isaac_teleop_stack(self, isaaclab_root, tmp_path):
        """Verify the Isaac Teleop packages are importable from the teleop environment.

        ``isaacsim`` is deliberately not imported: it needs a running Kit application, and
        the entry-point test above already covers the extra resolving as a whole.
        """
        result = run_cmd(
            ["uv", "run", *_TELEOP_EXTRA, "python", "-c", "import isaacteleop, isaaclab_teleop"],
            cwd=isaaclab_root,
            env={
                "UV_PROJECT_ENVIRONMENT": str(tmp_path / "venv"),
                "OMNI_KIT_ACCEPT_EULA": "yes",
                **aarch64_isaacsim_env(),
            },
            timeout=3300,
        )
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
