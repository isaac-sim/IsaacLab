# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - (none: uv run creates the environment from the committed uv.lock on first invocation)
Tests:
    - uv run --frozen --extra isaacsim --extra teleop python -c
        "import isaacsim, isaaclab_teleop, isaacteleop"
        -> verify the documented XR teleoperation extras co-resolve and import together
    - uv run --frozen --extra isaacsim --extra teleop isaaclab teleop --help
        -> verify the isaaclab teleop entry point runs from that environment
"""

from __future__ import annotations

import platform
import shutil

import pytest
from utils import aarch64_isaacsim_env, run_cmd

# The documented XR teleoperation extras. ``isaacsim`` supplies the Kit XR runtime and
# ``teleop`` supplies Isaac Teleop plus CloudXR. They only co-resolve because the
# ``websockets>=14.0`` override in the root pyproject relaxes isaacsim-kernel's ==12.0 pin.
_TELEOP_EXTRAS = ["--extra", "isaacsim", "--extra", "teleop"]


@pytest.mark.install_path_uv_run
class Test_Uv_Run_Teleop_Imports_Isaac_Teleop:
    """``uv run --extra isaacsim --extra teleop`` resolves and imports the XR teleop stack.

    This is the positive counterpart to
    ``cli/test_cli_install_core_in_uvenv_correctness.py``, which only asserts that
    ``isaaclab_teleop`` is absent after a core install. ``--frozen`` uses the committed
    ``uv.lock`` as-is; the environment goes to a temporary directory via
    ``UV_PROJECT_ENVIRONMENT`` so the repository checkout stays clean.
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
    def test_uv_run_teleop_extras_import_the_xr_stack(self, isaaclab_root, tmp_path):
        """Verify the teleop and isaacsim extras install together and import."""
        result = run_cmd(
            [
                "uv",
                "run",
                "--frozen",
                *_TELEOP_EXTRAS,
                "python",
                "-c",
                "import isaacsim, isaaclab_teleop, isaacteleop",
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
            ["uv", "run", "--frozen", *_TELEOP_EXTRAS, "isaaclab", "teleop", "--help"],
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
