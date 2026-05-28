# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test that isaaclab_physx is importable after a core install (./isaaclab.sh -i core).

Under the new installation model, ``isaaclab_physx`` is part of the always-installed
core set.  A plain ``./isaaclab.sh -i core`` (core only, no optional extras) is
therefore sufficient to make ``isaaclab_physx`` importable and to run its test suite.
"""

from __future__ import annotations

import shutil

import pytest
from utils import UV_Mixin, find_isaaclab_root


class Test_Install_Physx(UV_Mixin):
    """Core install (./isaaclab.sh -i core) makes isaaclab_physx importable."""

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

        try:
            import isaacsim  # noqa: F401
        except ImportError:
            isaac_sim_link = find_isaaclab_root() / "_isaac_sim"
            if not isaac_sim_link.exists():
                pytest.skip("isaacsim is not importable and _isaac_sim link not found, skipping")

    @pytest.mark.cli
    @pytest.mark.uv
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.timeout(3600)
    def test_core_install_includes_physx_and_runs_tests(self, isaaclab_root):
        """./isaaclab.sh -i core installs the core set (including physx) and tests pass."""

        try:
            self.create_uv_env(isaaclab_root)

            # Core install — physx is part of the always-installed set.
            result = self.run_in_uv_env([str(self.cli_script), "-i", "core"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i core failed:\n{result.stdout}\n{result.stderr}"

            # Verify isaaclab_physx is importable.
            result = self.run_in_uv_env(
                ["python", "-c", "import isaaclab_physx; print('isaaclab_physx ok')"],
            )
            assert result.returncode == 0, f"import isaaclab_physx failed:\n{result.stdout}\n{result.stderr}"

            # Run the isaaclab_physx test suite.
            test_dir = str(isaaclab_root / "source" / "isaaclab_physx" / "test")
            result = self.run_in_uv_env(
                ["python", "-m", "pytest", test_dir, "-sv", "--tb=short"],
                cwd=isaaclab_root,
            )
            output = result.stdout + result.stderr
            assert result.returncode == 0, f"isaaclab_physx tests failed (rc={result.returncode}):\n{output}"

        finally:
            self.destroy_uv_env()
