# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test installing isaaclab_physx and running its test suite."""

from __future__ import annotations

import shutil

import pytest
from utils import UV_Mixin, find_isaaclab_root


class Test_Install_Physx(UV_Mixin):
    """Install ./isaaclab.sh -i physx and run the isaaclab_physx test suite."""

    @classmethod
    def setup_class(cls):
        # check if uv is available
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

        # check if isaacsim is importable
        # or "_isaac_sim" link is present
        try:
            import isaacsim  # noqa: F401
        except ImportError:
            print("[DEBUG] Module isaacsim is not importable")
            isaac_sim_link = find_isaaclab_root() / "_isaac_sim"
            if not isaac_sim_link.exists():
                print(f'[DEBUG] Link "{isaac_sim_link}" does not exist')
                pytest.skip("isaacsim is not importable and _isaac_sim link not found, skipping")

    @pytest.mark.uv
    @pytest.mark.gpu
    @pytest.mark.slow
    @pytest.mark.native
    @pytest.mark.timeout(3600)
    def test_install_physx_and_run_tests(self, isaaclab_root):
        """Install physx extension and run the isaaclab_physx test suite."""

        try:
            self.create_uv_env(isaaclab_root)

            # ./isaaclab.sh -i physx
            result = self.run_in_uv_env([str(self.cli_script), "-i", "physx"], cwd=isaaclab_root)
            assert result.returncode == 0, f"isaaclab -i physx failed:\n{result.stdout}\n{result.stderr}"

            # Run isaaclab_physx test suite
            test_dir = str(isaaclab_root / "source" / "isaaclab_physx" / "test")
            result = self.run_in_uv_env(
                ["python", "-m", "pytest", test_dir, "-sv", "--tb=short"],
                cwd=isaaclab_root,
            )
            output = result.stdout + result.stderr
            assert result.returncode == 0, f"isaaclab_physx tests failed (rc={result.returncode}):\n{output}"

        finally:
            self.destroy_uv_env()
