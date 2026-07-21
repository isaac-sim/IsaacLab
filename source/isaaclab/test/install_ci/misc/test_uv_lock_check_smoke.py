# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - (none: the test only validates the committed lockfile and installs nothing)
Tests:
    - uv lock --check -> verify the committed uv.lock is up to date with pyproject.toml
"""

from __future__ import annotations

import shutil

import pytest
from utils import run_cmd


@pytest.mark.uv
@pytest.mark.smoke
class Test_Uv_Lock_Check_Smoke:
    """The committed ``uv.lock`` must stay in sync with ``pyproject.toml``.

    ``uv run`` silently re-locks when the lockfile is stale, so a stale committed
    lock would never fail the training tests in ``uv_run/``. This fast, GPU-free
    check fails in seconds instead, telling the author to commit a regenerated
    lock (``uv lock``) alongside any dependency change.
    """

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.mark.timeout(900)
    def test_uv_lock_check_reports_lockfile_current(self, isaaclab_root):
        """Verify ``uv lock --check`` passes, i.e. the lockfile matches the project."""
        if not (isaaclab_root / "uv.lock").exists():
            pytest.fail("uv.lock is missing from the repository root")
        result = run_cmd(["uv", "lock", "--check"], cwd=isaaclab_root, timeout=880)
        assert result.returncode == 0, (
            "The committed uv.lock is out of date with pyproject.toml. Regenerate it with"
            f" `uv lock` and commit the result:\n{result.stdout}\n{result.stderr}"
        )
