# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - (none: the test resolves dependencies only and does not install anything)
Tests:
    - uv lock --upgrade --dry-run -> verify the full dependency graph (all extras, all
      platforms) still resolves from the live public indexes without the committed
      lockfile, so an unresolvable dependency can never hide behind the current lock
"""

from __future__ import annotations

import shutil

import pytest
from utils import run_cmd


@pytest.mark.uv
@pytest.mark.smoke
class Test_Uv_Can_Resolve_Without_Lock:
    """Unresolvable dependencies must never hide behind the committed lockfile.

    Because everyone installs from the committed ``uv.lock``, the project can keep
    working long after its dependency graph has stopped being resolvable from the
    public indexes (a pin that only exists on internal infrastructure, like the old
    unpublished ``ovphysx==0.5.2+head...`` build, or an unsatisfiable marker split).
    Such a break would only surface later, when someone edits ``pyproject.toml`` and
    needs to regenerate the lock. ``uv lock --upgrade --dry-run`` re-resolves the
    full graph while ignoring the lockfile - and without writing or installing
    anything - so a resolution break hidden by the current lock fails CI now, not at
    the next dependency change.

    Complements the neighbouring checks: the ``uv_run/`` tests validate the committed
    lock installs and trains.
    """

    @classmethod
    def setup_class(cls):
        if not shutil.which("uv"):
            pytest.skip("uv is not available")

    @pytest.mark.timeout(900)
    def test_uv_lock_upgrade_resolves_all_extras_from_public_indexes(self, isaaclab_root):
        """Verify ``uv lock --upgrade --dry-run`` resolves the full graph fresh."""
        result = run_cmd(["uv", "lock", "--upgrade", "--dry-run"], cwd=isaaclab_root, timeout=880)
        assert result.returncode == 0, (
            "Fresh full-graph resolution failed: the lockfile can no longer be regenerated"
            " (`uv lock` after a pyproject.toml change will fail, and `uv run` will fail on"
            " a stale lock). A dependency, often in an optional extra, no longer resolves"
            f" from the public indexes, or a marker split is unsatisfiable:\n{result.stdout}\n{result.stderr}"
        )
