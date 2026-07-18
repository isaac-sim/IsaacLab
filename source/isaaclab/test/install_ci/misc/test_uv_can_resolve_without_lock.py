# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Setup:
    - (none: the test resolves dependencies only and does not install anything)
Tests:
    - uv lock --upgrade --dry-run -> verify the full dependency graph (all extras, all
      platforms) re-resolves from the live public indexes, so the lockfile can be
      regenerated after any pyproject.toml change
"""

from __future__ import annotations

import shutil

import pytest
from utils import run_cmd


@pytest.mark.uv
@pytest.mark.smoke
class Test_Uv_Can_Resolve_Without_Lock:
    """A fresh, full-graph resolution must succeed from the public indexes.

    ``uv lock --upgrade --dry-run`` re-resolves every dependency, extra, and platform
    fork from scratch, ignoring the committed lockfile and without writing or
    installing anything. The committed lock shields ``uv run`` users from resolution,
    but a fresh resolve still happens whenever the lock must be regenerated: any
    contributor changing ``pyproject.toml`` runs ``uv lock``, and ``uv run`` silently
    re-resolves when the lock is stale. This is the check that catches a pin that
    only resolves against internal infrastructure (e.g. an unpublished
    ``ovphysx==0.5.2+head...`` build) or a marker split uv cannot satisfy.

    Complements the neighbouring checks: ``test_uv_lock_check_smoke`` validates the
    committed lock is current, the ``uv_run/`` tests validate the committed lock
    installs and trains, and ``test_uv_universal_resolution_smoke`` covers the base
    dependencies only (no extras).
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
