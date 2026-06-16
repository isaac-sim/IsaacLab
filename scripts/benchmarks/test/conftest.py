# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures for the benchmark smoke tests."""

from pathlib import Path

import pytest

# scripts/benchmarks/test/conftest.py -> repo root is parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def require_isaacsim():
    """Skip the test up-front when Isaac Sim is unavailable in this environment.

    Checks before launching the benchmark subprocess so that a genuine non-zero
    exit (broken import, unregistered task, adapter crash) is reported as a real
    failure rather than masked as "Isaac Sim unavailable".
    """
    try:
        import isaacsim  # noqa: F401
    except ImportError:
        if not (_REPO_ROOT / "_isaac_sim").exists():
            pytest.skip("isaacsim not importable and _isaac_sim link not found; skipping smoke test")
