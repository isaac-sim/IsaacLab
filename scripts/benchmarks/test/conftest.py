# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures for the benchmark smoke tests."""

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def require_isaacsim():
    """Skip benchmark smoke tests when Isaac Sim is unavailable."""
    try:
        import isaacsim  # noqa: F401
    except ImportError:
        if not (_REPO_ROOT / "_isaac_sim").exists():
            pytest.skip("isaacsim not importable and _isaac_sim link not found; skipping smoke test")
