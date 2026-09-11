# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures for Isaac Lab tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def _find_source_checkout_root() -> Path | None:
    """Find the Isaac Lab source checkout containing this test tree, if present."""
    for parent in Path(__file__).resolve().parents:
        if (
            (parent / "isaaclab.sh").is_file()
            and (parent / "pyproject.toml").is_file()
            and (parent / "source").is_dir()
        ):
            return parent
    return None


@pytest.fixture(scope="session")
def source_checkout_root() -> Path:
    """Return the Isaac Lab source checkout root, or skip tests from installed packages."""
    root = _find_source_checkout_root()
    if root is None:
        pytest.skip("test requires an Isaac Lab source checkout")
    return root
