# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest configuration for OdinV2's offline test suite.

Puts the repository root on ``sys.path`` so ``tools.odin.*`` resolves when
pytest is invoked with ``--confcutdir`` pointed at this directory, which
bypasses the repository-level ``tools/conftest.py`` CI orchestrator.
"""

import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_GOLDEN_DIR = Path(__file__).parent / "golden"


@pytest.fixture
def assert_golden() -> Callable[[str, str], None]:
    """Compare rendered text against a committed golden file.

    Rendered artefacts are asserted whole rather than by substring, so a change
    anywhere in them shows up as a reviewable diff instead of passing unnoticed
    between the assertions.
    """

    def _assert(name: str, rendered: str) -> None:
        path = _GOLDEN_DIR / name
        if os.environ.get("ODIN_UPDATE_GOLDEN"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered)
        hint = f"re-render with ODIN_UPDATE_GOLDEN=1 and review the diff to {path}"
        assert path.exists(), f"no golden at {path}; {hint}"
        assert rendered == path.read_text(), f"rendered output differs from {path.name}; if intended, {hint}"

    return _assert
