# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Every ``isaaclab_ov`` test file must carry the module-level ``kitless`` marker.

CI runs this package only in the Kit-less image, which selects files by that marker, so an unmarked
file would silently run in no job.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.kitless


def _has_kitless_pytestmark(path: Path) -> bool:
    """Mirror ``tools/conftest.py``: a module-level ``pytestmark`` that applies ``pytest.mark.kitless``."""
    for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Attribute) and sub.attr == "kitless":
                if isinstance(sub.value, ast.Attribute) and sub.value.attr == "mark":
                    return True
    return False


def test_every_ov_test_file_runs_in_the_kitless_image():
    test_root = Path(__file__).parent
    unmarked = sorted(
        str(p.relative_to(test_root)) for p in test_root.rglob("test_*.py") if not _has_kitless_pytestmark(p)
    )
    assert not unmarked, f"add `pytestmark = pytest.mark.kitless` to: {unmarked}"
