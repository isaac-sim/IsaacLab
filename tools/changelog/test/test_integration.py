# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end checks: run the compiler against each worked example and verify
the resulting changelog matches the checked-in :file:`changelog_after.rst`.

This is what makes the examples *living docs* — if anything in the compile
pipeline drifts, an example's ``changelog_after.rst`` stops matching and
the corresponding test fails immediately.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import cli
import pytest

EXAMPLES = Path(__file__).parent / "integration"

# Strip the ``(YYYY-MM-DD)`` suffix from version headings so the fixed example
# files don't drift when the compiler stamps today's date.
_DATE_RE = re.compile(r"\(\d{4}-\d{2}-\d{2}\)")


def _normalize(text: str) -> str:
    return _DATE_RE.sub("(YYYY-MM-DD)", text)


@pytest.mark.parametrize(
    "demo,expected_version",
    [
        ("01_patch_bump", "1.2.4"),
        ("02_minor_bump", "1.3.0"),
        ("03_major_bump", "2.0.0"),
    ],
)
def test_demo_compile_matches_changelog_after(tmp_path, demo, expected_version):
    """Stage a fake package whose CHANGELOG.rst matches the demo's ``before``,
    run the compiler against the demo's fragments, and verify the file ends
    up byte-equal to the demo's ``after`` (modulo today's date)."""
    demo_dir = EXAMPLES / demo

    # Build a minimal package layout the compiler will accept.
    pkg_root = tmp_path / "demo_pkg"
    (pkg_root / "config").mkdir(parents=True)
    (pkg_root / "docs").mkdir(parents=True)
    (pkg_root / "config" / "extension.toml").write_text('version = "1.2.3"\n', encoding="utf-8")
    shutil.copy(demo_dir / "changelog_before.rst", pkg_root / "docs" / "CHANGELOG.rst")

    # Copy fragments into tmp_path so the compile's auto-clean doesn't
    # delete the live checked-in examples directory.
    fragments_tmp = tmp_path / "fragments"
    shutil.copytree(demo_dir / "fragments", fragments_tmp)

    # Run the compiler against the (copied) fragments.
    pkg = cli.Package(pkg_root)
    pkg.compile(fragments_dir=fragments_tmp)

    actual = (pkg_root / "docs" / "CHANGELOG.rst").read_text(encoding="utf-8")
    expected = (demo_dir / "changelog_after.rst").read_text(encoding="utf-8")
    assert _normalize(actual) == _normalize(expected)

    # Version should have bumped exactly as the demo name suggests.
    assert str(pkg.current_version()) == expected_version
