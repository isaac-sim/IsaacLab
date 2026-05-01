# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bump-tier inference: filename suffix → bump, and aggregating across a batch.

These tests use the worked examples under :file:`tools/changelog/examples/`
as fixtures so the same files double as human-readable demos and as
inputs the test suite verifies.
"""

from __future__ import annotations

from pathlib import Path

import cli
import pytest

EXAMPLES = Path(__file__).parent.parent.parent.parent.parent / "examples" / "changelog"


# ---------------------------------------------------------------------------
# Filename → bump tier (one demo per tier, tested separately)
# ---------------------------------------------------------------------------


def test_patch_bump_demo_aggregates_to_patch():
    """``examples/01_patch_bump/`` has two ``.rst`` files (no suffix) → patch."""
    batch = cli.FragmentBatch.from_dir(EXAMPLES / "01_patch_bump" / "fragments")
    assert batch.invalid == []
    assert {f.name for f in batch.valid} == {"8001.rst", "8002.rst"}
    assert all(f.bump == "patch" for f in batch.valid)
    assert batch.aggregate_bump() == "patch"


def test_minor_bump_demo_aggregates_to_minor():
    """``examples/02_minor_bump/`` mixes patch + minor fragments → minor wins."""
    batch = cli.FragmentBatch.from_dir(EXAMPLES / "02_minor_bump" / "fragments")
    assert batch.invalid == []
    assert {f.name for f in batch.valid} == {"8003.rst", "8004.minor.rst", "8005.minor.rst"}
    bumps = sorted(f.bump for f in batch.valid)
    assert bumps == ["minor", "minor", "patch"]
    assert batch.aggregate_bump() == "minor"


def test_major_bump_demo_aggregates_to_major():
    """``examples/03_major_bump/`` mixes patch + minor + major → major wins."""
    batch = cli.FragmentBatch.from_dir(EXAMPLES / "03_major_bump" / "fragments")
    assert batch.invalid == []
    assert {f.name for f in batch.valid} == {"8006.rst", "8007.minor.rst", "8008.major.rst"}
    bumps = sorted(f.bump for f in batch.valid)
    assert bumps == ["major", "minor", "patch"]
    assert batch.aggregate_bump() == "major"


# ---------------------------------------------------------------------------
# Pure aggregation logic (no filesystem)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bumps,expected",
    [
        ([], "patch"),
        (["patch"], "patch"),
        (["patch", "patch"], "patch"),
        (["patch", "minor"], "minor"),
        (["minor", "patch", "minor"], "minor"),
        (["patch", "minor", "major"], "major"),
        (["major", "patch"], "major"),
    ],
)
def test_aggregate_bump_logic(bumps, expected):
    assert cli.FragmentBatch._aggregate(bumps) == expected


# ---------------------------------------------------------------------------
# Filename regex — what the gate and compiler agree to accept
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,is_fragment,is_skip",
    [
        ("1234.rst", True, False),
        ("1234.minor.rst", True, False),
        ("1234.major.rst", True, False),
        ("1234.skip", False, True),
        (".gitkeep", False, False),
        ("README.md", False, False),
        ("1234.patch.rst", False, False),  # only minor/major are recognised tiers
        ("foo.rst", False, False),
        ("1234.minor", False, False),  # missing .rst extension
        ("1234.rst.bak", False, False),
    ],
)
def test_fragment_filename_regexes(name, is_fragment, is_skip):
    assert bool(cli.FRAGMENT_RE.match(name)) is is_fragment
    assert bool(cli.SKIP_RE.match(name)) is is_skip


# ---------------------------------------------------------------------------
# Fragment.pr_number — derived from filename for traceability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected_pr",
    [
        ("1234.rst", 1234),
        ("9001.minor.rst", 9001),
        ("42.major.rst", 42),
    ],
)
def test_fragment_pr_number_extracted_from_filename(tmp_path, name, expected_pr):
    p = tmp_path / name
    p.write_text("Added\n^^^^^\n\n* x\n", encoding="utf-8")
    assert cli.Fragment(p).pr_number == expected_pr
