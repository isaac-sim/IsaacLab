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

EXAMPLES = Path(__file__).parent / "integration"


# ---------------------------------------------------------------------------
# Filename → bump tier (one demo per tier, tested separately)
# ---------------------------------------------------------------------------


def test_patch_bump_demo_aggregates_to_patch():
    """``examples/01_patch_bump/`` has two ``.rst`` files (no suffix) → patch."""
    batch = cli.FragmentBatch.from_dir(EXAMPLES / "01_patch_bump" / "fragments")
    assert batch.invalid == []
    assert {f.name for f in batch.valid} == {
        "jdoe-fix-mass-units.rst",
        "asmith-fix-collision-margin.rst",
    }
    assert all(f.bump == "patch" for f in batch.valid)
    assert batch.aggregate_bump() == "patch"


def test_minor_bump_demo_aggregates_to_minor():
    """``examples/02_minor_bump/`` mixes patch + minor fragments → minor wins."""
    batch = cli.FragmentBatch.from_dir(EXAMPLES / "02_minor_bump" / "fragments")
    assert batch.invalid == []
    assert {f.name for f in batch.valid} == {
        "jdoe-fix-rotation-frame.rst",
        "asmith-add-multi-asset-spawner.minor.rst",
        "blee-add-camera-output-contract.minor.rst",
    }
    bumps = sorted(f.bump for f in batch.valid)
    assert bumps == ["minor", "minor", "patch"]
    assert batch.aggregate_bump() == "minor"


def test_major_bump_demo_aggregates_to_major():
    """``examples/03_major_bump/`` mixes patch + minor + major → major wins."""
    batch = cli.FragmentBatch.from_dir(EXAMPLES / "03_major_bump" / "fragments")
    assert batch.invalid == []
    assert {f.name for f in batch.valid} == {
        "jdoe-fix-articulation-state.rst",
        "asmith-add-warp-contact-stream.minor.rst",
        "blee-rename-articulation-api.major.rst",
    }
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
        # Plain slugs and the three tier suffixes.
        ("1234.rst", True, False),
        ("1234.minor.rst", True, False),
        ("1234.major.rst", True, False),
        ("1234.skip", False, True),
        ("jdoe-fix-bug.rst", True, False),
        ("jdoe-add-feature.minor.rst", True, False),
        ("jdoe-rename-api.major.rst", True, False),
        ("jdoe-ci-only.skip", False, True),
        # Dotted slugs (version-bearing branch names) — accepted; the longest
        # matching tier suffix wins, so the slug keeps its embedded dots.
        ("bump-newton-1.2.0rc2.minor.rst", True, False),
        ("foo.bar.rst", True, False),  # slug = ``foo.bar``, tier = patch
        ("1234.patch.rst", True, False),  # slug = ``1234.patch``, tier = patch
        # Files that are not fragments at all.
        (".gitkeep", False, False),
        ("README.md", False, False),
        ("1234.minor", False, False),  # missing .rst extension
        ("1234.rst.bak", False, False),
        # Slugs that violate git-refname-style rules: leading ``.`` / ``-``,
        # consecutive dots, ``.lock`` ending, forbidden chars, ``/``.
        (".leading-dot.rst", False, False),
        ("-leading-dash.rst", False, False),
        ("trailing-dot..rst", False, False),  # `..` not allowed
        ("ends-in.lock.rst", False, False),
        ("has space.rst", False, False),
        ("has~tilde.rst", False, False),
        ("has^caret.rst", False, False),
        ("nested/path.rst", False, False),
    ],
)
def test_fragment_filename_classifies(name, is_fragment, is_skip):
    fn = cli.FragmentFilename(name)
    assert fn.is_fragment is is_fragment
    assert fn.is_skip is is_skip


def test_fragment_filename_extracts_dotted_slug_and_tier():
    """Slugs with dots round-trip when paired with a tier suffix."""
    fn = cli.FragmentFilename("bump-newton-1.2.0rc2.minor.rst")
    assert fn.slug == "bump-newton-1.2.0rc2"
    assert fn.tier == "minor"


# ---------------------------------------------------------------------------
# Fragment.parse_slug — derived from filename for collision detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected_slug",
    [
        ("1234.rst", "1234"),
        ("jdoe-add-feature.minor.rst", "jdoe-add-feature"),
        ("blee-rename-api.major.rst", "blee-rename-api"),
        ("ci-only.skip", "ci-only"),
        ("README.md", None),
        (".gitkeep", None),
    ],
)
def test_parse_slug_for_filenames(name, expected_slug):
    assert cli.Fragment.parse_slug(name) == expected_slug
