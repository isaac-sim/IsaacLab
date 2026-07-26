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

import subprocess
from pathlib import Path

import packages
import pytest

EXAMPLES = Path(__file__).parent / "integration"


# ---------------------------------------------------------------------------
# Filename → bump tier (one demo per tier, tested separately)
# ---------------------------------------------------------------------------


def test_patch_bump_demo_aggregates_to_patch():
    """``examples/01_patch_bump/`` has two ``.rst`` files (no suffix) → patch."""
    batch = packages.FragmentBatch.from_dir(EXAMPLES / "01_patch_bump" / "fragments")
    assert batch.invalid == []
    assert {f.name for f in batch.valid} == {
        "jdoe-fix-mass-units.rst",
        "asmith-fix-collision-margin.rst",
    }
    assert all(f.bump == "patch" for f in batch.valid)
    assert batch.aggregate_bump() == "patch"


def test_minor_bump_demo_aggregates_to_minor():
    """``examples/02_minor_bump/`` mixes patch + minor fragments → minor wins."""
    batch = packages.FragmentBatch.from_dir(EXAMPLES / "02_minor_bump" / "fragments")
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
    batch = packages.FragmentBatch.from_dir(EXAMPLES / "03_major_bump" / "fragments")
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
    assert packages.FragmentBatch._aggregate(bumps) == expected


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
        ("jdoe-fix-bug.rst", True, False),
        ("jdoe-add-feature.minor.rst", True, False),
        ("jdoe-rename-api.major.rst", True, False),
        ("jdoe-ci-only.skip", False, True),
        (".gitkeep", False, False),
        ("README.md", False, False),
        # Dotted slugs (version-bearing branch names) — accepted; the longest
        # matching tier suffix wins, so the slug keeps its embedded dots.
        ("bump-newton-1.2.0rc2.minor.rst", True, False),
        ("foo.bar.rst", True, False),  # slug = ``foo.bar``, tier = patch
        ("1234.patch.rst", True, False),  # slug = ``1234.patch``, tier = patch
        # The contributor footgun worth pinning: ``foo.skip.rst`` is a patch
        # fragment with slug ``foo.skip``, not a skip marker — ``.skip`` is
        # its own suffix, mutually exclusive with ``.rst``.
        ("foo.skip.rst", True, False),
        # Slugs violating git-refname rules: leading `.`/`-`, consecutive
        # dots, `.lock` ending, forbidden chars, `/`.
        (".leading-dot.rst", False, False),
        ("-leading-dash.rst", False, False),
        ("trailing-dot..rst", False, False),
        ("has..consecutive.rst", False, False),
        ("ends-in.lock.rst", False, False),
        ("has space.rst", False, False),
        ("has~tilde.rst", False, False),
        ("nested/path.rst", False, False),
        ("1234.minor", False, False),  # missing .rst extension
        ("1234.rst.bak", False, False),
    ],
)
def test_fragment_filename_classifies(name, is_fragment, is_skip):
    fn = packages.FragmentFilename(name)
    assert fn.is_fragment is is_fragment
    assert fn.is_skip is is_skip


def test_dotted_slug_round_trips_with_its_tier():
    """A version-bearing branch name survives as a slug.

    The motivating case: branch names routinely carry version numbers,
    and the old pattern reserved every dot for the tier suffix.
    """
    fn = packages.FragmentFilename("bump-newton-1.2.0rc2.minor.rst")
    assert fn.slug == "bump-newton-1.2.0rc2"
    assert fn.tier == "minor"


def test_user_facing_patterns_derive_from_the_suffix_list():
    """One source of truth: adding a tier updates every message at once."""
    assert packages.FragmentFilename.pattern_summary() == (
        "<slug>.rst, <slug>.minor.rst, <slug>.major.rst, or <slug>.skip"
    )
    lines = packages.FragmentFilename.help_lines_for_package("isaaclab_newton")
    assert len(lines) == len(packages.FragmentFilename.SUFFIXES)
    assert all("source/isaaclab_newton/changelog.d/" in line for line in lines)


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
    assert packages.Fragment.parse_slug(name) == expected_slug


@pytest.mark.parametrize(
    "slug",
    [
        # Accepted by git, and so by us.
        "simple",
        "with-dash",
        "with_underscore",
        "MixedCase",
        "1234",
        "bump-newton-1.2.0rc2",
        "foo.bar",
        "v1.0",
        "@",
        # Rejected by git, and so by us.
        "-leading-dash",
        ".leading-dot",
        "trailing-dot.",
        "has..dots",
        "ends.lock",
        "has space",
        "has~tilde",
        "has^caret",
        "has:colon",
        "has?question",
        "has*star",
        "has[bracket",
        "has\\backslash",
        "has@{brace",
        "",
        "..",
    ],
)
def test_slug_rules_track_git_branch_rules(slug):
    """Our slug rule must stay equal to what git accepts as a branch name.

    The convention contributors are given is "name the fragment after your
    branch", so any name git allows for a branch has to be allowed here and
    any name it refuses has to be refused. Asking git directly is what keeps
    the two from drifting — the rule is reimplemented in Python only because
    validating every fragment through a subprocess would be absurd.

    ``/`` is the one deliberate exception, covered separately below: git
    allows it in a branch, a filename cannot contain it.
    """
    ours = packages.FragmentFilename(f"{slug}.rst").is_valid
    theirs = subprocess.run(["git", "check-ref-format", "--branch", slug], capture_output=True).returncode == 0
    assert ours is theirs, f"{slug!r}: ours={ours} git={theirs}"


def test_slug_rejects_the_path_separator_git_allows():
    """The single, deliberate divergence from git's branch rules.

    ``feature/thing`` is a fine branch name and an impossible filename, so
    the documented convention is to replace ``/`` with ``-``.
    """
    assert subprocess.run(["git", "check-ref-format", "--branch", "nested/path"], capture_output=True).returncode == 0
    assert packages.FragmentFilename("nested/path.rst").is_valid is False
