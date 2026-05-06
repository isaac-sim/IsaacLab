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
        # Pin the easy contributor footgun: ``foo.skip.rst`` is a *patch
        # fragment* with slug ``foo.skip`` (the file extension is ``.rst``),
        # not a skip marker — ``.skip`` is its own suffix, mutually
        # exclusive with ``.rst``. Locking this in so a future "fix" can't
        # silently flip the semantics.
        ("foo.skip.rst", True, False),
        # Files that are not fragments at all.
        (".gitkeep", False, False),
        ("README.md", False, False),
        ("1234.minor", False, False),  # missing .rst extension
        ("1234.rst.bak", False, False),
        # Slugs that violate git-refname-style rules: leading ``.`` / ``-``,
        # consecutive dots, ``.lock`` ending, forbidden chars, ``/``.
        (".leading-dot.rst", False, False),
        ("-leading-dash.rst", False, False),
        ("trailing-dot..rst", False, False),  # slug ``trailing-dot.`` ends in `.`
        ("has..consecutive.rst", False, False),  # slug contains `..`
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


@pytest.mark.parametrize(
    "name,expected_slug,expected_tier",
    [
        # One representative case per tier so each branch of the SUFFIXES
        # tuple is exercised on its own.
        ("plain.rst", "plain", "patch"),
        ("with-feature.minor.rst", "with-feature", "minor"),
        ("with-break.major.rst", "with-break", "major"),
        ("ci-only.skip", "ci-only", "skip"),
        # Dotted slug carries the most-specific suffix.
        ("v1.2.3-bump.major.rst", "v1.2.3-bump", "major"),
        # Filenames that don't match any suffix yield ``(None, None)``.
        ("not-a-fragment", None, None),
        ("README.md", None, None),
    ],
)
def test_fragment_filename_slug_and_tier(name, expected_slug, expected_tier):
    fn = cli.FragmentFilename(name)
    assert fn.slug == expected_slug
    assert fn.tier == expected_tier


@pytest.mark.parametrize(
    "name,is_valid,is_fragment,is_skip",
    [
        # ``is_valid`` is true for both fragments and skip markers; only the
        # latter two flags partition the parsed names. This grid asserts
        # they're consistent for the four interesting outcomes.
        ("plain.rst", True, True, False),
        ("plain.minor.rst", True, True, False),
        ("ci-only.skip", True, False, True),
        ("not-a-fragment", False, False, False),
    ],
)
def test_fragment_filename_validity_and_kind(name, is_valid, is_fragment, is_skip):
    fn = cli.FragmentFilename(name)
    assert fn.is_valid is is_valid
    assert fn.is_fragment is is_fragment
    assert fn.is_skip is is_skip


@pytest.mark.parametrize(
    "bad_char",
    # Each forbidden char in :attr:`FragmentFilename._FORBIDDEN_CHARS` plus a
    # representative ASCII control char and the ``DEL`` sentinel — the regex
    # used to call these out via membership checks; the parser should still
    # reject them per character.
    [" ", "~", "^", ":", "?", "*", "[", "\\", "\x01", "\x7f"],
)
def test_fragment_filename_rejects_forbidden_chars(bad_char):
    fn = cli.FragmentFilename(f"slug{bad_char}with-bad-char.rst")
    assert fn.is_valid is False
    assert fn.slug is None
    assert fn.tier is None


@pytest.mark.parametrize(
    "name",
    # Edge cases that don't fit cleanly into the parametrized validity grid:
    # an empty filename, a filename that's *only* a suffix (slug would be
    # empty), and the ``@{`` substring git refnames forbid.
    [
        "",
        ".rst",
        ".minor.rst",
        ".skip",
        "has@{atbrace}.rst",
    ],
)
def test_fragment_filename_rejects_structural_edge_cases(name):
    fn = cli.FragmentFilename(name)
    assert fn.is_valid is False


def test_fragment_filename_suffixes_are_canonical():
    """``SUFFIXES`` is the wire-format contract — pin the exact tuple."""
    assert cli.FragmentFilename.SUFFIXES == (
        (".minor.rst", "minor"),
        (".major.rst", "major"),
        (".skip", "skip"),
        (".rst", "patch"),
    )


def test_fragment_filename_pattern_summary_is_derived_from_suffixes():
    """User-facing list keeps tiers in display order and ends with ``or``."""
    assert cli.FragmentFilename.pattern_summary() == ("<slug>.rst, <slug>.minor.rst, <slug>.major.rst, or <slug>.skip")


def test_fragment_filename_help_lines_format_per_tier():
    """Help lines for a missing package fragment cover every tier with aligned columns."""
    lines = cli.FragmentFilename.help_lines_for_package("isaaclab_newton")
    assert lines == [
        "add  source/isaaclab_newton/changelog.d/<slug>.rst         (patch bump)",
        "or   source/isaaclab_newton/changelog.d/<slug>.minor.rst   (minor bump)",
        "or   source/isaaclab_newton/changelog.d/<slug>.major.rst   (major bump)",
        "or   source/isaaclab_newton/changelog.d/<slug>.skip        (no entry, no bump)",
    ]


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
