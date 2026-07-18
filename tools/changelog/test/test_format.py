# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FragmentBatch._merge_sections + ._format_entry + Version.bumped — the rendering pipeline."""

from __future__ import annotations

import cli
import pytest

# ---------------------------------------------------------------------------
# merge_fragments — collapses bullets across fragments under the same section
# ---------------------------------------------------------------------------


def test_merge_fragments_collapses_same_section_across_fragments():
    f1 = {"Added": ["* a1\n"]}
    f2 = {"Added": ["* a2\n"], "Fixed": ["* f1\n"]}
    merged = cli.FragmentBatch._merge_sections([f1, f2])
    # Bullets from separate fragments concatenate with no blank line in between
    # (matching IsaacLab's repo convention, where successive bullets are run-on).
    assert merged["Added"] == ["* a1\n", "* a2\n"]
    assert merged["Fixed"] == ["* f1\n"]


# ---------------------------------------------------------------------------
# format_entry — section ordering + version heading
# ---------------------------------------------------------------------------


def test_format_entry_orders_canonical_sections():
    sections = {
        "Fixed": ["* f1\n"],
        "Added": ["* a1\n"],
        "Removed": ["* r1\n"],
    }
    out = cli.FragmentBatch._format_entry("1.2.4", sections)
    # Canonical order is Added, Changed, Deprecated, Removed, Fixed.
    a_pos = out.index("Added")
    r_pos = out.index("Removed")
    f_pos = out.index("Fixed")
    assert a_pos < r_pos < f_pos


def test_format_entry_includes_version_heading():
    out = cli.FragmentBatch._format_entry("9.9.9", {"Added": ["* x\n"]})
    assert "9.9.9 (" in out
    assert "~~~~~~" in out  # tilde underline


def test_format_entry_unknown_sections_appear_after_canonical():
    sections = {"Performance": ["* p1\n"], "Added": ["* a1\n"]}
    out = cli.FragmentBatch._format_entry("1.0.0", sections)
    assert out.index("Added") < out.index("Performance")


# ---------------------------------------------------------------------------
# bump_version — semver maths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "current,part,expected",
    [
        ("1.2.3", "patch", "1.2.4"),
        ("1.2.3", "minor", "1.3.0"),  # minor bump zeros patch
        ("1.2.3", "major", "2.0.0"),  # major bump zeros minor and patch
        ("4.6.21", "patch", "4.6.22"),
        ("4.6.21.dev20260301", "patch", "4.6.22"),  # dev suffix stripped
    ],
)
def test_version_bumped(current, part, expected):
    assert cli.Version(current).bumped(part).text == expected
    assert str(cli.Version(current).bumped(part)) == expected


def test_version_bumped_rejects_non_semver():
    # Construction itself rejects malformed input — fail-fast for bad ``--version``.
    with pytest.raises(ValueError):
        cli.Version("1.2")
    with pytest.raises(ValueError):
        cli.Version("not-semver")
    with pytest.raises(ValueError):
        cli.Version("1.2.3.4.5")


def test_version_accepts_dev_suffix():
    """PEP 440 ``.devN`` suffixes are tolerated on construction (they appear in
    real ``extension.toml`` files between releases) and stripped on bump."""
    v = cli.Version("4.6.21.dev20260301")
    assert v.text == "4.6.21.dev20260301"
    assert v.bumped("patch").text == "4.6.22"
