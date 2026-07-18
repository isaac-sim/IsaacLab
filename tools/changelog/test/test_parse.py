# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fragment.parse + FragmentBatch.from_dir + Package.discover — directory scanning."""

from __future__ import annotations

from pathlib import Path

import cli

FIXTURES = Path(__file__).parent


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# parse_fragment — section header detection (pure function)
# ---------------------------------------------------------------------------


def test_parse_fragment_single_section(tmp_path):
    p = _write(tmp_path / "1.rst", "Added\n^^^^^\n\n* Added :class:`~pkg.Foo`.\n")
    sections = cli.Fragment(p).parse()
    assert list(sections.keys()) == ["Added"]
    assert sections["Added"] == ["* Added :class:`~pkg.Foo`.\n"]


def test_parse_fragment_multiple_sections_preserves_dict_order(tmp_path):
    p = _write(tmp_path / "1.rst", "Added\n^^^^^\n\n* a1\n\nFixed\n^^^^^\n\n* f1\n* f2\n")
    sections = cli.Fragment(p).parse()
    assert list(sections.keys()) == ["Added", "Fixed"]
    assert sections["Added"] == ["* a1\n"]
    assert sections["Fixed"] == ["* f1\n", "* f2\n"]


def test_parse_fragment_underline_must_be_at_least_heading_length(tmp_path):
    """Heading 'Added' (5 chars) needs >=5 carets; '^^' must not match."""
    p = _write(tmp_path / "1.rst", "Added\n^^\n\n* a1\n")
    assert cli.Fragment(p).parse() == {}


def test_parse_fragment_empty_file(tmp_path):
    p = _write(tmp_path / "1.rst", "")
    assert cli.Fragment(p).parse() == {}


def test_parse_fragment_no_section_headings(tmp_path):
    p = _write(tmp_path / "1.rst", "Just a free-form note with no headings.\n")
    assert cli.Fragment(p).parse() == {}


# ---------------------------------------------------------------------------
# Fragment.parse — same logic, exposed as a method on the wrapper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FragmentBatch.from_dir — separates valid filenames from the rest
# ---------------------------------------------------------------------------


def test_fragment_batch_flags_invalid_filenames_from_fixture():
    """Files with dotted slugs or unknown bump tiers go in ``invalid``."""
    batch = cli.FragmentBatch.from_dir(FIXTURES / "invalid_filenames")
    assert batch.valid == []
    assert {p.name for p in batch.invalid} == {"multi.dot.slug.rst", "1234.notabump.rst"}


def test_fragment_batch_missing_directory(tmp_path):
    """A non-existent directory is treated as empty, not an error."""
    batch = cli.FragmentBatch.from_dir(tmp_path / "does-not-exist")
    assert batch.valid == []
    assert batch.invalid == []
    assert batch.skip_paths == []


def test_fragment_batch_collects_skip_files_separately(tmp_path):
    """``.skip`` files are tolerated — exposed via ``skip_paths``, not ``valid``."""
    (tmp_path / "1234.skip").write_text("", encoding="utf-8")
    (tmp_path / "1235.rst").write_text("Added\n^^^^^\n\n* x\n", encoding="utf-8")
    batch = cli.FragmentBatch.from_dir(tmp_path)
    assert {f.name for f in batch.valid} == {"1235.rst"}
    assert {p.name for p in batch.skip_paths} == {"1234.skip"}


# ---------------------------------------------------------------------------
# Package.discover — a package is "managed" iff it has both
# config/extension.toml and docs/CHANGELOG.rst
# ---------------------------------------------------------------------------


def _make_pkg(root: Path, name: str, *, has_ext: bool = True, has_changelog: bool = True) -> None:
    pkg = root / name
    if has_ext:
        (pkg / "config").mkdir(parents=True, exist_ok=True)
        (pkg / "config" / "extension.toml").write_text('version = "0.0.0"\n', encoding="utf-8")
    if has_changelog:
        (pkg / "docs").mkdir(parents=True, exist_ok=True)
        (pkg / "docs" / "CHANGELOG.rst").write_text("Changelog\n---------\n\n", encoding="utf-8")


def test_package_discover_includes_complete_packages(tmp_path):
    _make_pkg(tmp_path, "complete_a")
    _make_pkg(tmp_path, "complete_b")
    pkgs = cli.Package.discover(tmp_path)
    assert [p.name for p in pkgs] == ["complete_a", "complete_b"]
    assert all(p.is_managed for p in pkgs)


def test_package_discover_excludes_packages_missing_changelog(tmp_path):
    _make_pkg(tmp_path, "complete")
    _make_pkg(tmp_path, "no_changelog", has_changelog=False)
    assert [p.name for p in cli.Package.discover(tmp_path)] == ["complete"]


def test_package_discover_excludes_packages_missing_extension_toml(tmp_path):
    _make_pkg(tmp_path, "complete")
    _make_pkg(tmp_path, "no_extension", has_ext=False)
    assert [p.name for p in cli.Package.discover(tmp_path)] == ["complete"]


def test_package_discover_returns_sorted_alphabetically(tmp_path):
    _make_pkg(tmp_path, "zebra")
    _make_pkg(tmp_path, "alpha")
    _make_pkg(tmp_path, "mango")
    assert [p.name for p in cli.Package.discover(tmp_path)] == ["alpha", "mango", "zebra"]


def test_package_discover_missing_root_returns_empty(tmp_path):
    assert cli.Package.discover(tmp_path / "does-not-exist") == []


def test_package_is_managed_property(tmp_path):
    _make_pkg(tmp_path, "complete")
    _make_pkg(tmp_path, "no_changelog", has_changelog=False)
    assert cli.Package(tmp_path / "complete").is_managed is True
    assert cli.Package(tmp_path / "no_changelog").is_managed is False
