# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fragment.validate — PR-gate filename + content rules."""

from __future__ import annotations

from pathlib import Path

import cli
import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Acceptance — well-formed fragments
# ---------------------------------------------------------------------------


def test_validate_accepts_well_formed(tmp_path):
    p = _write(tmp_path / "1234.rst", "Added\n^^^^^\n\n* Added X.\n")
    assert cli.Fragment(p).validate() is None


def test_validate_accepts_minor_suffix(tmp_path):
    p = _write(tmp_path / "1234.minor.rst", "Added\n^^^^^\n\n* Added X.\n")
    assert cli.Fragment(p).validate() is None


def test_validate_accepts_major_suffix(tmp_path):
    p = _write(tmp_path / "1234.major.rst", "Removed\n^^^^^^^\n\n* Removed X.\n")
    assert cli.Fragment(p).validate() is None


# ---------------------------------------------------------------------------
# Rejection — uses checked-in fixtures so the malformed inputs are reviewable
# ---------------------------------------------------------------------------


def test_validate_rejects_unknown_filename_from_fixture():
    err = cli.Fragment(FIXTURES / "invalid_filenames" / "weird-name.rst").validate()
    assert err is not None and "invalid filename" in err


def test_validate_rejects_unknown_bump_tier_from_fixture():
    err = cli.Fragment(FIXTURES / "invalid_filenames" / "1234.notabump.rst").validate()
    assert err is not None and "invalid filename" in err


def test_validate_rejects_empty_file_from_fixture():
    err = cli.Fragment(FIXTURES / "invalid_content" / "3001.rst").validate()
    assert err is not None and "empty" in err


def test_validate_rejects_missing_section_heading_from_fixture():
    err = cli.Fragment(FIXTURES / "invalid_content" / "3002.rst").validate()
    assert err is not None and "section" in err.lower()


def test_validate_rejects_section_without_bullets_from_fixture():
    err = cli.Fragment(FIXTURES / "invalid_content" / "3003.rst").validate()
    assert err is not None and "bullet" in err.lower()


# ---------------------------------------------------------------------------
# Fragment.parse_pr_number — extract the declared PR number from a fragment's name
# ---------------------------------------------------------------------------


def test_parse_pr_number_for_recognised_filenames():
    assert cli.Fragment.parse_pr_number("4444.rst") == 4444
    assert cli.Fragment.parse_pr_number("4444.minor.rst") == 4444
    assert cli.Fragment.parse_pr_number("4444.major.rst") == 4444
    assert cli.Fragment.parse_pr_number("4444.skip") == 4444


def test_parse_pr_number_returns_none_for_unrecognised():
    assert cli.Fragment.parse_pr_number("README.md") is None
    assert cli.Fragment.parse_pr_number(".gitkeep") is None
    assert cli.Fragment.parse_pr_number("not-a-fragment.rst") is None


# ---------------------------------------------------------------------------
# check_fragments — gate orchestration: immutability, chain tolerance, and
# the per-PR "must own a fragment" rule
# ---------------------------------------------------------------------------


def _pkg_under(tmp_path: Path, name: str) -> cli.Package:
    """Build a managed-looking Package rooted at ``tmp_path/source/<name>``."""
    root = tmp_path / "source" / name
    (root / "config").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)
    (root / "config" / "extension.toml").write_text('version = "0.0.0"\n', encoding="utf-8")
    (root / "docs" / "CHANGELOG.rst").write_text("Changelog\n---------\n\n", encoding="utf-8")
    return cli.Package(root)


def test_check_fragments_immutability_rejects_modified_fragment(tmp_path):
    """Modifying an existing fragment is forbidden — must add a new one instead."""
    pkg = _pkg_under(tmp_path, "isaaclab")
    changed = {"source/isaaclab/code.py", "source/isaaclab/changelog.d/4444.rst"}
    added = {"source/isaaclab/code.py"}  # 4444.rst exists already; the PR only modified it
    missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate(5555, [pkg])
    assert missing == ["isaaclab"]
    invalid_map = dict(invalid)
    assert "source/isaaclab/changelog.d/4444.rst" in invalid_map
    assert "immutable" in invalid_map["source/isaaclab/changelog.d/4444.rst"]


def test_check_fragments_chain_allows_other_pr_fragment(tmp_path):
    """A chained PR (B based on develop, parent A still open) sees A's fragment in
    its diff. That should not fail — A's fragment is silently tolerated as long as
    B contributes its own fragment for the touched package."""
    pkg = _pkg_under(tmp_path, "isaaclab")
    (pkg.root / "changelog.d").mkdir()
    (pkg.root / "changelog.d" / "4444.rst").write_text("Fixed\n^^^^^\n\n* x\n", encoding="utf-8")
    (pkg.root / "changelog.d" / "5555.rst").write_text("Added\n^^^^^\n\n* y\n", encoding="utf-8")
    changed = {
        "source/isaaclab/code.py",
        "source/isaaclab/changelog.d/4444.rst",  # parent PR's fragment
        "source/isaaclab/changelog.d/5555.rst",  # this PR's own fragment
    }
    added = changed - {"source/isaaclab/code.py"} | {"source/isaaclab/code.py"}  # all three are added
    missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate(5555, [pkg])
    assert missing == []
    assert invalid == []


def test_check_fragments_requires_own_fragment_when_pr_set(tmp_path):
    """If the PR touches a package but only adds someone else's fragment, fail."""
    pkg = _pkg_under(tmp_path, "isaaclab")
    (pkg.root / "changelog.d").mkdir()
    (pkg.root / "changelog.d" / "4444.rst").write_text("Fixed\n^^^^^\n\n* x\n", encoding="utf-8")
    changed = {"source/isaaclab/code.py", "source/isaaclab/changelog.d/4444.rst"}
    added = changed
    missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate(5555, [pkg])
    # Source touched, but no fragment with PR=5555 → missing.
    assert missing == ["isaaclab"]
    # The 4444.rst is tolerated (no error) since chained-PR fragments are allowed.
    assert invalid == []


def test_check_fragments_no_pr_falls_back_to_any_valid_fragment(tmp_path):
    """Without ``--pr``, any valid added fragment satisfies the requirement."""
    pkg = _pkg_under(tmp_path, "isaaclab")
    (pkg.root / "changelog.d").mkdir()
    (pkg.root / "changelog.d" / "4444.rst").write_text("Fixed\n^^^^^\n\n* x\n", encoding="utf-8")
    changed = {"source/isaaclab/code.py", "source/isaaclab/changelog.d/4444.rst"}
    added = changed
    missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate(None, [pkg])
    assert missing == []
    assert invalid == []


def test_check_fragments_skip_file_satisfies_when_pr_matches(tmp_path):
    """A `<pr>.skip` opt-out is a valid form of "PR owns a fragment for this pkg"."""
    pkg = _pkg_under(tmp_path, "isaaclab")
    (pkg.root / "changelog.d").mkdir()
    (pkg.root / "changelog.d" / "5555.skip").write_text("", encoding="utf-8")
    changed = {"source/isaaclab/code.py", "source/isaaclab/changelog.d/5555.skip"}
    added = changed
    missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate(5555, [pkg])
    assert missing == []
    assert invalid == []


def test_check_fragments_no_source_changes_means_no_required_fragment(tmp_path):
    """Pure docs / CI / changelog-tooling PRs don't trigger the requirement."""
    pkg = _pkg_under(tmp_path, "isaaclab")
    changed = {"docs/something.rst"}  # not under source/isaaclab/
    added = changed
    missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate(5555, [pkg])
    assert missing == []
    assert invalid == []


# ---------------------------------------------------------------------------
# _display_path — handles paths inside *and* outside REPO_ROOT
# ---------------------------------------------------------------------------


def test_display_path_strips_repo_root_for_internal_paths():
    """Inside-repo paths are shown relative for terse log lines."""
    p = cli.REPO_ROOT / "tools" / "changelog" / "cli.py"
    assert cli._display_path(p) == "tools/changelog/cli.py"


def test_display_path_falls_back_to_absolute_for_external(tmp_path):
    """External paths (e.g. ``--fragments-dir /tmp/foo`` outside the repo)
    used to crash on ``relative_to(REPO_ROOT)``; the helper now returns the
    absolute path in that case."""
    external = tmp_path / "external_fragments" / "1234.rst"
    external.parent.mkdir(parents=True)
    external.write_text("", encoding="utf-8")
    assert cli._display_path(external) == str(external)


# ---------------------------------------------------------------------------
# Package.compile bails on unmanaged packages instead of silently warning
# ---------------------------------------------------------------------------


def test_compile_raises_on_package_missing_changelog(tmp_path):
    """Constructing a Package directly at an unmanaged root and calling
    ``compile()`` must raise (not silently warn-and-write a stale toml)."""
    pkg_root = tmp_path / "pkg"
    (pkg_root / "config").mkdir(parents=True)
    (pkg_root / "config" / "extension.toml").write_text('version = "1.2.3"\n', encoding="utf-8")
    # No docs/CHANGELOG.rst — package is not managed.
    pkg = cli.Package(pkg_root)
    assert pkg.is_managed is False

    fragments = tmp_path / "fragments"
    fragments.mkdir()
    (fragments / "1234.rst").write_text("Fixed\n^^^^^\n\n* x\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not managed"):
        pkg.compile(fragments_dir=fragments, dry_run=True)


# ---------------------------------------------------------------------------
# cmd_compile parser guards — argparse-level errors fire as SystemExit
# ---------------------------------------------------------------------------


def _parse_compile(argv: list[str]):
    """Build the parser and parse a compile invocation. Returns (parser, args)."""
    parser = cli._build_parser()
    return parser, parser.parse_args(argv)


def test_compile_guard_version_with_all_errors():
    """``--version`` with ``--all`` is meaningless — each package has its own version."""
    parser, args = _parse_compile(["compile", "--all", "--version", "1.2.3"])
    with pytest.raises(SystemExit):
        cli.cmd_compile(args, parser)


def test_compile_guard_fragments_dir_with_all_errors():
    """``--fragments-dir`` with ``--all`` is meaningless — different dirs per package."""
    parser, args = _parse_compile(["compile", "--all", "--fragments-dir", "/tmp/x"])
    with pytest.raises(SystemExit):
        cli.cmd_compile(args, parser)


def test_compile_guard_malformed_version_errors():
    """A garbage ``--version`` value fails before any file is touched."""
    parser, args = _parse_compile(["compile", "--package", "isaaclab", "--version", "not-semver"])
    with pytest.raises(SystemExit):
        cli.cmd_compile(args, parser)


def test_compile_guard_nonexistent_package_errors():
    """A ``--package`` that doesn't exist on disk fails fast."""
    parser, args = _parse_compile(["compile", "--package", "definitely_not_a_real_package_xyz"])
    with pytest.raises(SystemExit):
        cli.cmd_compile(args, parser)


def test_compile_rejects_fragments_that_check_would_reject(tmp_path):
    """``compile`` must enforce the same content rules as ``check``.

    Regression: a fragment with a section heading but no bullet body
    used to slip past compile (parsed to ``{"Added": []}``, emitted an
    empty Added section), while check correctly rejected it. The two
    paths must agree on what a valid fragment looks like.
    """
    pkg_root = tmp_path / "pkg"
    (pkg_root / "config").mkdir(parents=True)
    (pkg_root / "docs").mkdir(parents=True)
    (pkg_root / "config" / "extension.toml").write_text('version = "1.2.3"\n', encoding="utf-8")
    (pkg_root / "docs" / "CHANGELOG.rst").write_text("Changelog\n---------\n\n", encoding="utf-8")
    pkg = cli.Package(pkg_root)

    fragments = tmp_path / "fragments"
    fragments.mkdir()
    # Header but no bullets — same shape as fixtures/invalid_content/3003.rst.
    (fragments / "1234.rst").write_text("Added\n^^^^^\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="failed content validation"):
        pkg.compile(fragments_dir=fragments, dry_run=True)
