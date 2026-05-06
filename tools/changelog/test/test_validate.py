# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fragment.validate — PR-gate filename + content rules."""

from __future__ import annotations

from pathlib import Path

import cli
import pytest

FIXTURES = Path(__file__).parent


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
    err = cli.Fragment(FIXTURES / "invalid_filenames" / "multi.dot.slug.rst").validate()
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
# check_fragments — gate orchestration: immutability, slug uniqueness, and
# the "PR must add at least one fragment per touched package" rule
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
    changed = {"source/isaaclab/code.py", "source/isaaclab/changelog.d/jdoe-fix-bug.rst"}
    added = {"source/isaaclab/code.py"}  # fragment exists already; the PR only modified it
    missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate([pkg])
    assert missing == ["isaaclab"]
    invalid_map = dict(invalid)
    assert "source/isaaclab/changelog.d/jdoe-fix-bug.rst" in invalid_map
    assert "immutable" in invalid_map["source/isaaclab/changelog.d/jdoe-fix-bug.rst"]


def test_check_fragments_chain_allows_other_pr_fragment(tmp_path):
    """A chained PR (B based on A's branch, A still open) sees A's fragment in
    its diff. That should pass — both fragments have distinct slugs and B
    contributes its own fragment for the touched package."""
    pkg = _pkg_under(tmp_path, "isaaclab")
    (pkg.root / "changelog.d").mkdir()
    (pkg.root / "changelog.d" / "alice-feature-a.rst").write_text("Fixed\n^^^^^\n\n* x\n", encoding="utf-8")
    (pkg.root / "changelog.d" / "bob-feature-b.rst").write_text("Added\n^^^^^\n\n* y\n", encoding="utf-8")
    changed = {
        "source/isaaclab/code.py",
        "source/isaaclab/changelog.d/alice-feature-a.rst",  # parent PR's fragment
        "source/isaaclab/changelog.d/bob-feature-b.rst",  # this PR's own fragment
    }
    added = changed
    missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate([pkg])
    assert missing == []
    assert invalid == []


def test_check_fragments_slug_collision_with_existing(tmp_path):
    """Adding a fragment whose slug collides with one already in changelog.d/ fails."""
    pkg = _pkg_under(tmp_path, "isaaclab")
    (pkg.root / "changelog.d").mkdir()
    # Pre-existing fragment on develop with the same slug as the one this PR adds.
    (pkg.root / "changelog.d" / "jdoe-fix-bug.rst").write_text("Fixed\n^^^^^\n\n* x\n", encoding="utf-8")
    # PR adds a fresh fragment whose slug collides — different tier, same slug.
    (pkg.root / "changelog.d" / "jdoe-fix-bug.minor.rst").write_text("Added\n^^^^^\n\n* y\n", encoding="utf-8")
    changed = {"source/isaaclab/code.py", "source/isaaclab/changelog.d/jdoe-fix-bug.minor.rst"}
    added = changed
    missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate([pkg])
    invalid_map = dict(invalid)
    assert "source/isaaclab/changelog.d/jdoe-fix-bug.minor.rst" in invalid_map
    assert "collides" in invalid_map["source/isaaclab/changelog.d/jdoe-fix-bug.minor.rst"]


def test_check_fragments_collision_independent_of_iterdir_order(tmp_path, monkeypatch):
    """Regression: an added file must not be allowed to *replace* a colliding
    pre-existing fragment in the existing-slug map. The CI checkout contains
    both, and depending on filesystem iteration order the added file could
    end up as the "existing" entry, hiding the collision."""
    pkg = _pkg_under(tmp_path, "isaaclab")
    (pkg.root / "changelog.d").mkdir()
    (pkg.root / "changelog.d" / "jdoe-foo.rst").write_text("Fixed\n^^^^^\n\n* x\n", encoding="utf-8")
    (pkg.root / "changelog.d" / "jdoe-foo.minor.rst").write_text("Added\n^^^^^\n\n* y\n", encoding="utf-8")
    changed = {"source/isaaclab/code.py", "source/isaaclab/changelog.d/jdoe-foo.minor.rst"}
    added = changed

    # Force iterdir() to return the added file *last* so it would overwrite
    # the pre-existing entry in a buggy implementation. Sort with the added
    # file ranked highest, so it lands at the tail regardless of natural
    # alphabetical order.
    real_iterdir = Path.iterdir
    added_name = "jdoe-foo.minor.rst"

    def ordered_iterdir(self):
        if self == pkg.root / "changelog.d":
            return iter(sorted(real_iterdir(self), key=lambda p: (p.name == added_name, p.name)))
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", ordered_iterdir)

    missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate([pkg])
    invalid_map = dict(invalid)
    assert "source/isaaclab/changelog.d/jdoe-foo.minor.rst" in invalid_map
    assert "collides" in invalid_map["source/isaaclab/changelog.d/jdoe-foo.minor.rst"]


def test_check_fragments_slug_collision_within_pr(tmp_path):
    """Two added fragments in the same PR that share a slug (e.g. across tiers) fail."""
    pkg = _pkg_under(tmp_path, "isaaclab")
    (pkg.root / "changelog.d").mkdir()
    (pkg.root / "changelog.d" / "jdoe-fix.rst").write_text("Fixed\n^^^^^\n\n* x\n", encoding="utf-8")
    (pkg.root / "changelog.d" / "jdoe-fix.minor.rst").write_text("Added\n^^^^^\n\n* y\n", encoding="utf-8")
    changed = {
        "source/isaaclab/code.py",
        "source/isaaclab/changelog.d/jdoe-fix.rst",
        "source/isaaclab/changelog.d/jdoe-fix.minor.rst",
    }
    added = changed
    missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate([pkg])
    # One of the two is the offender; the other is the first-seen "winner".
    invalid_paths = [p for p, _ in invalid]
    assert any("jdoe-fix" in p for p in invalid_paths)
    assert any("collides" in r for _, r in invalid)


def test_check_fragments_skip_file_satisfies_requirement(tmp_path):
    """A ``<slug>.skip`` opt-out is a valid form of "PR owns a fragment for this pkg"."""
    pkg = _pkg_under(tmp_path, "isaaclab")
    (pkg.root / "changelog.d").mkdir()
    (pkg.root / "changelog.d" / "ci-only.skip").write_text("", encoding="utf-8")
    changed = {"source/isaaclab/code.py", "source/isaaclab/changelog.d/ci-only.skip"}
    added = changed
    missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate([pkg])
    assert missing == []
    assert invalid == []


def test_check_fragments_no_source_changes_means_no_required_fragment(tmp_path):
    """Pure docs / CI / changelog-tooling PRs don't trigger the requirement."""
    pkg = _pkg_under(tmp_path, "isaaclab")
    changed = {"docs/something.rst"}  # not under source/isaaclab/
    added = changed
    missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate([pkg])
    assert missing == []
    assert invalid == []


def test_check_fragments_missing_when_source_touched_without_fragment(tmp_path):
    """If the PR touches a package's source but adds no fragment, the package is missing."""
    pkg = _pkg_under(tmp_path, "isaaclab")
    changed = {"source/isaaclab/code.py"}
    added = changed
    missing, invalid = cli.PRDiff(changed=changed, added=added).evaluate([pkg])
    assert missing == ["isaaclab"]
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
